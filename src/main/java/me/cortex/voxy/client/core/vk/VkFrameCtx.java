package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkEventCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkSubmitInfo;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

public final class VkFrameCtx {
    public interface FrameRetireListener {
        void onFramesRetired(long retiredUpToInclusive);
    }

    private final VulkanContext ctx;
    private VkCommandBuffer frameCmd;
    private VkCommandBuffer immediateCmd;
    private boolean anyWorkThisFrame;

    private long frameCounter = 0;
    private long retiredCounter = -1;

    private final Deque<InFlightFrame> inFlight = new ArrayDeque<>();
    private final ArrayList<Long> eventPool = new ArrayList<>();
    private final ArrayList<PendingDestroy> pendingDestroys = new ArrayList<>();
    private final ArrayList<PendingPipelineDestroy> pendingPipelineDestroys = new ArrayList<>();
    private final ArrayList<FrameRetireListener> retireListeners = new ArrayList<>();

    private record InFlightFrame(long frameIdx, long event) {}
    private record PendingDestroy(long frameIdx, long buffer, long image, long imageView, long memory) {}
    private record PendingPipelineDestroy(long frameIdx, long pipeline, long pipelineLayout, long[] modules) {}

    public VkFrameCtx(VulkanContext ctx) {
        this.ctx = ctx;
    }

    public VulkanContext vk() {
        return this.ctx;
    }

    public void addRetireListener(FrameRetireListener listener) {
        this.retireListeners.add(listener);
    }

    public long currentFrame() {
        return this.frameCounter;
    }

    public long retiredFrame() {
        return this.retiredCounter;
    }

    public int inFlightFrameCount() {
        return this.inFlight.size();
    }

    public int pendingDestroyCount() {
        return this.pendingDestroys.size() + this.pendingPipelineDestroys.size();
    }

    public int pooledEventCount() {
        return this.eventPool.size();
    }

    public void beginFrame(VkCommandBuffer mcFrameCommandBuffer) {
        if (this.frameCmd != null) throw new IllegalStateException("Frame already begun");
        this.frameCmd = mcFrameCommandBuffer;
    }

    public void endFrame() {
        if (this.frameCmd == null) throw new IllegalStateException("No frame begun");
        if (this.anyWorkThisFrame) {
            long event = this.obtainEvent();
            vkCmdSetEvent(this.frameCmd, event, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            this.inFlight.add(new InFlightFrame(this.frameCounter, event));
            this.frameCounter++;
            this.anyWorkThisFrame = false;
        }
        this.frameCmd = null;
    }

    public VkCommandBuffer cmd() {
        this.anyWorkThisFrame = true;
        if (this.frameCmd != null) return this.frameCmd;
        if (this.immediateCmd == null) {
            VkCommandBuffer allocated = null;
            try (MemoryStack stack = stackPush()) {
                var cbai = VkCommandBufferAllocateInfo.calloc(stack).sType$Default()
                        .commandPool(this.ctx.commandPool)
                        .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                        .commandBufferCount(1);
                var pCmd = stack.mallocPointer(1);
                check(vkAllocateCommandBuffers(this.ctx.device, cbai, pCmd), "vkAllocateCommandBuffers(immediate)");
                allocated = new VkCommandBuffer(pCmd.get(0), this.ctx.device);
                var begin = VkCommandBufferBeginInfo.calloc(stack).sType$Default()
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
                check(vkBeginCommandBuffer(allocated, begin), "vkBeginCommandBuffer(immediate)");
                this.immediateCmd = allocated;
            } catch (RuntimeException | Error failure) {
                if (allocated != null) {
                    vkFreeCommandBuffers(this.ctx.device, this.ctx.commandPool, allocated);
                }
                throw failure;
            }
        }
        return this.immediateCmd;
    }

    public void flushImmediate() {
        if (this.immediateCmd == null) return;
        var cmd = this.immediateCmd;
        this.immediateCmd = null;
        long fence = VK_NULL_HANDLE;
        boolean submitted = false;
        boolean completed = false;
        try (MemoryStack stack = stackPush()) {
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(immediate)");
            var fci = VkFenceCreateInfo.calloc(stack).sType$Default();
            var pFence = stack.callocLong(1);
            int createFence = vkCreateFence(this.ctx.device, fci, null, pFence);
            fence = pFence.get(0);
            check(createFence, "vkCreateFence(immediate)");
            var submit = VkSubmitInfo.calloc(stack).sType$Default()
                    .pCommandBuffers(stack.pointers(cmd));
            check(vkQueueSubmit(this.ctx.queue, submit, fence), "vkQueueSubmit(immediate)");
            submitted = true;
            check(vkWaitForFences(this.ctx.device, fence, true, Long.MAX_VALUE), "vkWaitForFences(immediate)");
            completed = true;
        } finally {
            if (submitted && !completed) {
                vkQueueWaitIdle(this.ctx.queue);
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(this.ctx.device, fence, null);
            }
            vkFreeCommandBuffers(this.ctx.device, this.ctx.commandPool, cmd);
        }
    }

    public void pollRetired() {
        boolean any = false;
        while (!this.inFlight.isEmpty()) {
            var frame = this.inFlight.peek();
            int status = vkGetEventStatus(this.ctx.device, frame.event);
            if (status == VK_EVENT_RESET) break;
            if (status != VK_EVENT_SET) {
                check(status, "vkGetEventStatus(frame " + frame.frameIdx + ")");
            }
            this.inFlight.pop();
            check(vkResetEvent(this.ctx.device, frame.event), "vkResetEvent");
            this.eventPool.add(frame.event);
            this.retiredCounter = frame.frameIdx;
            any = true;
        }
        if (any) this.runRetirement();
    }

    public void waitIdleRetireAll() {
        this.flushImmediate();
        int idle = vkDeviceWaitIdle(this.ctx.device);
        if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) {
            check(idle, "vkDeviceWaitIdle");
        }
        if (idle == VK_ERROR_DEVICE_LOST) {
            Logger.warn("Voxy VK: device lost while waiting for retirement; destroying Voxy-owned objects during teardown");
        }
        while (!this.inFlight.isEmpty()) {
            vkDestroyEvent(this.ctx.device, this.inFlight.pop().event, null);
        }
        this.retiredCounter = this.frameCounter;
        this.runRetirement();
    }

    private static Throwable collectFailure(Throwable first, Throwable next) {
        if (first == null) return next;
        first.addSuppressed(next);
        return first;
    }

    private static void rethrowFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error errorFailure) throw errorFailure;
    }

    private void runRetirement() {
        Throwable failure = null;

        //A readback callback is application logic and may fail. Never let it
        //prevent the other staging stream or native deferred-destroy queues from
        //retiring resources whose GPU work has already completed.
        for (var listener : this.retireListeners) {
            try {
                listener.onFramesRetired(this.retiredCounter);
            } catch (RuntimeException | Error listenerFailure) {
                failure = collectFailure(failure, listenerFailure);
            }
        }

        var destroyIterator = this.pendingDestroys.iterator();
        while (destroyIterator.hasNext()) {
            var d = destroyIterator.next();
            if (d.frameIdx > this.retiredCounter) continue;
            try {
                if (d.buffer != VK_NULL_HANDLE) vkDestroyBuffer(this.ctx.device, d.buffer, null);
                if (d.imageView != VK_NULL_HANDLE) vkDestroyImageView(this.ctx.device, d.imageView, null);
                if (d.image != VK_NULL_HANDLE) vkDestroyImage(this.ctx.device, d.image, null);
                if (d.memory != VK_NULL_HANDLE) vkFreeMemory(this.ctx.device, d.memory, null);
            } catch (RuntimeException | Error destroyFailure) {
                failure = collectFailure(failure, destroyFailure);
            } finally {
                destroyIterator.remove();
            }
        }

        var pipelineIterator = this.pendingPipelineDestroys.iterator();
        while (pipelineIterator.hasNext()) {
            var d = pipelineIterator.next();
            if (d.frameIdx > this.retiredCounter) continue;
            try {
                if (d.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(this.ctx.device, d.pipeline, null);
                if (d.pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(this.ctx.device, d.pipelineLayout, null);
                if (d.modules != null) {
                    for (long module : d.modules) {
                        if (module != VK_NULL_HANDLE) vkDestroyShaderModule(this.ctx.device, module, null);
                    }
                }
            } catch (RuntimeException | Error destroyFailure) {
                failure = collectFailure(failure, destroyFailure);
            } finally {
                pipelineIterator.remove();
            }
        }

        rethrowFailure(failure);
    }

    private long obtainEvent() {
        if (!this.eventPool.isEmpty()) {
            return this.eventPool.remove(this.eventPool.size() - 1);
        }
        try (MemoryStack stack = stackPush()) {
            var eci = VkEventCreateInfo.calloc(stack).sType$Default();
            var pEvent = stack.mallocLong(1);
            check(vkCreateEvent(this.ctx.device, eci, null, pEvent), "vkCreateEvent");
            return pEvent.get(0);
        }
    }

    private void markDeferredWorkForCurrentFrame() {
        if (this.frameCmd != null) {
            this.anyWorkThisFrame = true;
        }
    }

    public void deferDestroy(long buffer, long memory) {
        this.pendingDestroys.add(new PendingDestroy(this.frameCounter, buffer, VK_NULL_HANDLE, VK_NULL_HANDLE, memory));
        this.markDeferredWorkForCurrentFrame();
    }

    public void deferDestroyImage(long image, long view, long memory) {
        this.pendingDestroys.add(new PendingDestroy(this.frameCounter, VK_NULL_HANDLE, image, view, memory));
        this.markDeferredWorkForCurrentFrame();
    }

    public void deferDestroyPipeline(long pipeline, long pipelineLayout, long[] modules) {
        this.pendingPipelineDestroys.add(new PendingPipelineDestroy(
                this.frameCounter, pipeline, pipelineLayout, modules == null ? null : modules.clone()));
        this.markDeferredWorkForCurrentFrame();
    }

    public void fillBuffer(VkBuffer buffer, long offset, long size, int value) {
        vkCmdFillBuffer(this.cmd(), buffer.buffer, offset, size, value);
        this.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }

    public void barrier(int srcStage, int srcAccess, int dstStage, int dstAccess) {
        try (MemoryStack stack = stackPush()) {
            var mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(srcAccess).dstAccessMask(dstAccess);
            vkCmdPipelineBarrier(this.cmd(), srcStage, dstStage, 0, mb, null, null);
        }
    }

    public void computeToComputeBarrier() {
        this.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    public void computeToDrawBarrier() {
        this.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    }

    public void computeToTransferBarrier() {
        this.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }

    public void free() {
        this.waitIdleRetireAll();
        for (long event : this.eventPool) {
            vkDestroyEvent(this.ctx.device, event, null);
        }
        this.eventPool.clear();
        int pending = this.pendingDestroys.size() + this.pendingPipelineDestroys.size();
        if (pending != 0) {
            Logger.warn("VkFrameCtx freed with " + pending + " pending destroys remaining");
        }
    }
}
