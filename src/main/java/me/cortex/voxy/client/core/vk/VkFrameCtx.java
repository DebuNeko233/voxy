package me.cortex.voxy.client.core.vk;

import com.mojang.blaze3d.systems.RenderSystem;
import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkMemoryBarrier;

import java.util.ArrayList;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

/**
 * Per-frame Vulkan recording/lifetime context.
 *
 * Voxy records into Minecraft's live command buffer and submission ownership
 * remains entirely with Blaze3D. Resource retirement is tied to Minecraft's
 * timeline-semaphore-backed DestructionQueue; Voxy never infers completion from
 * an in-command-buffer event and never submits independently to the graphics
 * queue.
 */
public final class VkFrameCtx {
    public interface FrameRetireListener {
        void onFramesRetired(long retiredUpToInclusive);
    }

    private final VulkanContext ctx;
    private VkCommandBuffer frameCmd;
    private boolean synchronousHostWork;
    private boolean anyWorkThisFrame;
    private boolean closed;

    private long frameCounter;
    private volatile long retiredCounter = -1;
    private volatile int pendingFrameCallbacks;
    private volatile int pendingNativeDestroys;

    private final ArrayList<FrameRetireListener> retireListeners = new ArrayList<>();
    private Throwable deferredFailure;

    public VkFrameCtx(VulkanContext ctx) {
        this.ctx = ctx;
    }

    public VulkanContext vk() {
        return this.ctx;
    }

    public void addRetireListener(FrameRetireListener listener) {
        if (this.closed) throw new IllegalStateException("VkFrameCtx is closed");
        this.retireListeners.add(listener);
    }

    public long currentFrame() {
        return this.frameCounter;
    }

    public long retiredFrame() {
        return this.retiredCounter;
    }

    public int inFlightFrameCount() {
        return this.pendingFrameCallbacks;
    }

    /** Native destruction callbacks already handed to Minecraft's safe queue. */
    public int pendingDestroyCount() {
        return this.pendingNativeDestroys;
    }

    /** Kept for the existing debug line; host-backed retirement uses no VkEvent pool. */
    public int pooledEventCount() {
        return 0;
    }

    public void beginFrame(VkCommandBuffer mcFrameCommandBuffer) {
        RenderSystem.assertOnRenderThread();
        this.throwDeferredFailure();
        if (this.closed) throw new IllegalStateException("VkFrameCtx is closed");
        if (this.frameCmd != null) throw new IllegalStateException("Frame already begun");
        if (this.synchronousHostWork) throw new IllegalStateException("Unflushed synchronous Vulkan work before frame begin");
        if (mcFrameCommandBuffer == null) throw new IllegalArgumentException("Minecraft frame command buffer is null");
        this.frameCmd = mcFrameCommandBuffer;
        this.anyWorkThisFrame = false;
    }

    /**
     * Finish Voxy's recording interval. If this frame recorded GPU work, queue a
     * tiny Java completion callback in Minecraft's own DestructionQueue. That
     * callback runs only after the corresponding whole graphics submission has
     * completed according to Blaze3D's timeline semaphore.
     */
    public void endFrame() {
        RenderSystem.assertOnRenderThread();
        if (this.frameCmd == null) throw new IllegalStateException("No frame begun");
        this.frameCmd = null;
        if (!this.anyWorkThisFrame) return;

        final long frameIdx = this.frameCounter++;
        this.anyWorkThisFrame = false;
        this.pendingFrameCallbacks++;
        try {
            this.ctx.deferUntilSubmissionComplete(() -> this.onHostFrameCompleted(frameIdx));
        } catch (RuntimeException | Error failure) {
            this.pendingFrameCallbacks--;
            throw failure;
        }
    }

    private void onHostFrameCompleted(long frameIdx) {
        try {
            if (this.closed) return;
            this.retiredCounter = Math.max(this.retiredCounter, frameIdx);
            Throwable failure = null;
            //A readback listener is application logic and may fail. Never throw
            //from Mojang's DestructionQueue callback: doing so could abort
            //Minecraft's own destruction-queue rotation. Aggregate and surface
            //the error on the next Voxy call instead.
            for (var listener : this.retireListeners) {
                try {
                    listener.onFramesRetired(this.retiredCounter);
                } catch (RuntimeException | Error listenerFailure) {
                    failure = collectFailure(failure, listenerFailure);
                }
            }
            if (failure != null) this.recordDeferredFailure(failure);
        } finally {
            if (this.pendingFrameCallbacks > 0) this.pendingFrameCallbacks--;
        }
    }

    /**
     * Current recording target. During normal rendering this is the Minecraft
     * frame command buffer supplied by beginFrame(). Outside a frame, Voxy asks
     * Minecraft's encoder for its current primary command buffer and marks that
     * batch for synchronous submission by flushImmediate().
     */
    public VkCommandBuffer cmd() {
        RenderSystem.assertOnRenderThread();
        if (this.closed) throw new IllegalStateException("VkFrameCtx is closed");
        if (this.frameCmd != null) {
            this.anyWorkThisFrame = true;
            return this.frameCmd;
        }
        this.synchronousHostWork = true;
        return this.ctx.hostCommandBuffer();
    }

    /**
     * Historical name retained for callers. There is no Voxy-owned immediate
     * command buffer anymore: this submits Minecraft's current encoder batch and
     * waits for that exact timeline submit to complete.
     */
    public void flushImmediate() {
        RenderSystem.assertOnRenderThread();
        this.throwDeferredFailure();
        if (!this.synchronousHostWork) return;
        if (this.frameCmd != null) throw new IllegalStateException("Cannot synchronously submit while a Voxy frame is active");

        //Clear first. If submission itself fails, the encoder/device state is no
        //longer safe to blindly re-submit on a cleanup path.
        this.synchronousHostWork = false;
        this.ctx.submitAndWaitCurrent();
        this.throwDeferredFailure();
    }

    /** Host callbacks execute during Minecraft submit; polling is now only an error checkpoint. */
    public void pollRetired() {
        RenderSystem.assertOnRenderThread();
        this.throwDeferredFailure();
    }

    /**
     * Synchronous device-idle helper for construction/teardown only. Current
     * recorded work is first submitted through Minecraft's encoder so CPU/GPU
     * ordering remains identical to Blaze3D's own command stream.
     */
    public void waitIdleRetireAll() {
        RenderSystem.assertOnRenderThread();
        if (this.frameCmd != null) {
            throw new IllegalStateException("Cannot wait/retire while Minecraft frame is still being recorded");
        }
        this.flushImmediate();
        int idle = vkDeviceWaitIdle(this.ctx.device);
        if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) {
            check(idle, "vkDeviceWaitIdle");
        }
        if (idle == VK_ERROR_DEVICE_LOST) {
            Logger.warn("Voxy VK: device lost while waiting for queue idle during teardown");
        }
        this.throwDeferredFailure();
    }

    /**
     * Flush every Minecraft destruction slot after Voxy has queued its frees.
     * Device-idle alone is insufficient because Mojang rotates its two-slot
     * DestructionQueue only from VulkanCommandEncoder.submit().
     */
    public void drainDeferredDestruction() {
        RenderSystem.assertOnRenderThread();
        if (this.frameCmd != null) {
            throw new IllegalStateException("Cannot drain Vulkan destruction queues while a frame is active");
        }
        if (this.closed) throw new IllegalStateException("VkFrameCtx is closed");
        this.flushImmediate();
        this.ctx.drainDeferredDestruction();
        this.throwDeferredFailure();
        if (this.pendingNativeDestroys != 0) {
            Logger.warn("Voxy VK: " + this.pendingNativeDestroys
                    + " native destroys remain after host destruction-queue drain");
        }
    }

    private static Throwable collectFailure(Throwable first, Throwable next) {
        if (first == null) return next;
        first.addSuppressed(next);
        return first;
    }

    private synchronized void recordDeferredFailure(Throwable failure) {
        this.deferredFailure = collectFailure(this.deferredFailure, failure);
    }

    private synchronized void throwDeferredFailure() {
        Throwable failure = this.deferredFailure;
        this.deferredFailure = null;
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error errorFailure) throw errorFailure;
    }

    private void queueNativeDestroy(Runnable destroy) {
        RenderSystem.assertOnRenderThread();
        this.pendingNativeDestroys++;
        try {
            this.ctx.deferUntilSubmissionComplete(() -> {
                try {
                    destroy.run();
                } catch (RuntimeException | Error failure) {
                    //Never let Voxy abort Mojang's DestructionQueue rotation.
                    this.recordDeferredFailure(failure);
                } finally {
                    if (this.pendingNativeDestroys > 0) this.pendingNativeDestroys--;
                }
            });
        } catch (RuntimeException | Error failure) {
            this.pendingNativeDestroys--;
            throw failure;
        }
    }

    /** Destroy a buffer suballocated from Minecraft's VMA only after its submission retires. */
    public void deferDestroyVmaBuffer(long buffer, long allocation) {
        if (buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE) return;
        if (buffer == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) {
            throw new IllegalArgumentException("Incomplete VMA buffer handle pair");
        }
        this.queueNativeDestroy(() -> Vma.vmaDestroyBuffer(this.ctx.vmaAllocator, buffer, allocation));
    }

    /**
     * Retire all views and the VMA image allocation as one ordered callback.
     * Vulkan requires every image view to be destroyed before its image; keeping
     * this in one callback avoids relying on ordering between separate queued
     * destruction actions.
     */
    public void deferDestroyVmaImage(long image, long mainView, long allocation, long[] additionalViews) {
        if (image == VK_NULL_HANDLE && mainView == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE
                && (additionalViews == null || additionalViews.length == 0)) return;
        if (image == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) {
            throw new IllegalArgumentException("Incomplete VMA image handle pair");
        }
        long[] ownedViews = additionalViews == null ? new long[0] : additionalViews.clone();
        this.queueNativeDestroy(() -> {
            for (long view : ownedViews) {
                if (view != VK_NULL_HANDLE) vkDestroyImageView(this.ctx.device, view, null);
            }
            if (mainView != VK_NULL_HANDLE) vkDestroyImageView(this.ctx.device, mainView, null);
            Vma.vmaDestroyImage(this.ctx.vmaAllocator, image, allocation);
        });
    }

    public void deferDestroyPipeline(long pipeline, long pipelineLayout, long[] modules) {
        long[] ownedModules = modules == null ? null : modules.clone();
        if (pipeline == VK_NULL_HANDLE && pipelineLayout == VK_NULL_HANDLE
                && (ownedModules == null || ownedModules.length == 0)) return;
        this.queueNativeDestroy(() -> {
            if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(this.ctx.device, pipeline, null);
            if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(this.ctx.device, pipelineLayout, null);
            if (ownedModules != null) {
                for (long module : ownedModules) {
                    if (module != VK_NULL_HANDLE) vkDestroyShaderModule(this.ctx.device, module, null);
                }
            }
        });
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
        RenderSystem.assertOnRenderThread();
        if (this.frameCmd != null) {
            throw new IllegalStateException("Cannot free VkFrameCtx while a Minecraft frame is being recorded");
        }
        try {
            this.flushImmediate();
        } catch (RuntimeException | Error failure) {
            Logger.error("Error flushing Voxy synchronous Vulkan work during frame-context close", failure);
        }
        this.closed = true;
        this.retireListeners.clear();
        //Native destroys and old frame-completion callbacks already queued in
        //Minecraft remain valid: their closures own the raw handles they need.
        //Completion callbacks observe closed=true and become no-ops.
    }
}
