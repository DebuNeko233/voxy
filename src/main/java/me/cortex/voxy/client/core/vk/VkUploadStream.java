package me.cortex.voxy.client.core.vk;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import me.cortex.voxy.client.core.rendering.util.AbstractUploadStream;
import me.cortex.voxy.client.core.rendering.util.IDeviceBuffer;
import me.cortex.voxy.common.util.AllocationArena;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkBufferCopy;

import java.util.ArrayDeque;
import java.util.Deque;

import static me.cortex.voxy.common.util.AllocationArena.SIZE_LIMIT;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

public class VkUploadStream extends AbstractUploadStream {
    private final VkFrameCtx ctx;
    private final VkBuffer stagingBuffer;
    private final long stagingPtr;
    private final int alignment;

    private final AllocationArena allocationArena = new AllocationArena();
    private final Deque<UploadFrame> frames = new ArrayDeque<>();
    private final LongArrayList thisFrameAllocations = new LongArrayList();
    private final Deque<UploadData> uploadList = new ArrayDeque<>();

    private long caddr = -1;
    private long offset = 0;

    public VkUploadStream(VkFrameCtx ctx, long size) {
        this.ctx = ctx;
        VkBuffer staging = null;
        long mapped = 0;
        int resolvedAlignment = 0;
        try {
            staging = new VkBuffer(ctx, size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
            mapped = staging.map();
            long minAlign = Math.max(16, ctx.vk().storageBufferOffsetAlignment());
            minAlign = Math.max(minAlign, ctx.vk().uniformBufferOffsetAlignment());
            if (minAlign > Integer.MAX_VALUE) throw new IllegalStateException("Vulkan staging alignment too large: " + minAlign);
            resolvedAlignment = (int) minAlign;
            this.allocationArena.setLimit(size);
        } catch (RuntimeException | Error failure) {
            if (staging != null) staging.free();
            ctx.waitIdleRetireAll();
            throw failure;
        }
        this.stagingBuffer = staging;
        this.stagingPtr = mapped;
        this.alignment = resolvedAlignment;
        ctx.addRetireListener(this::retireUpTo);
    }

    public long stagingBufferHandle() {
        return this.stagingBuffer.buffer;
    }

    public long stagingArenaBytes() {
        return this.allocationArena.getSize();
    }

    public int pendingFrameCount() {
        return this.frames.size();
    }

    @Override
    public long upload(IDeviceBuffer buffer, long destOffset, long size) {
        if (!(buffer instanceof VkBuffer vkBuffer)) throw new IllegalArgumentException("Vulkan upload requires a VkBuffer destination");
        if (size <= 0 || size > Integer.MAX_VALUE) throw new IllegalArgumentException("Invalid Vulkan upload size: " + size);
        if (destOffset < 0 || destOffset > buffer.sizeBytes() - size) throw new IllegalArgumentException("Vulkan upload exceeds destination buffer");
        long addr = this.rawUploadAddress((int) size);
        this.uploadList.add(new UploadData(vkBuffer, addr, destOffset, size));
        return this.stagingPtr + addr;
    }

    @Override
    public long rawUploadAddress(int size) {
        if (size < 0) throw new IllegalStateException("Negative size");
        size = this.alignUpAlloc(size);
        if (size > this.stagingBuffer.size()) throw new IllegalArgumentException("Upload larger than Vulkan staging buffer");

        long addr;
        if (this.caddr == -1 || !this.allocationArena.expand(this.caddr, size)) {
            this.caddr = this.allocationArena.alloc(size);
            if (this.caddr == SIZE_LIMIT) {
                //The current Minecraft frame has not been submitted yet, so
                //vkDeviceWaitIdle cannot make allocations consumed by this frame
                //safe to reuse. Older versions attempted exactly that, which could
                //overwrite staging bytes still referenced by an unsubmitted copy.
                throw new IllegalStateException("Vulkan upload staging exhausted ("
                        + (this.stagingBuffer.size() >> 20) + " MiB, pendingFrames=" + this.frames.size()
                        + "). Cannot safely force-reuse staging memory before Minecraft submits the current frame.");
            }
            this.thisFrameAllocations.add(this.caddr);
            this.offset = size;
            addr = this.caddr;
        } else {
            addr = this.caddr + this.offset;
            this.offset += size;
        }
        if (this.caddr + this.offset > this.stagingBuffer.size()) throw new IllegalStateException("Vulkan upload staging allocation exceeded buffer");
        return addr;
    }

    @Override
    public void commit() {
        if (this.uploadList.isEmpty()) {
            this.caddr = -1;
            this.offset = 0;
            return;
        }
        var cmd = this.ctx.cmd();
        this.ctx.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        try (MemoryStack stack = stackPush()) {
            for (var entry : this.uploadList) {
                var region = VkBufferCopy.calloc(1, stack)
                        .srcOffset(entry.uploadOffset).dstOffset(entry.targetOffset).size(entry.size);
                vkCmdCopyBuffer(cmd, this.stagingBuffer.buffer, entry.target.buffer, region);
            }
        }
        this.uploadList.clear();
        this.ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
        this.caddr = -1;
        this.offset = 0;
    }

    @Override
    public void tick() {
        this.commit();
        if (!this.thisFrameAllocations.isEmpty()) {
            this.frames.add(new UploadFrame(this.ctx.currentFrame(), new LongArrayList(this.thisFrameAllocations)));
            this.thisFrameAllocations.clear();
        }
    }

    private void retireUpTo(long retiredFrame) {
        while (!this.frames.isEmpty() && this.frames.peek().frameIdx <= retiredFrame) {
            var frame = this.frames.pop();
            Throwable failure = null;
            for (int i = 0; i < frame.allocations.size(); i++) {
                try {
                    this.allocationArena.free(frame.allocations.getLong(i));
                } catch (RuntimeException | Error releaseFailure) {
                    if (failure == null) failure = releaseFailure;
                    else failure.addSuppressed(releaseFailure);
                }
            }
            if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure instanceof Error errorFailure) throw errorFailure;
        }
    }

    @Override
    public long getBaseAddress() {
        return this.stagingPtr;
    }

    @Override
    public int baseAlignment() {
        return this.alignment;
    }

    @Override
    public void free() {
        //CPU allocator bookkeeping can be discarded immediately because the
        //stream will never allocate again. The VkBuffer itself is retired by
        //Minecraft's submission-completion destruction queue.
        this.frames.clear();
        this.thisFrameAllocations.clear();
        this.uploadList.clear();
        this.stagingBuffer.free();
    }

    private record UploadFrame(long frameIdx, LongArrayList allocations) {}
    private record UploadData(VkBuffer target, long uploadOffset, long targetOffset, long size) {}
}
