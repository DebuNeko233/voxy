package me.cortex.voxy.client.core.vk;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import me.cortex.voxy.client.core.rendering.util.AbstractDownloadStream;
import me.cortex.voxy.client.core.rendering.util.IDeviceBuffer;
import me.cortex.voxy.common.util.AllocationArena;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkBufferCopy;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;

import static me.cortex.voxy.common.util.AllocationArena.SIZE_LIMIT;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

public class VkDownloadStream extends AbstractDownloadStream {
    private final VkFrameCtx ctx;
    private final VkBuffer readbackBuffer;
    private final long readbackPtr;

    private final AllocationArena allocationArena = new AllocationArena();
    private final Deque<DownloadFrame> frames = new ArrayDeque<>();
    private final LongArrayList thisFrameAllocations = new LongArrayList();
    private final Deque<DownloadData> downloadList = new ArrayDeque<>();
    private final ArrayList<DownloadData> thisFrameDownloadList = new ArrayList<>();

    private long caddr = -1;
    private long offset = 0;
    private long recordFrame = -1;
    private boolean closed;

    public VkDownloadStream(VkFrameCtx ctx, long size) {
        this.ctx = ctx;
        VkBuffer readback = null;
        long mapped = 0;
        try {
            readback = new VkBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
            mapped = readback.map();
            this.allocationArena.setLimit(size);
        } catch (RuntimeException | Error failure) {
            if (readback != null) readback.free();
            ctx.waitIdleRetireAll();
            throw failure;
        }
        this.readbackBuffer = readback;
        this.readbackPtr = mapped;
        ctx.addRetireListener(this::retireUpTo);
    }

    public long stagingArenaBytes() {
        return this.allocationArena.getSize();
    }

    public int pendingFrameCount() {
        return this.frames.size();
    }

    @Override
    public void download(IDeviceBuffer buffer, long downloadOffset, long size, DownloadResultConsumer resultConsumer) {
        if (this.closed) throw new IllegalStateException("Vulkan download stream is closed");
        if (!(buffer instanceof VkBuffer vkBuffer)) throw new IllegalArgumentException("Vulkan download requires a VkBuffer source");
        if (resultConsumer == null) throw new IllegalArgumentException("Vulkan download requires a result consumer");
        if (size > Integer.MAX_VALUE || size <= 0) throw new IllegalArgumentException("Invalid Vulkan download size: " + size);
        if (downloadOffset < 0 || downloadOffset > buffer.sizeBytes() - size) throw new IllegalArgumentException("Vulkan download exceeds source buffer");

        long addr;
        if (this.caddr == -1 || !this.allocationArena.expand(this.caddr, (int) size)) {
            this.caddr = this.allocationArena.alloc((int) size);
            if (this.caddr == SIZE_LIMIT) {
                //The current Minecraft frame may still only be recorded, not
                //submitted. Device-idle cannot make that readback range safe to
                //reuse, so fail instead of fabricating retirement.
                throw new IllegalStateException("Vulkan readback staging exhausted before Minecraft submission retirement");
            }
            this.thisFrameAllocations.add(this.caddr);
            this.offset = size;
            addr = this.caddr;
        } else {
            addr = this.caddr + this.offset;
            this.offset += size;
        }
        if (this.caddr + this.offset > this.readbackBuffer.size()) throw new IllegalStateException("Vulkan readback staging allocation exceeded buffer");
        this.downloadList.add(new DownloadData(vkBuffer, addr, downloadOffset, size, resultConsumer));
        this.commit();
    }

    @Override
    public void commit() {
        if (this.closed || this.downloadList.isEmpty()) return;
        this.recordFrame = this.ctx.currentFrame();
        var cmd = this.ctx.cmd();
        this.ctx.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        try (MemoryStack stack = stackPush()) {
            for (var entry : this.downloadList) {
                var region = VkBufferCopy.calloc(1, stack)
                        .srcOffset(entry.targetOffset).dstOffset(entry.downloadStreamOffset).size(entry.size);
                vkCmdCopyBuffer(cmd, entry.target.buffer, this.readbackBuffer.buffer, region);
            }
        }
        this.ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        this.thisFrameDownloadList.addAll(this.downloadList);
        this.downloadList.clear();
        this.caddr = -1;
        this.offset = 0;
    }

    @Override
    public void tick() {
        if (this.closed) return;
        this.commit();
        if (!this.thisFrameAllocations.isEmpty()) {
            this.frames.add(new DownloadFrame(this.recordFrame,
                    new LongArrayList(this.thisFrameAllocations), new ArrayList<>(this.thisFrameDownloadList)));
            this.thisFrameAllocations.clear();
            this.thisFrameDownloadList.clear();
        }
    }

    private void retireUpTo(long retiredFrame) {
        if (this.closed) return;
        while (!this.frames.isEmpty() && this.frames.peek().frameIdx <= retiredFrame) {
            var frame = this.frames.pop();
            Throwable failure = null;
            for (var data : frame.data) {
                try {
                    data.resultConsumer.consume(this.readbackPtr + data.downloadStreamOffset, data.size);
                } catch (RuntimeException | Error callbackFailure) {
                    if (failure == null) failure = callbackFailure;
                    else failure.addSuppressed(callbackFailure);
                }
            }
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
    public void waitDiscard() {
        //AbstractDownloadStream reserves this for shutdown paths. On Vulkan we
        //must not force Minecraft's still-unsubmitted command buffer to retire;
        //closing drops only CPU callback bookkeeping and defers the VkBuffer via
        //Blaze3D's submission-safe destruction queue.
        this.free();
    }

    @Override
    public void flushWaitClear() {
        //For the same reason, shutdown cannot legally force pending host-owned
        //submissions to complete callbacks here. The renderer is going away, so
        //discard the callbacks and keep the native readback buffer alive until
        //Minecraft retires the submission.
        this.free();
    }

    @Override
    public void free() {
        if (this.closed) return;
        this.closed = true;
        //Do not invoke pending readback callbacks during shutdown. Their GPU
        //copies may belong to submissions that Minecraft has not retired yet.
        //Dropping CPU bookkeeping is safe because the VkBuffer itself remains
        //alive until Mojang's destruction queue says the submission completed.
        this.downloadList.clear();
        this.thisFrameDownloadList.clear();
        this.thisFrameAllocations.clear();
        this.frames.clear();
        this.caddr = -1;
        this.offset = 0;
        this.recordFrame = -1;
        this.allocationArena.reset();
        this.readbackBuffer.free();
    }

    private record DownloadFrame(long frameIdx, LongArrayList allocations, ArrayList<DownloadData> data) {}
    private record DownloadData(VkBuffer target, long downloadStreamOffset, long targetOffset, long size, DownloadResultConsumer resultConsumer) {}
}
