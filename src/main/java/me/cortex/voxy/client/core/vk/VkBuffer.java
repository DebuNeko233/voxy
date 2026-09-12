package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.client.core.rendering.IRenderList;
import me.cortex.voxy.client.core.rendering.util.IDeviceBuffer;
import me.cortex.voxy.common.util.TrackedObject;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.util.vma.VmaAllocationInfo;
import org.lwjgl.vulkan.VkBufferCreateInfo;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Device buffer on the pure-Vulkan path; the VK analogue of GlBuffer. All Voxy
//buffers get a superset of usage flags (storage/indirect/index/transfer) so a
//single class covers every role the GL path used raw buffer ids for.
//
//Memory comes from Minecraft's existing VMA allocator. This avoids creating a
//dedicated VkDeviceMemory object for every Voxy buffer and lets VMA suballocate
//and reuse backing blocks alongside Blaze3D's own Vulkan resources.
//
//Freeing is DEFERRED through VkFrameCtx — a buffer may still be referenced by
//command buffers in flight when free() is called.
public class VkBuffer extends TrackedObject implements IDeviceBuffer, IRenderList {
    public static final int USAGE_DEFAULT = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
            | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    private final VkFrameCtx ctx;
    public final long buffer;
    public final long allocation;
    private final long size;
    private final long allocationSize;
    private long mappedAddress;

    private static int COUNT;
    private static long TOTAL_SIZE;
    private static long TOTAL_ALLOCATION_SIZE;

    public VkBuffer(VkFrameCtx ctx, long size) {
        this(ctx, size, USAGE_DEFAULT, false);
    }

    public VkBuffer(VkFrameCtx ctx, long size, int usage, boolean hostVisible) {
        this.ctx = ctx;
        this.size = size;
        var vctx = ctx.vk();

        long createdBuffer = VK_NULL_HANDLE;
        long createdAllocation = VK_NULL_HANDLE;
        long allocatedBytes = 0;
        try (MemoryStack stack = stackPush()) {
            var bci = VkBufferCreateInfo.calloc(stack).sType$Default()
                    .size(size).usage(usage).sharingMode(VK_SHARING_MODE_EXCLUSIVE);

            var aci = VmaAllocationCreateInfo.calloc(stack);
            if (hostVisible) {
                //Keep the old staging/readback contract: callers write/read the
                //mapped pointer directly and never flush/invalidate, so coherent
                //host-visible memory remains a hard requirement.
                aci.usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
                        .flags(Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
                        .requiredFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            } else {
                //Do not let large Voxy allocations silently push the shared
                //Minecraft allocator beyond the driver's reported heap budget.
                //Geometry already has a retry/downsize path, and other resources
                //fail cleanly instead of forcing the driver into oversubscription.
                aci.usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
                        .flags(Vma.VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT)
                        .preferredFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            }

            var pBuf = stack.mallocLong(1);
            var pAllocation = stack.mallocPointer(1);
            var allocationInfo = VmaAllocationInfo.calloc(stack);
            check(Vma.vmaCreateBuffer(vctx.vmaAllocator, bci, aci, pBuf, pAllocation, allocationInfo),
                    "vmaCreateBuffer");
            createdBuffer = pBuf.get(0);
            createdAllocation = pAllocation.get(0);
            allocatedBytes = allocationInfo.size();
        } catch (RuntimeException | Error failure) {
            if (createdBuffer != VK_NULL_HANDLE && createdAllocation != VK_NULL_HANDLE) {
                Vma.vmaDestroyBuffer(vctx.vmaAllocator, createdBuffer, createdAllocation);
            }
            throw failure;
        }

        this.buffer = createdBuffer;
        this.allocation = createdAllocation;
        this.allocationSize = allocatedBytes;
        COUNT++;
        TOTAL_SIZE += size;
        TOTAL_ALLOCATION_SIZE += allocatedBytes;
    }

    /** Maps the whole allocation; only valid for host-visible buffers. */
    public long map() {
        if (this.mappedAddress != 0) return this.mappedAddress;
        try (MemoryStack stack = stackPush()) {
            var pp = stack.mallocPointer(1);
            check(Vma.vmaMapMemory(this.ctx.vk().vmaAllocator, this.allocation, pp), "vmaMapMemory");
            this.mappedAddress = pp.get(0);
            return this.mappedAddress;
        }
    }

    @Override
    public long sizeBytes() {
        return this.size;
    }

    public long size() {
        return this.size;
    }

    /** Actual VMA allocation size backing this VkBuffer. */
    public long allocationSize() {
        return this.allocationSize;
    }

    @Override
    public int glId() {
        throw new UnsupportedOperationException("VkBuffer has no GL id");
    }

    /** Records a fill of 0 across the given range into the current frame commands. */
    public VkBuffer zeroRange(long offset, long len) {
        this.ctx.fillBuffer(this, offset, len, 0);
        return this;
    }

    public VkBuffer zero() {
        return this.zeroRange(0, VK_WHOLE_SIZE);
    }

    public VkBuffer fill(int value) {
        this.ctx.fillBuffer(this, 0, VK_WHOLE_SIZE, value);
        return this;
    }

    private void releaseJavaOwnership() {
        this.free0();
        if (this.mappedAddress != 0) {
            Vma.vmaUnmapMemory(this.ctx.vk().vmaAllocator, this.allocation);
            this.mappedAddress = 0;
        }
        COUNT--;
        TOTAL_SIZE -= this.size;
        TOTAL_ALLOCATION_SIZE -= this.allocationSize;
    }

    /**
     * Immediate destruction is only legal after the caller has synchronously
     * proven that every submitted command using this buffer completed. Normal
     * frame resources must use free() and Minecraft's retirement queue.
     */
    public void freeCompletedNow() {
        this.releaseJavaOwnership();
        Vma.vmaDestroyBuffer(this.ctx.vk().vmaAllocator, this.buffer, this.allocation);
    }

    @Override
    public void free() {
        this.releaseJavaOwnership();
        //Unmapping CPU access does not return the allocation to VMA. Native
        //buffer+allocation destruction is still deferred until Minecraft's
        //submission retirement says the GPU is done with it.
        this.ctx.deferDestroyVmaBuffer(this.buffer, this.allocation);
    }

    public static int getCount() {
        return COUNT;
    }

    /** Logical sizes requested by Voxy. */
    public static long getTotalSize() {
        return TOTAL_SIZE;
    }

    /** Actual VMA allocation bytes backing live VkBuffers. */
    public static long getTotalAllocationSize() {
        return TOTAL_ALLOCATION_SIZE;
    }
}
