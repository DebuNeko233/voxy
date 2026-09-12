package me.cortex.voxy.client.core.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.util.vma.VmaAllocationInfo;
import org.lwjgl.vulkan.VkImageCreateInfo;
import org.lwjgl.vulkan.VkImageFormatListCreateInfo;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkImageViewCreateInfo;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//2D image + full view (+ optional per-mip views) for the pure-VK path:
//offscreen colour/depth targets, the HiZ mip pyramid, and the model atlas.
//Memory is suballocated from Minecraft's existing VMA allocator instead of
//creating one VkDeviceMemory object per image. Tracks the current layout for
//whole-image transitions (Voxy transitions whole subresource ranges only).
public final class VkImage2D {
    private final VkFrameCtx ctx;
    public final long image;
    public final long allocation;
    public final long view;
    public final long[] mipViews;//null unless requested
    public final int width, height, mipLevels;
    public final int format;
    public final int aspect;
    private final long allocationSize;
    private int currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    private boolean freed;

    private static int COUNT;
    private static long TOTAL_ALLOCATION_SIZE;

    public VkImage2D(VkFrameCtx ctx, int width, int height, int mipLevels, int format, int usage, int aspect, boolean perMipViews) {
        this(ctx, width, height, mipLevels, format, usage, aspect, perMipViews, null);
    }

    public VkImage2D(VkFrameCtx ctx, int width, int height, int mipLevels, int format, int usage, int aspect,
                     boolean perMipViews, int[] viewFormats) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.mipLevels = mipLevels;
        this.format = format;
        this.aspect = aspect;
        var vctx = ctx.vk();

        long createdImage = VK_NULL_HANDLE;
        long createdAllocation = VK_NULL_HANDLE;
        long allocatedBytes = 0;
        long createdView = VK_NULL_HANDLE;
        long[] createdMipViews = perMipViews ? new long[mipLevels] : null;
        try (MemoryStack stack = stackPush()) {
            var ici = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK_IMAGE_TYPE_2D)
                    .format(format)
                    .extent(e -> e.width(width).height(height).depth(1))
                    .mipLevels(mipLevels).arrayLayers(1)
                    .samples(VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK_IMAGE_TILING_OPTIMAL)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
            if (viewFormats != null && viewFormats.length > 0) {
                ici.flags(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
                var formatList = VkImageFormatListCreateInfo.calloc(stack).sType$Default()
                        .pViewFormats(stack.ints(viewFormats));
                ici.pNext(formatList.address());
            }

            //Mirror Minecraft 26.2's VulkanGpuTexture allocator strategy: VMA
            //chooses/suballocates device-preferred memory from the allocator
            //owned by the adopted VulkanDevice. WITHIN_BUDGET prevents Voxy's
            //large atlas/frame targets from silently oversubscribing the heap.
            var aci = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
                    .flags(Vma.VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT)
                    .preferredFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            var pImg = stack.mallocLong(1);
            var pAllocation = stack.mallocPointer(1);
            var allocationInfo = VmaAllocationInfo.calloc(stack);
            check(Vma.vmaCreateImage(vctx.vmaAllocator, ici, aci, pImg, pAllocation, allocationInfo),
                    "vmaCreateImage");
            createdImage = pImg.get(0);
            createdAllocation = pAllocation.get(0);
            allocatedBytes = allocationInfo.size();

            createdView = createView(stack, vctx, createdImage, format, aspect, 0, mipLevels);
            if (createdMipViews != null) {
                for (int i = 0; i < mipLevels; i++) {
                    createdMipViews[i] = createView(stack, vctx, createdImage, format, aspect, i, 1);
                }
            }
        } catch (RuntimeException | Error failure) {
            if (createdMipViews != null) {
                for (long mipView : createdMipViews) {
                    if (mipView != VK_NULL_HANDLE) vkDestroyImageView(vctx.device, mipView, null);
                }
            }
            if (createdView != VK_NULL_HANDLE) vkDestroyImageView(vctx.device, createdView, null);
            if (createdImage != VK_NULL_HANDLE && createdAllocation != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vctx.vmaAllocator, createdImage, createdAllocation);
            }
            throw failure;
        }

        this.image = createdImage;
        this.allocation = createdAllocation;
        this.view = createdView;
        this.mipViews = createdMipViews;
        this.allocationSize = allocatedBytes;
        COUNT++;
        TOTAL_ALLOCATION_SIZE += allocatedBytes;
    }

    private static long createView(MemoryStack stack, VulkanContext vctx, long image, int format, int aspect,
                                   int baseMip, int mipCount) {
        var vci = VkImageViewCreateInfo.calloc(stack).sType$Default()
                .image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(format);
        vci.subresourceRange().aspectMask(aspect).baseMipLevel(baseMip).levelCount(mipCount).baseArrayLayer(0).layerCount(1);
        var pView = stack.mallocLong(1);
        check(vkCreateImageView(vctx.device, vci, null, pView), "vkCreateImageView");
        return pView.get(0);
    }

    /** Whole-image layout transition. UNDEFINED-layout images get TOP_OF_PIPE/0
     *  (no prior producer to synchronize — contents are discarded). */
    public void transition(int newLayout, int srcStage, int srcAccess, int dstStage, int dstAccess) {
        if (this.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            srcAccess = 0;
        }
        try (MemoryStack stack = stackPush()) {
            var imb = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(srcAccess).dstAccessMask(dstAccess)
                    .oldLayout(this.currentLayout).newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(this.image);
            imb.subresourceRange().aspectMask(this.aspect).levelCount(this.mipLevels).layerCount(1);
            vkCmdPipelineBarrier(this.ctx.cmd(), srcStage, dstStage, 0, null, null, imb);
            this.currentLayout = newLayout;
        }
    }

    public record BatchEntry(VkImage2D image, int newLayout, int srcAccess, int dstAccess) {}
    public static void transitionBatch(java.util.List<BatchEntry> entries, int unionSrcStage, int unionDstStage) {
        if (entries.isEmpty()) return;
        boolean anyUndefined = false;
        try (MemoryStack stack = stackPush()) {
            var imbs = VkImageMemoryBarrier.calloc(entries.size(), stack);
            for (int i = 0; i < entries.size(); i++) {
                var e = entries.get(i);
                boolean undef = e.image.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED;
                if (undef) anyUndefined = true;
                imbs.get(i).sType$Default()
                        .srcAccessMask(undef ? 0 : e.srcAccess).dstAccessMask(e.dstAccess)
                        .oldLayout(e.image.currentLayout).newLayout(e.newLayout)
                        .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .image(e.image.image)
                        .subresourceRange().aspectMask(e.image.aspect).levelCount(e.image.mipLevels).layerCount(1);
            }
            var cmd = entries.get(0).image.ctx.cmd();
            int actualSrcStage = anyUndefined ? unionSrcStage | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : unionSrcStage;
            vkCmdPipelineBarrier(cmd, actualSrcStage, unionDstStage, 0, null, null, imbs);
            for (var e : entries) e.image.currentLayout = e.newLayout;
        }
    }

    private final java.util.ArrayList<Long> extraViews = new java.util.ArrayList<>();

    /** Additional full-image view with a different aspect (e.g. DEPTH-only sampling view of a depth-stencil image). */
    public long createAspectView(int viewAspect) {
        try (MemoryStack stack = stackPush()) {
            var vci = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(this.image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(this.format);
            vci.subresourceRange().aspectMask(viewAspect).baseMipLevel(0).levelCount(this.mipLevels).baseArrayLayer(0).layerCount(1);
            var pView = stack.mallocLong(1);
            check(vkCreateImageView(this.ctx.vk().device, vci, null, pView), "vkCreateImageView(aspect)");
            long view = pView.get(0);
            this.extraViews.add(view);
            return view;
        }
    }

    public long allocationSize() {
        return this.allocationSize;
    }

    private void accountFreed() {
        this.freed = true;
        COUNT--;
        TOTAL_ALLOCATION_SIZE -= this.allocationSize;
    }

    /**
     * Immediate destruction is only for resources that were created as a
     * transactional replacement but never referenced by any command buffer.
     * Submitted/live images must use free() and Minecraft's retirement queue.
     */
    public void freeUnsubmittedNow() {
        if (this.freed) return;
        this.accountFreed();
        var vctx = this.ctx.vk();
        if (this.mipViews != null) {
            for (long mipView : this.mipViews) {
                if (mipView != VK_NULL_HANDLE) vkDestroyImageView(vctx.device, mipView, null);
            }
        }
        for (long extraView : this.extraViews) {
            if (extraView != VK_NULL_HANDLE) vkDestroyImageView(vctx.device, extraView, null);
        }
        if (this.view != VK_NULL_HANDLE) vkDestroyImageView(vctx.device, this.view, null);
        Vma.vmaDestroyImage(vctx.vmaAllocator, this.image, this.allocation);
    }

    public void free() {
        if (this.freed) return;
        this.accountFreed();

        int mipCount = this.mipViews == null ? 0 : this.mipViews.length;
        long[] additionalViews = new long[mipCount + this.extraViews.size()];
        int index = 0;
        if (this.mipViews != null) {
            for (long mipView : this.mipViews) additionalViews[index++] = mipView;
        }
        for (long extraView : this.extraViews) additionalViews[index++] = extraView;

        //Destroy every view before returning the image allocation to VMA, all in
        //one Minecraft submission-retirement callback.
        this.ctx.deferDestroyVmaImage(this.image, this.view, this.allocation, additionalViews);
    }

    public static int getCount() {
        return COUNT;
    }

    public static long getTotalAllocationSize() {
        return TOTAL_ALLOCATION_SIZE;
    }

    /** Simple sampler factory (nearest/clamped or nearest-mipmap for HiZ etc). */
    private record SamplerKey(long deviceAddress, boolean mipmapNearest, boolean linear) {}
    private static final java.util.Map<SamplerKey, Long> SAMPLER_CACHE = new java.util.concurrent.ConcurrentHashMap<>();

    public static long createSampler(VulkanContext ctx, boolean mipmapNearest, boolean linear) {
        var key = new SamplerKey(ctx.device.address(), mipmapNearest, linear);
        Long cached = SAMPLER_CACHE.get(key);
        if (cached != null) return cached;
        try (MemoryStack stack = stackPush()) {
            var sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST)
                    .minFilter(linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0).maxLod(mipmapNearest ? VK_LOD_CLAMP_NONE : 0.25f);
            var pSampler = stack.mallocLong(1);
            check(vkCreateSampler(ctx.device, sci, null, pSampler), "vkCreateSampler");
            long handle = pSampler.get(0);
            Long raced = SAMPLER_CACHE.putIfAbsent(key, handle);
            if (raced != null) {
                vkDestroySampler(ctx.device, handle, null);
                return raced;
            }
            return handle;
        }
    }

    /** Destroy all cached sampler handles owned by this adopted VkDevice. */
    public static void destroySamplers(VulkanContext ctx) {
        long deviceAddress = ctx.device.address();
        for (var entry : SAMPLER_CACHE.entrySet()) {
            if (entry.getKey().deviceAddress() == deviceAddress
                    && SAMPLER_CACHE.remove(entry.getKey(), entry.getValue())) {
                vkDestroySampler(ctx.device, entry.getValue(), null);
            }
        }
    }
}
