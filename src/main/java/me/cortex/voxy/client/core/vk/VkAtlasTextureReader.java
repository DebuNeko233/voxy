package me.cortex.voxy.client.core.vk;

import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.vulkan.VulkanGpuTexture;
import me.cortex.voxy.client.core.model.bakery.IAtlasTextureReader;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageMemoryBarrier;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Synchronous Vulkan block-atlas readback used during model-bakery creation.
public final class VkAtlasTextureReader extends IAtlasTextureReader {
    private final VkFrameCtx frameCtx;

    public VkAtlasTextureReader(VkFrameCtx frameCtx) {
        this.frameCtx = frameCtx;
    }

    @Override
    public int[] read(GpuTexture atlas, int width, int height) {
        long image = ((VulkanGpuTexture) atlas).vkImage();
        long size = (long) width * height * 4;
        var staging = new VkBuffer(this.frameCtx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        try {
            var cmd = this.frameCtx.cmd();

            //Minecraft 26.2 keeps VulkanGpuTexture images in GENERAL layout and
            //also binds/render-targets them as GENERAL. Do not transition the
            //block atlas behind Blaze3D's back. A same-layout barrier is enough
            //to make prior texture writes visible to our transfer read.
            barrierGeneral(cmd, image,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);

            try (MemoryStack stack = stackPush()) {
                var region = VkBufferImageCopy.calloc(1, stack)
                        .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
                region.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.imageOffset().set(0, 0, 0);
                region.imageExtent().set(width, height, 1);
                vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, region);
            }

            //Keep the image in the layout Blaze3D expects and make the transfer
            //read complete before any subsequent Minecraft access in this batch.
            barrierGeneral(cmd, image,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

            this.frameCtx.flushImmediate();

            var out = new int[width * height];
            long ptr = staging.map();
            MemoryUtil.memIntBuffer(ptr, out.length).get(out);
            return out;
        } finally {
            staging.free();
            //The copy was synchronously submitted through Minecraft's encoder.
            //waitIdle here is only teardown hygiene for this one-shot allocation;
            //native destruction itself still follows the host destruction queue.
            this.frameCtx.waitIdleRetireAll();
        }
    }

    private static void barrierGeneral(VkCommandBuffer cmd, long image,
                                       int srcStage, int srcAccess,
                                       int dstStage, int dstAccess) {
        try (MemoryStack stack = stackPush()) {
            var imb = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(srcAccess)
                    .dstAccessMask(dstAccess)
                    .oldLayout(VK_IMAGE_LAYOUT_GENERAL)
                    .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image);
            imb.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(VK_REMAINING_MIP_LEVELS)
                    .baseArrayLayer(0).layerCount(VK_REMAINING_ARRAY_LAYERS);
            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, imb);
        }
    }
}
