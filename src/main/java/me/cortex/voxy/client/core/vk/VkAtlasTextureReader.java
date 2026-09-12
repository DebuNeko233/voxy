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
            transition(cmd, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            try {
                try (MemoryStack stack = stackPush()) {
                    var region = VkBufferImageCopy.calloc(1, stack)
                            .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
                    region.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                            .mipLevel(0).baseArrayLayer(0).layerCount(1);
                    region.imageOffset().set(0, 0, 0);
                    region.imageExtent().set(width, height, 1);
                    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, region);
                }
            } finally {
                //Even a Java-side failure while preparing the copy must not leave
                // Minecraft's atlas recorded in TRANSFER_SRC layout.
                transition(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            this.frameCtx.flushImmediate();

            var out = new int[width * height];
            long ptr = staging.map();
            MemoryUtil.memIntBuffer(ptr, out.length).get(out);
            return out;
        } finally {
            staging.free();
            //This staging allocation is one-shot and can be large. The atlas copy
            // is synchronous, so retire it immediately rather than carrying it
            // until the first rendered frame happens to advance the retire queue.
            this.frameCtx.waitIdleRetireAll();
        }
    }

    private static void transition(VkCommandBuffer cmd, long image, int oldLayout, int newLayout) {
        try (MemoryStack stack = stackPush()) {
            var imb = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT)
                    .dstAccessMask(VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT)
                    .oldLayout(oldLayout).newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image);
            imb.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(VK_REMAINING_MIP_LEVELS)
                    .baseArrayLayer(0).layerCount(VK_REMAINING_ARRAY_LAYERS);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, null, null, imb);
        }
    }
}
