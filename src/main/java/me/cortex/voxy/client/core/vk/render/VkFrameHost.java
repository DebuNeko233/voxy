package me.cortex.voxy.client.core.vk.render;

import com.mojang.blaze3d.textures.GpuTextureView;
import com.mojang.blaze3d.vulkan.VulkanConst;
import com.mojang.blaze3d.vulkan.VulkanGpuTexture;
import com.mojang.blaze3d.vulkan.VulkanGpuTextureView;
import net.minecraft.client.Minecraft;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageMemoryBarrier;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Accessors for Minecraft's live Vulkan frame resources at the Sodium hook point.
public final class VkFrameHost {
    private VkFrameHost() {}

    public static long lightmapView() {
        return ((VulkanGpuTextureView) Minecraft.getInstance().gameRenderer.levelLightmap()).vkImageView();
    }

    public static long vkView(GpuTextureView view) {
        return ((VulkanGpuTextureView) view).vkImageView();
    }

    public static int vkFormat(GpuTextureView view) {
        return VulkanConst.toVk(view.texture().getFormat());
    }

    //Temporarily transitions one of Minecraft's own frame images for sampling,
    // then callers restore it to its attachment layout before returning control
    // to Blaze3D. Aspect bits are derived from the actual 26.2 GpuFormat: the
    // normal main depth target can be D32_FLOAT (depth-only), while other targets
    // may use D32_FLOAT_S8_UINT/D24_S8. Supplying STENCIL for a depth-only image
    // is invalid Vulkan usage.
    public static void transitionMcImage(VkCommandBuffer cmd, GpuTextureView view,
                                          boolean depth, int oldLayout, int newLayout) {
        try (MemoryStack stack = stackPush()) {
            var texture = view.texture();
            long image = ((VulkanGpuTexture) texture).vkImage();
            int srcStage, srcAccess, dstStage, dstAccess;
            boolean toSampled = newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (depth) {
                if (toSampled) {
                    srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                    srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    dstAccess = VK_ACCESS_SHADER_READ_BIT;
                } else {
                    srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    srcAccess = VK_ACCESS_SHADER_READ_BIT;
                    dstStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                    dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                }
            } else {
                if (toSampled) {
                    srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                    srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    dstAccess = VK_ACCESS_SHADER_READ_BIT;
                } else {
                    srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    srcAccess = VK_ACCESS_SHADER_READ_BIT;
                    dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                    dstAccess = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                }
            }

            var format = texture.getFormat();
            int aspectMask = 0;
            if (format.hasColorAspect()) aspectMask |= VK_IMAGE_ASPECT_COLOR_BIT;
            if (format.hasDepthAspect()) aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (format.hasStencilAspect()) aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            if (aspectMask == 0) {
                throw new IllegalStateException("Minecraft Vulkan texture has no usable image aspect: " + format);
            }
            if (depth && (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) == 0) {
                throw new IllegalArgumentException("Expected depth texture, got " + format);
            }
            if (!depth && (aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
                throw new IllegalArgumentException("Expected color texture, got " + format);
            }

            var imb = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(srcAccess).dstAccessMask(dstAccess)
                    .oldLayout(oldLayout).newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image);
            imb.subresourceRange()
                    .aspectMask(aspectMask)
                    .baseMipLevel(0).levelCount(VK_REMAINING_MIP_LEVELS)
                    .baseArrayLayer(0).layerCount(VK_REMAINING_ARRAY_LAYERS);
            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, imb);
        }
    }
}
