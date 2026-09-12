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

    /**
     * Minecraft 26.2's VulkanGpuTexture keeps images in GENERAL layout for
     * sampling and attachments. Voxy must not transition those images behind
     * Blaze3D's back; it only inserts same-layout memory dependencies.
     */
    public static void barrierMcImageForSampling(VkCommandBuffer cmd, GpuTextureView view, boolean depth) {
        int dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        barrierMcImageGeneral(cmd, view, depth,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                dstStage,
                VK_ACCESS_SHADER_READ_BIT);
    }

    public static void barrierMcImageForAttachment(VkCommandBuffer cmd, GpuTextureView view, boolean depth) {
        int dstStage;
        int dstAccess;
        if (depth) {
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        } else {
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dstAccess = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }
        barrierMcImageGeneral(cmd, view, depth,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                dstStage,
                dstAccess);
    }

    private static void barrierMcImageGeneral(VkCommandBuffer cmd, GpuTextureView view, boolean depth,
                                               int srcStage, int srcAccess,
                                               int dstStage, int dstAccess) {
        try (MemoryStack stack = stackPush()) {
            var texture = view.texture();
            long image = ((VulkanGpuTexture) texture).vkImage();
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
                    .srcAccessMask(srcAccess)
                    .dstAccessMask(dstAccess)
                    .oldLayout(VK_IMAGE_LAYOUT_GENERAL)
                    .newLayout(VK_IMAGE_LAYOUT_GENERAL)
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
