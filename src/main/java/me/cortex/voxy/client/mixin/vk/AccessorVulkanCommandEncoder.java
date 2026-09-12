package me.cortex.voxy.client.mixin.vk;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

//Exposes Minecraft 26.2's persistent Vulkan command encoder state. Voxy hooks
// only after Sodium's RenderPass has closed, so it is safe to ask the encoder to
// ensure its primary command buffer exists before recording Voxy commands.
@Mixin(VulkanCommandEncoder.class)
public interface AccessorVulkanCommandEncoder {
    @Accessor("currentCommandBuffer")
    VkCommandBuffer voxy$currentCommandBuffer();

    @Invoker("commandBuffer")
    VkCommandBuffer voxy$ensureCommandBuffer();
}
