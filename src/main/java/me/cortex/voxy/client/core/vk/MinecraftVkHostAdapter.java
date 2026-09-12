package me.cortex.voxy.client.core.vk;

import com.mojang.blaze3d.vulkan.VulkanDevice;
import me.cortex.voxy.client.mixin.vk.AccessorVulkanCommandEncoder;
import me.cortex.voxy.client.mixin.vk.AccessorVulkanDevice;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

//IVkHost backed by Minecraft 26.2's live Blaze3D Vulkan device.
public final class MinecraftVkHostAdapter implements IVkHost {
    private final VulkanDevice device;

    public MinecraftVkHostAdapter(VulkanDevice device) {
        this.device = device;
    }

    @Override public VkInstance instance() { return this.device.instance().vkInstance(); }
    @Override public VkPhysicalDevice physicalDevice() { return this.device.vkDevice().getPhysicalDevice(); }
    @Override public VkDevice device() { return this.device.vkDevice(); }
    @Override public VkQueue graphicsQueue() { return this.device.graphicsQueue().vkQueue(); }
    @Override public int graphicsQueueFamily() { return this.device.graphicsQueue().queueFamilyIndex(); }

    @Override
    public VkCommandBuffer frameCommandBuffer() {
        //Minecraft 26.2 stores one persistent final commandEncoder on the device.
        //Access that exact object instead of depending on createCommandEncoder()
        // returning the same encoder implementation forever.
        var encoder = ((AccessorVulkanDevice) (Object) this.device).voxy$commandEncoder();
        return ((AccessorVulkanCommandEncoder) (Object) encoder).voxy$currentCommandBuffer();
    }
}
