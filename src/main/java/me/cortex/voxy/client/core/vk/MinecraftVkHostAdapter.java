package me.cortex.voxy.client.core.vk;

import com.mojang.blaze3d.systems.RenderSystem;
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

    private com.mojang.blaze3d.vulkan.VulkanCommandEncoder encoder() {
        return ((AccessorVulkanDevice) (Object) this.device).voxy$commandEncoder();
    }

    @Override public VkInstance instance() { return this.device.instance().vkInstance(); }
    @Override public VkPhysicalDevice physicalDevice() { return this.device.vkDevice().getPhysicalDevice(); }
    @Override public VkDevice device() { return this.device.vkDevice(); }
    @Override public VkQueue graphicsQueue() { return this.device.graphicsQueue().vkQueue(); }
    @Override public int graphicsQueueFamily() { return this.device.graphicsQueue().queueFamilyIndex(); }

    @Override
    public VkCommandBuffer frameCommandBuffer() {
        RenderSystem.assertOnRenderThread();
        var accessor = (AccessorVulkanCommandEncoder) (Object) this.encoder();
        var current = accessor.voxy$currentCommandBuffer();
        //Sodium may legitimately draw no vanilla chunk batches. In that case
        //Blaze3D may not have needed a command buffer yet, but Voxy still has LOD
        //work to record. The hook is after Sodium's RenderPass has closed, so use
        //Minecraft's own private commandBuffer() path to allocate/begin/attach a
        //primary command buffer to the current submission instead of skipping.
        return current != null ? current : accessor.voxy$ensureCommandBuffer();
    }

    @Override
    public void deferUntilSubmissionComplete(Runnable action) {
        RenderSystem.assertOnRenderThread();
        if (action == null) throw new IllegalArgumentException("retirement action is null");
        //Mojang's VulkanCommandEncoder owns a two-slot DestructionQueue. submit()
        //waits the matching timeline-semaphore value before rotating a slot, so
        //this callback runs only after every command in the relevant submission
        //has completed. That is the Vulkan-valid lifetime boundary for Voxy's
        //buffers/images/pipelines and CPU staging bookkeeping.
        this.encoder().queueForDestroy(action::run);
    }
}
