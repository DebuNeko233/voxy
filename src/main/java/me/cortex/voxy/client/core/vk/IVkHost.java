package me.cortex.voxy.client.core.vk;

import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;

/**
 * The pure-Vulkan integration seam. Voxy adopts Minecraft's Vulkan device and
 * records into the game's live frame rather than creating a second VkDevice.
 */
public interface IVkHost {
    VkInstance instance();
    VkPhysicalDevice physicalDevice();
    VkDevice device();
    VkQueue graphicsQueue();
    int graphicsQueueFamily();

    /** Command buffer currently recording for this frame at Voxy's injection point. */
    VkCommandBuffer frameCommandBuffer();

    /**
     * Run {@code action} only after the Minecraft Vulkan submission associated
     * with the host's current destruction slot is known complete.
     *
     * Minecraft 26.2 already tracks submit completion with a timeline semaphore
     * and rotates its DestructionQueue only after the corresponding submit has
     * completed. Voxy must use that lifecycle boundary rather than inferring
     * completion from an in-command-buffer VkEvent.
     */
    void deferUntilSubmissionComplete(Runnable action);
}
