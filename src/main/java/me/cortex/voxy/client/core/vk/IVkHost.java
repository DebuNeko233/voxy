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

    /** Minecraft-owned VMA allocator associated with the adopted VkDevice. */
    long vmaAllocator();

    /**
     * Minecraft's primary graphics command buffer for the current submission.
     * The host ensures one exists when Voxy asks at a safe, render-pass-free
     * integration point.
     */
    VkCommandBuffer frameCommandBuffer();

    /**
     * Submit Minecraft's current Vulkan encoder batch and synchronously wait for
     * that exact timeline-semaphore submit to complete. This is used only for
     * construction/readback work that truly needs CPU-visible completion.
     */
    void submitAndWaitCurrent();

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
