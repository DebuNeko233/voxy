package me.cortex.voxy.client.core.vk;

import org.lwjgl.vulkan.VK10;

public final class VkUtil {
    private VkUtil() {}

    /** Keeps the VkResult available so callers can recover only from errors they
     * actually understand (for example VMA budget/OOM), without hiding format,
     * descriptor, or image-view programming errors behind a generic retry. */
    public static final class VulkanCallException extends IllegalStateException {
        private final String operation;
        private final int result;

        public VulkanCallException(String operation, int result) {
            super("Vulkan call failed: " + operation + " -> " + result);
            this.operation = operation;
            this.result = result;
        }

        public String operation() {
            return this.operation;
        }

        public int result() {
            return this.result;
        }

        public boolean isOutOfMemory() {
            return this.result == VK10.VK_ERROR_OUT_OF_DEVICE_MEMORY
                    || this.result == VK10.VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    public static int check(int vkResult, String what) {
        if (vkResult != VK10.VK_SUCCESS) {
            throw new VulkanCallException(what, vkResult);
        }
        return vkResult;
    }
}
