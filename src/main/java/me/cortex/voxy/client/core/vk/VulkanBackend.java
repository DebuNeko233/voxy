package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.common.Logger;

//Capability detection + lifecycle for the Vulkan backend. Voxy adopts
//Minecraft's live Vulkan device and never owns/destroys the VkDevice itself.
public final class VulkanBackend {
    private static Boolean supported;
    private static VulkanContext context;
    private static String unsupportedReason = "not probed";

    public static boolean shouldUseVulkan() {
        if (!MinecraftVkHost.isMinecraftOnVulkan()) return false;
        if (MinecraftVkHost.get() == null) {
            Logger.info("Voxy: Minecraft on Vulkan but host adapter not yet registered");
            return false;
        }
        return isSupported();
    }

    public static synchronized boolean isSupported() {
        if (supported == null) {
            try {
                Class.forName("org.lwjgl.vulkan.VK10");
                var host = MinecraftVkHost.get();
                if (host == null) {
                    supported = false;
                    unsupportedReason = "no Minecraft Vulkan host adapter";
                } else {
                    context = VulkanContext.adopt(host);
                    supported = true;
                    unsupportedReason = null;
                    Logger.info("Voxy Vulkan backend adopting Minecraft's device: " + context.deviceName);
                }
            } catch (Throwable t) {
                context = null;
                supported = false;
                unsupportedReason = t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage();
                Logger.info("Voxy Vulkan backend unavailable: " + unsupportedReason);
            }
        }
        return supported;
    }

    public static synchronized VulkanContext context() {
        if (!isSupported()) throw new IllegalStateException("Vulkan not supported: " + unsupportedReason);
        return context;
    }

    public static synchronized String statusLine() {
        if (supported == null) return "vk: unprobed";
        if (!supported || context == null) return "vk: unavailable (" + unsupportedReason + ")";
        try {
            var budget = context.deviceLocalBudget();
            return "vk: host(" + context.deviceName + "), images=" + VkImage2D.getCount()
                    + "/" + (VkImage2D.getTotalAllocationSize() >> 20) + "MiB"
                    + ", vma=" + (budget.usageBytes() >> 20) + "/" + (budget.budgetBytes() >> 20) + "MiB"
                    + ", free=" + (budget.availableBytes() >> 20) + "MiB)";
        } catch (RuntimeException | Error budgetFailure) {
            //Debug rendering must never be able to take the renderer down just
            //because a driver/VMA budget query is temporarily unavailable.
            return "vk: host(" + context.deviceName + "), images=" + VkImage2D.getCount()
                    + "/" + (VkImage2D.getTotalAllocationSize() >> 20) + "MiB, vma=n/a)";
        }
    }

    public static synchronized void shutdown() {
        //Detach Java-visible state first. If a native cleanup call below fails,
        //no later caller can reacquire a stale context pointing at a VkDevice
        //Minecraft may already be in the process of closing.
        VulkanContext oldContext = context;
        context = null;
        supported = null;
        unsupportedReason = "not probed";
        if (oldContext != null) {
            oldContext.destroy();
        }
    }

    private VulkanBackend() {}
}
