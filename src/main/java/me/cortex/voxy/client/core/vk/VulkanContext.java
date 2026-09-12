package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK11;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupProperties;
import org.lwjgl.vulkan.VkQueue;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Wraps Minecraft's already-created Vulkan device. Voxy never creates or
// reconfigures the logical device, so optional features are usable only when
// Minecraft itself is known to have enabled them.
public final class VulkanContext {
    private final IVkHost host;
    public final VkInstance instance;
    public final VkPhysicalDevice physicalDevice;
    public final VkDevice device;
    public final VkQueue queue;
    public final int queueFamily;
    public final boolean hasDrawIndirectCount;

    private static final boolean ENABLE_SUBGROUP_PATHS = false;
    //VkSubgroupFeatureFlagBits values from Vulkan 1.1. Keep these local because
    //the LWJGL class bundled by a given Minecraft runtime may not re-export the
    //named constants even though VkPhysicalDeviceSubgroupProperties is present.
    private static final int SUBGROUP_FEATURE_BASIC_BIT = 0x00000001;
    private static final int SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004;
    private static final int SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040;

    public final boolean subgroupArithmetic;
    public final int subgroupSize;
    public final String deviceName;
    public final boolean integratedGpu;
    public final long deviceLocalHeapBytes;
    public final boolean needsSampleMaskDiscard;
    public final long commandPool;
    private VkPhysicalDeviceSubgroupProperties subgroupProps;

    public static VulkanContext adopt(IVkHost host) { return new VulkanContext(host); }

    private VulkanContext(IVkHost host) {
        this.host = host;
        this.instance = host.instance();
        this.physicalDevice = host.physicalDevice();
        this.device = host.device();
        this.queue = host.graphicsQueue();
        this.queueFamily = host.graphicsQueueFamily();

        //Physical-device support is not proof that Minecraft enabled this
        // optional feature on the adopted logical device, so stay on the
        // fixed-count compatibility path until Blaze3D exposes that fact.
        this.hasDrawIndirectCount = false;

        var subgroup = querySubgroupProperties(this.physicalDevice);
        this.subgroupProps = subgroup;
        this.subgroupSize = subgroup != null ? subgroup.subgroupSize() : 1;
        int ops = subgroup != null ? subgroup.supportedOperations() : 0;
        int stages = subgroup != null ? subgroup.supportedStages() : 0;
        int needOps = SUBGROUP_FEATURE_ARITHMETIC_BIT | SUBGROUP_FEATURE_BASIC_BIT | SUBGROUP_FEATURE_CLUSTERED_BIT;
        boolean deviceSupportsSubgroups = (ops & needOps) == needOps
                && (stages & VK_SHADER_STAGE_COMPUTE_BIT) != 0
                && this.subgroupSize >= 16;
        this.subgroupArithmetic = ENABLE_SUBGROUP_PATHS && deviceSupportsSubgroups;

        String name;
        int vendorId;
        boolean integrated;
        long localHeapBytes = 0;
        long createdCommandPool = VK_NULL_HANDLE;
        try (MemoryStack stack = stackPush()) {
            var props = VkPhysicalDeviceProperties.calloc(stack);
            vkGetPhysicalDeviceProperties(this.physicalDevice, props);
            name = props.deviceNameString();
            vendorId = props.vendorID();
            integrated = props.deviceType() == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

            var memProps = VkPhysicalDeviceMemoryProperties.calloc(stack);
            vkGetPhysicalDeviceMemoryProperties(this.physicalDevice, memProps);
            for (int i = 0; i < memProps.memoryHeapCount(); i++) {
                var heap = memProps.memoryHeaps(i);
                if ((heap.flags() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                    localHeapBytes = Math.max(localHeapBytes, heap.size());
                }
            }

            var cpci = VkCommandPoolCreateInfo.calloc(stack).sType$Default()
                    .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(this.queueFamily);
            var pPool = stack.callocLong(1);
            int result = vkCreateCommandPool(this.device, cpci, null, pPool);
            createdCommandPool = pPool.get(0);
            check(result, "vkCreateCommandPool(adopted)");
        } catch (RuntimeException | Error failure) {
            if (createdCommandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(this.device, createdCommandPool, null);
            }
            if (this.subgroupProps != null) {
                this.subgroupProps.free();
                this.subgroupProps = null;
            }
            throw failure;
        }
        this.commandPool = createdCommandPool;
        this.integratedGpu = integrated;
        this.deviceLocalHeapBytes = localHeapBytes;
        boolean macOS = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac");
        this.needsSampleMaskDiscard = macOS && vendorId != 0x106B;
        this.deviceName = name + " (MC host)";
        Logger.info("Voxy Vulkan context adopted Minecraft device: " + this.deviceName
                + " (drawIndirectCount=" + this.hasDrawIndirectCount + " [conservative adopted-device policy]"
                + ", integratedGpu=" + this.integratedGpu
                + ", deviceLocalHeapMiB=" + (this.deviceLocalHeapBytes >> 20)
                + ", sampleMaskDiscard=" + this.needsSampleMaskDiscard
                + ", subgroupArithmetic=" + this.subgroupArithmetic
                + " (deviceCapable=" + deviceSupportsSubgroups + ", gate=" + ENABLE_SUBGROUP_PATHS + ")"
                + ", subgroupSize=" + this.subgroupSize + ")");
    }

    /**
     * Uses Minecraft 26.2's own timeline-semaphore-backed DestructionQueue.
     * The action therefore runs only after the associated graphics submission
     * has completed, satisfying Vulkan object-lifetime rules.
     */
    public void deferUntilSubmissionComplete(Runnable action) {
        this.host.deferUntilSubmissionComplete(action);
    }

    private static VkPhysicalDeviceSubgroupProperties querySubgroupProperties(VkPhysicalDevice pd) {
        try (MemoryStack stack = stackPush()) {
            var sg = VkPhysicalDeviceSubgroupProperties.calloc(stack).sType$Default();
            var p2 = VkPhysicalDeviceProperties2.calloc(stack).sType$Default().pNext(sg.address());
            VK11.vkGetPhysicalDeviceProperties2(pd, p2);
            var copy = VkPhysicalDeviceSubgroupProperties.malloc();
            copy.set(sg);
            return copy;
        }
    }

    private long storageAlign = -1;
    public long storageBufferOffsetAlignment() {
        if (this.storageAlign == -1) {
            try (MemoryStack stack = stackPush()) {
                var props = VkPhysicalDeviceProperties.calloc(stack);
                vkGetPhysicalDeviceProperties(this.physicalDevice, props);
                this.storageAlign = props.limits().minStorageBufferOffsetAlignment();
            }
        }
        return this.storageAlign;
    }

    private long uniformAlign = -1;
    public long uniformBufferOffsetAlignment() {
        if (this.uniformAlign == -1) {
            try (MemoryStack stack = stackPush()) {
                var props = VkPhysicalDeviceProperties.calloc(stack);
                vkGetPhysicalDeviceProperties(this.physicalDevice, props);
                this.uniformAlign = props.limits().minUniformBufferOffsetAlignment();
            }
        }
        return this.uniformAlign;
    }

    public long recommendedGeometryCapacityBytes() {
        final long minimum = 256L << 20;
        final long maximum = (this.integratedGpu ? 1024L : 2048L) << 20;
        long target;
        if (this.deviceLocalHeapBytes > 0) {
            target = this.deviceLocalHeapBytes / (this.integratedGpu ? 8 : 4);
        } else {
            target = this.integratedGpu ? (512L << 20) : (1024L << 20);
        }
        target = Math.max(minimum, Math.min(maximum, target));
        return target & ~7L;
    }

    private VkPhysicalDeviceMemoryProperties memoryProperties;
    public int findMemoryType(int typeBits, int required) {
        if (this.memoryProperties == null) {
            this.memoryProperties = VkPhysicalDeviceMemoryProperties.malloc();
            vkGetPhysicalDeviceMemoryProperties(this.physicalDevice, this.memoryProperties);
        }
        var mem = this.memoryProperties;
        for (int i = 0; i < mem.memoryTypeCount(); i++) {
            if ((typeBits & (1 << i)) != 0 && (mem.memoryTypes(i).propertyFlags() & required) == required) return i;
        }
        throw new IllegalStateException("No suitable VK memory type");
    }

    public void destroy() {
        vkDeviceWaitIdle(this.device);
        VkImage2D.destroySamplers(this);
        VkShaderPipeline.destroyCachedLayouts(this);
        vkDestroyCommandPool(this.device, this.commandPool, null);
        if (this.subgroupProps != null) {
            this.subgroupProps.free();
            this.subgroupProps = null;
        }
        if (this.memoryProperties != null) {
            this.memoryProperties.free();
            this.memoryProperties = null;
        }
    }
}
