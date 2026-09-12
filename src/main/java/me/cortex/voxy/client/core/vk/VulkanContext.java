package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaBudget;
import org.lwjgl.vulkan.VK11;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupProperties;
import org.lwjgl.vulkan.VkQueue;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Wraps Minecraft's already-created Vulkan device. Voxy never creates or
//reconfigures the logical device and no longer owns a graphics command pool:
//all primary recording/submission and memory allocation are routed through
//Minecraft's encoder and VMA allocator.
public final class VulkanContext {
    private static final long MIB = 1024L * 1024L;

    private final IVkHost host;
    public final VkInstance instance;
    public final VkPhysicalDevice physicalDevice;
    public final VkDevice device;
    public final VkQueue queue;
    public final int queueFamily;
    public final long vmaAllocator;
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
    public final int deviceLocalHeapIndex;
    public final boolean needsSampleMaskDiscard;
    private VkPhysicalDeviceSubgroupProperties subgroupProps;

    public static VulkanContext adopt(IVkHost host) { return new VulkanContext(host); }

    private VulkanContext(IVkHost host) {
        this.host = host;
        this.instance = host.instance();
        this.physicalDevice = host.physicalDevice();
        this.device = host.device();
        this.queue = host.graphicsQueue();
        this.queueFamily = host.graphicsQueueFamily();
        this.vmaAllocator = host.vmaAllocator();
        if (this.vmaAllocator == 0L) {
            throw new IllegalStateException("Minecraft Vulkan host exposed a null VMA allocator");
        }

        //Physical-device support is not proof that Minecraft enabled this
        //optional feature on the adopted logical device, so stay on the
        //fixed-count compatibility path until Blaze3D exposes that fact.
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
        int localHeapIndex = -1;
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
                if ((heap.flags() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && heap.size() > localHeapBytes) {
                    localHeapBytes = heap.size();
                    localHeapIndex = i;
                }
            }
        } catch (RuntimeException | Error failure) {
            if (this.subgroupProps != null) {
                this.subgroupProps.free();
                this.subgroupProps = null;
            }
            throw failure;
        }
        if (localHeapIndex < 0) {
            if (this.subgroupProps != null) {
                this.subgroupProps.free();
                this.subgroupProps = null;
            }
            throw new IllegalStateException("Vulkan device exposes no device-local memory heap");
        }
        this.integratedGpu = integrated;
        this.deviceLocalHeapBytes = localHeapBytes;
        this.deviceLocalHeapIndex = localHeapIndex;
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

    /** Minecraft-owned command buffer for the current graphics submission. */
    public VkCommandBuffer hostCommandBuffer() {
        return this.host.frameCommandBuffer();
    }

    /** Submit and synchronously wait for the current Minecraft encoder batch. */
    public void submitAndWaitCurrent() {
        this.host.submitAndWaitCurrent();
    }

    /**
     * Uses Minecraft 26.2's own timeline-semaphore-backed DestructionQueue.
     * The action therefore runs only after the associated graphics submission
     * has completed, satisfying Vulkan object-lifetime rules.
     */
    public void deferUntilSubmissionComplete(Runnable action) {
        this.host.deferUntilSubmissionComplete(action);
    }

    /** Advance every host destruction slot through Minecraft's own submit path. */
    public void drainDeferredDestruction() {
        this.host.drainDeferredDestruction();
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

    public record DeviceLocalBudget(long usageBytes, long budgetBytes, long availableBytes) {}

    /** Snapshot of Minecraft's VMA accounting for the largest device-local heap. */
    public DeviceLocalBudget deviceLocalBudget() {
        try (MemoryStack stack = stackPush()) {
            var budgets = VmaBudget.calloc(VK_MAX_MEMORY_HEAPS, stack);
            Vma.vmaGetHeapBudgets(this.vmaAllocator, budgets);
            var heap = budgets.get(this.deviceLocalHeapIndex);
            long budget = heap.budget();
            long usage = heap.usage();
            if (budget <= 0) budget = this.deviceLocalHeapBytes;
            if (usage < 0) usage = 0;
            return new DeviceLocalBudget(usage, budget, Math.max(0L, budget - usage));
        }
    }

    /**
     * Geometry allocation policy based on live VMA budget, not just total VRAM.
     * The query happens after Voxy's atlas and staging buffers exist, so it also
     * accounts for Minecraft/other current allocations visible to the shared VMA.
     */
    public long recommendedGeometryCapacityBytes() {
        final long floor = 64L * MIB;
        final long policyMaximum = (this.integratedGpu ? 1024L : 2048L) * MIB;

        long policyTarget;
        if (this.deviceLocalHeapBytes > 0) {
            policyTarget = this.deviceLocalHeapBytes / (this.integratedGpu ? 8 : 4);
        } else {
            policyTarget = this.integratedGpu ? (512L * MIB) : (1024L * MIB);
        }
        policyTarget = Math.min(policyMaximum, Math.max(floor, policyTarget));

        DeviceLocalBudget live = this.deviceLocalBudget();
        if (live.budgetBytes() > 0) {
            //Keep explicit headroom for Minecraft's later frame targets, Sodium,
            //other mods, and allocation spikes. Then let geometry consume at most
            //half of what remains instead of racing the heap to 100% utilization.
            long reserve = this.integratedGpu
                    ? Math.max(256L * MIB, live.budgetBytes() / 8)
                    : Math.max(512L * MIB, live.budgetBytes() / 10);
            long afterReserve = Math.max(0L, live.availableBytes() - reserve);
            long budgetTarget = afterReserve / 2;
            policyTarget = Math.min(policyTarget, Math.max(floor, budgetTarget));
        }

        return policyTarget & ~7L;
    }

    public void destroy() {
        vkDeviceWaitIdle(this.device);
        VkImage2D.destroySamplers(this);
        VkShaderPipeline.destroyCachedLayouts(this);
        if (this.subgroupProps != null) {
            this.subgroupProps.free();
            this.subgroupProps = null;
        }
    }
}
