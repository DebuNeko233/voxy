package me.cortex.voxy.client.mixin.vk;

import com.mojang.blaze3d.vulkan.VulkanDevice;
import me.cortex.voxy.client.core.IVoxyRenderSystemHolder;
import me.cortex.voxy.client.core.vk.MinecraftVkHost;
import me.cortex.voxy.client.core.vk.MinecraftVkHostAdapter;
import me.cortex.voxy.client.core.vk.VulkanBackend;
import me.cortex.voxy.common.Logger;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

//The Blaze3D-VK adapter. When MC 26.2 initialises its own Vulkan device
// (Graphics API = Vulkan), this registers an IVkHost backed by that live
// device so Voxy adopts it (no second VkDevice) and records into MC's frame.
@Mixin(VulkanDevice.class)
public class MixinVulkanDevice {
    @Inject(method = "<init>", at = @At("TAIL"))
    private void voxy$registerHost(CallbackInfo ci) {
        try {
            MinecraftVkHost.register(new MinecraftVkHostAdapter((VulkanDevice) (Object) this));
            Logger.info("Voxy: adopted Minecraft's Vulkan device (pure-VK host mode available)");
        } catch (Throwable t) {
            //Never let Voxy's adapter break MC's device creation. With no host
            // registered Voxy stays inactive.
            Logger.warn("Voxy: failed to register Vulkan host adapter, Voxy will be inactive: " + t);
            MinecraftVkHost.clear();
        }
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void voxy$shutdownRendererBeforeDeviceClose(CallbackInfo ci) {
        //Phase 1: stop Voxy while Minecraft's command encoder is still alive.
        //VkBuffer/VkImage/VkPipeline frees can therefore enqueue destruction
        //callbacks onto Blaze3D's submission-safe DestructionQueue.
        try {
            var holder = IVoxyRenderSystemHolder.getNullableHolder();
            if (holder != null) {
                holder.voxy$shutdownRenderer();
            }
        } catch (Throwable t) {
            Logger.error("Voxy: failed to shut renderer down before Vulkan device close", t);
        }
    }

    @Inject(
            method = "close",
            at = @At(
                    value = "INVOKE",
                    target = "Lcom/mojang/blaze3d/vulkan/VulkanCommandEncoder;destroy()V",
                    shift = At.Shift.AFTER
            )
    )
    private void voxy$shutdownBackendAfterSubmissionDrain(CallbackInfo ci) {
        //Phase 2: VulkanCommandEncoder.destroy() has waited the graphics queue
        //idle and drained its destruction queues. Voxy can now destroy only its
        //context-owned static caches and private immediate command pool without
        //racing any submitted Minecraft command buffer.
        try {
            VulkanBackend.shutdown();
        } catch (Throwable t) {
            Logger.error("Voxy: failed to release Vulkan backend after Minecraft submission drain", t);
        } finally {
            MinecraftVkHost.clear();
        }
    }
}
