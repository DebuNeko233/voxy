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
    private void voxy$shutdownBeforeDeviceClose(CallbackInfo ci) {
        //This is the final safety net for shutdown ordering. LevelRenderer.close
        //normally tears Voxy down first, but MC must never destroy its VkDevice
        //while Voxy still owns buffers/images/pipelines or its command pool.
        //Keep the host registered until teardown is complete so VkRenderCore can
        //prove the adopted device is still alive and safely destroy its objects.
        try {
            var holder = IVoxyRenderSystemHolder.getNullableHolder();
            if (holder != null) {
                holder.voxy$shutdownRenderer();
            }
        } catch (Throwable t) {
            Logger.error("Voxy: failed to shut renderer down before Vulkan device close", t);
        }

        try {
            VulkanBackend.shutdown();
        } catch (Throwable t) {
            Logger.error("Voxy: failed to release Vulkan backend before Minecraft device close", t);
        } finally {
            MinecraftVkHost.clear();
        }
    }
}
