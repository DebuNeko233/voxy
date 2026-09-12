package me.cortex.voxy.client.mixin.minecraft;

import me.cortex.voxy.client.core.IVoxyRenderSystemHolder;
import me.cortex.voxy.client.core.vk.compat.VitrailCompat;
import me.cortex.voxy.common.Logger;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.extract.LevelExtractor;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(LevelExtractor.class)
public class MixinLevelExtractor {
    @Shadow
    @Final
    private LevelRenderer levelRenderer;

    @Inject(method = "setLevel", at = @At("HEAD"))
    private void voxy$onSetLevel(ClientLevel level, CallbackInfo cir) {
        ((IVoxyRenderSystemHolder)this.levelRenderer).voxy$setWorld(level);
    }

    @Inject(method = "allChanged", at = @At("HEAD"))
    private void voxy$reload(CallbackInfo cir) {
        var holder = (IVoxyRenderSystemHolder)this.levelRenderer;
        var renderer = holder.voxy$getRenderSystem();

        //Vitrail uses allChanged as a frame-boundary request to rebuild its own
        //Sodium terrain mesh/vertex format when a shader pack changes requirements.
        //The pure-Vulkan Voxy renderer owns independent geometry/model buffers, so
        //tearing down its ~GiB-scale core for those calls is unnecessary and causes
        //visible LOD flashes plus large transient VMA pressure during pack loading.
        //Keep normal Minecraft/resource-pack/F3+A allChanged behavior unchanged so
        //a real block-model/texture-atlas reload still refreshes Voxy resources.
        if (renderer != null && renderer.isVulkanBackend() && VitrailCompat.isVitrailDrivenWorldRebuild()) {
            Logger.info("Voxy VK: keeping native renderer across Vitrail-only Sodium terrain rebuild");
            return;
        }

        holder.voxy$shutdownRenderer();
        holder.voxy$createRenderer();
    }
}
