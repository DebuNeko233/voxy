package me.cortex.voxy.client.mixin.vk;

import com.mojang.blaze3d.textures.GpuSampler;
import me.cortex.voxy.client.core.IVoxyRenderSystemHolder;
import me.cortex.voxy.client.core.vk.MinecraftVkHost;
import me.cortex.voxy.client.core.vk.MinecraftVkHostAdapter;
import me.cortex.voxy.client.core.vk.compat.VitrailCompat;
import me.cortex.voxy.client.core.vk.render.VkRenderCore;
import me.cortex.voxy.common.Logger;
import net.caffeinemc.mods.sodium.client.render.SodiumWorldRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.minecraft.client.renderer.chunk.ChunkSectionLayerGroup;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.lang.ref.WeakReference;

//Pure-Vulkan render entry point after Sodium's opaque terrain pass.
@Mixin(value = SodiumWorldRenderer.class, remap = false)
public class MixinSodiumOpaqueVkFrame {
    //A Java/Vulkan failure during a frame can leave assumptions about resource
    //state invalid. Do not hammer the same broken core every subsequent frame.
    //Keep this weak: SodiumWorldRenderer can outlive a level transition, and a
    //failure latch must not retain the old VkRenderCore/world after shutdown.
    @Unique
    private WeakReference<VkRenderCore> voxy$failedVkCore = new WeakReference<>(null);

    @Inject(method = "drawChunkLayer", at = @At("TAIL"), remap = false)
    private void voxy$renderVkFrame(ChunkSectionLayerGroup group, ChunkRenderMatrices matrices,
                                    double x, double y, double z, GpuSampler sampler, CallbackInfo ci) {
        if (group != ChunkSectionLayerGroup.OPAQUE) return;
        if (!(MinecraftVkHost.get() instanceof MinecraftVkHostAdapter adapter)) return;

        var renderer = IVoxyRenderSystemHolder.getNullable();
        if (renderer == null || renderer.vkCore == null) return;
        VkRenderCore core = renderer.vkCore;
        if (this.voxy$failedVkCore.get() == core) return;

        try {
            //Vitrail holds a Vulkan dynamic-rendering pass open across Sodium's
            //terrain/sky/entity families and also re-enters this exact OPAQUE
            //method for its shadow map. Close that held pass through Vitrail's
            //own RenderPass object before any raw Voxy commands, and completely
            //skip the shadow re-entry so it cannot be mistaken for a camera frame.
            if (!VitrailCompat.prepareForWorldFrame()) return;

            core.renderFrame(group.outputTarget(), adapter, matrices, x, y, z);
            //If a new core replaced a previously failed one and renders
            //successfully, forget the old instance completely.
            this.voxy$failedVkCore.clear();
        } catch (Throwable t) {
            this.voxy$failedVkCore = new WeakReference<>(core);
            Logger.error("Voxy VK frame failed; native Vulkan rendering is disabled for this renderer instance", t);
        }
    }
}
