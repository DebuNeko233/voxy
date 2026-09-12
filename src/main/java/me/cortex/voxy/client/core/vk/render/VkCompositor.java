package me.cortex.voxy.client.core.vk.render;

import com.mojang.blaze3d.textures.GpuTextureView;
import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkCmd;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
import me.cortex.voxy.client.core.vk.VkImage2D;
import me.cortex.voxy.client.core.vk.VkShaderPipeline;
import me.cortex.voxy.client.core.vk.VkShaderSource;
import me.cortex.voxy.client.core.vk.VkUploadStream;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkRect2D;
import org.lwjgl.vulkan.VkRenderingAttachmentInfoKHR;
import org.lwjgl.vulkan.VkRenderingInfoKHR;

import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRDynamicRendering.vkCmdBeginRenderingKHR;
import static org.lwjgl.vulkan.KHRDynamicRendering.vkCmdEndRenderingKHR;
import static org.lwjgl.vulkan.VK10.*;

//The two fullscreen passes bracketing Voxy's VK frame, mirroring the GL
//NormalRenderPipeline:
//
// SETUP — clear Voxy's offscreen depth-stencil (depth=clear, stencil=1) and
// colour, then copy MC's depth in (transformed into Voxy's projection space by
// the fragment shader) writing stencil=0 where vanilla terrain exists. LOD
// terrain then renders with stencil==1 only.
//
// COMPOSITE — alpha-blend Voxy's offscreen colour into MC's frame, emitting
// depth transformed back into vanilla's projection space, with environmental
// fog applied.
public class VkCompositor {
    private final VkFrameCtx ctx;
    private final VkUploadStream uploadStream;
    private final RenderProperties properties;
    private final boolean useEnvFog;

    private final VkBuffer compositeParams;
    private final long depthSampler;
    private final long colourSampler;

    private VkShaderPipeline depthSetup;
    private VkShaderPipeline composite;
    private int setupDepthFormat = -1;
    private int compositeColorFormat = -1, compositeDepthFormat = -1;

    private final Matrix4f scratchA = new Matrix4f();

    public VkCompositor(VkFrameCtx ctx, VkUploadStream uploadStream, RenderProperties properties, boolean useEnvFog) {
        this.ctx = ctx;
        this.uploadStream = uploadStream;
        this.properties = properties;
        this.useEnvFog = useEnvFog;

        VkBuffer createdParams = null;
        long createdDepthSampler;
        long createdColourSampler;
        try {
            createdParams = new VkBuffer(ctx, 256).zero();
            ctx.flushImmediate();
            createdDepthSampler = VkImage2D.createSampler(ctx.vk(), false, false);
            createdColourSampler = VkImage2D.createSampler(ctx.vk(), false, false);
        } catch (RuntimeException | Error failure) {
            if (createdParams != null) {
                createdParams.free();
                ctx.waitIdleRetireAll();
            }
            throw failure;
        }
        this.compositeParams = createdParams;
        this.depthSampler = createdDepthSampler;
        this.colourSampler = createdColourSampler;
    }

    private void ensureSetupPipeline(VkViewportRT viewport) {
        int targetDepthFormat = viewport.viewport.depthStencil.format;
        if (this.depthSetup != null && this.setupDepthFormat == targetDepthFormat) return;

        var d = new VkShaderPipeline.GfxDesc();
        d.name = "depth-setup";
        d.vertGlsl = VkShaderSource.load("voxy:post/fullscreen2.vert", VkShaderSource.defs().props(this.properties).build());
        d.fragGlsl = VkShaderSource.load("voxy:post/setup_stencil_depth.frag", VkShaderSource.defs().props(this.properties).build());
        d.pushConstantBytes = 8;
        d.colorFormat = viewport.viewport.colour.format;
        d.depthFormat = targetDepthFormat;
        d.stencilFormat = targetDepthFormat;
        d.depthTest = true;
        d.depthWrite = true;
        d.depthCompare = VK_COMPARE_OP_ALWAYS;
        d.colorWrite = false;
        d.blend = false;
        d.stencilWriteAlways1 = true;
        d.stencilWriteRef = 0;
        d.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        d.bindings = List.of(VkShaderPipeline.sampler(0));

        VkShaderPipeline replacement = new VkShaderPipeline(this.ctx, d);
        VkShaderPipeline old = this.depthSetup;
        this.depthSetup = replacement;
        this.setupDepthFormat = targetDepthFormat;
        if (old != null) old.free();
    }

    private void ensureCompositePipeline(int mcColorFormat, int mcDepthFormat) {
        if (this.composite != null && this.compositeColorFormat == mcColorFormat && this.compositeDepthFormat == mcDepthFormat) return;

        var d = new VkShaderPipeline.GfxDesc();
        d.name = "composite";
        d.vertGlsl = VkShaderSource.load("voxy:post/fullscreen2.vert", VkShaderSource.defs().props(this.properties).build());
        d.fragGlsl = VkShaderSource.load("voxy:post/blit_texture_depth_cutout.frag", VkShaderSource.defs().props(this.properties)
                .def("EMIT_COLOUR")
                .defIf("USE_ENV_FOG", this.useEnvFog)
                .build());
        d.colorFormat = mcColorFormat;
        d.depthFormat = mcDepthFormat;
        d.stencilFormat = VK_FORMAT_UNDEFINED;
        d.depthTest = true;
        d.depthWrite = true;
        d.depthCompare = VkCmd.closerEqual(this.properties);
        d.blend = true;
        d.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        d.bindings = List.of(VkShaderPipeline.sampler(0), VkShaderPipeline.ubo(1), VkShaderPipeline.sampler(3));

        VkShaderPipeline replacement = new VkShaderPipeline(this.ctx, d);
        VkShaderPipeline old = this.composite;
        this.composite = replacement;
        this.compositeColorFormat = mcColorFormat;
        this.compositeDepthFormat = mcDepthFormat;
        if (old != null) old.free();
    }

    public record VkViewportRT(VkViewport viewport,
                               GpuTextureView mcColour, GpuTextureView mcDepth,
                               int mcWidth, int mcHeight) {}

    public void setupDepthStencil(VkViewportRT rt) {
        this.ensureSetupPipeline(rt);
        var cmd = this.ctx.cmd();
        var viewport = rt.viewport;

        VkImage2D.transitionBatch(java.util.List.of(
                new VkImage2D.BatchEntry(viewport.colour, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                new VkImage2D.BatchEntry(viewport.depthStencil, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)),
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

        //Minecraft-owned images remain in GENERAL for their entire Blaze3D
        //lifetime. Only establish the dependency needed for shader sampling.
        VkFrameHost.barrierMcImageForSampling(cmd, rt.mcDepth, true);

        boolean rendering = false;
        try {
            try (MemoryStack stack = stackPush()) {
                //pColorAttachments takes a Buffer, while pDepthAttachment and
                //pStencilAttachment take one VkRenderingAttachmentInfo struct.
                var colorAttach = VkRenderingAttachmentInfoKHR.calloc(1, stack).sType$Default()
                        .imageView(viewport.colour.view)
                        .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                colorAttach.clearValue().color().float32(0, 0).float32(1, 0).float32(2, 0).float32(3, 0);
                var depthAttach = VkRenderingAttachmentInfoKHR.calloc(stack).sType$Default()
                        .imageView(viewport.depthStencil.view)
                        .imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                depthAttach.clearValue().depthStencil().depth(this.properties.clearDepth()).stencil(1);
                var stencilAttach = VkRenderingAttachmentInfoKHR.calloc(stack).sType$Default()
                        .imageView(viewport.depthStencil.view)
                        .imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                stencilAttach.clearValue().depthStencil().depth(this.properties.clearDepth()).stencil(1);
                var info = VkRenderingInfoKHR.calloc(stack).sType$Default()
                        .renderArea(VkRect2D.calloc(stack).extent(e -> e.width(viewport.width).height(viewport.height)))
                        .layerCount(1)
                        .pColorAttachments(colorAttach)
                        .pDepthAttachment(depthAttach)
                        .pStencilAttachment(stencilAttach);
                vkCmdBeginRenderingKHR(cmd, info);
                rendering = true;
            }

            this.depthSetup.bind(cmd);
            VkCmd.setViewportScissor(cmd, viewport.width, viewport.height);
            try (var b = this.depthSetup.binder()) {
                b.sampler(0, VkFrameHost.vkView(rt.mcDepth), this.depthSampler, VK_IMAGE_LAYOUT_GENERAL)
                        .push(cmd);
            }
            try (MemoryStack stack = stackPush()) {
                var pc = stack.malloc(8);
                pc.putFloat(0, ((float) viewport.width) / rt.mcWidth);
                pc.putFloat(4, ((float) viewport.height) / rt.mcHeight);
                this.depthSetup.pushConstants(cmd, pc);
            }
            vkCmdDraw(cmd, 4, 1, 0, 0);
        } finally {
            if (rendering) {
                vkCmdEndRenderingKHR(cmd);
            }
            VkFrameHost.barrierMcImageForAttachment(cmd, rt.mcDepth, true);
        }
    }

    public void offscreenToSampled(VkViewport viewport) {
        viewport.depthStencil.transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    public void offscreenToAttachment(VkViewport viewport) {
        viewport.depthStencil.transition(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    }

    public void composite(VkViewportRT rt) {
        var viewport = rt.viewport;
        boolean fogCoversAllRendering = viewport.fogParameters != null
                && viewport.fogParameters.environmentalEnd() < VoxyRenderSystem.getVanillaRenderDistance();
        if (fogCoversAllRendering) return;

        int mcColorFormat = VkFrameHost.vkFormat(rt.mcColour);
        int mcDepthFormat = VkFrameHost.vkFormat(rt.mcDepth);
        this.ensureCompositePipeline(mcColorFormat, mcDepthFormat);
        var cmd = this.ctx.cmd();

        {
            long ptr = this.uploadStream.upload(this.compositeParams, 0, 256);
            this.scratchA.set(viewport.MVP).invert().getToAddress(ptr); ptr += 64;
            this.scratchA.set(viewport.vanillaProjection).mul(viewport.modelView).getToAddress(ptr); ptr += 64;
            float e0 = 0, e1 = 0, e2 = 0, f0 = 0, f1 = 0, f2 = 0, f3 = 0;
            if (this.useEnvFog && viewport.fogParameters != null) {
                float start = viewport.fogParameters.environmentalStart();
                float endF = viewport.fogParameters.environmentalEnd();
                if (Math.abs(endF - start) > 1) {
                    float invEndFogDelta = 1f / (endF - start);
                    float endDistance = Math.max(VoxyRenderSystem.getVanillaRenderDistance(), 20 * 16) * (float) Math.sqrt(3);
                    float startDelta = -start * invEndFogDelta;
                    e0 = invEndFogDelta;
                    e1 = startDelta;
                    e2 = Math.clamp(endDistance * invEndFogDelta + startDelta, 0, 1);
                    f0 = viewport.fogParameters.red();
                    f1 = viewport.fogParameters.green();
                    f2 = viewport.fogParameters.blue();
                    f3 = viewport.fogParameters.alpha();
                }
            }
            MemoryUtil.memPutFloat(ptr, e0);
            MemoryUtil.memPutFloat(ptr + 4, e1);
            MemoryUtil.memPutFloat(ptr + 8, e2);
            MemoryUtil.memPutFloat(ptr + 12, 0f);
            ptr += 16;
            MemoryUtil.memPutFloat(ptr, f0);
            MemoryUtil.memPutFloat(ptr + 4, f1);
            MemoryUtil.memPutFloat(ptr + 8, f2);
            MemoryUtil.memPutFloat(ptr + 12, f3);
            this.uploadStream.commit();
        }

        //Translucency is conditional and fog-covered frames can skip this
        //composite entirely, so colourSSAO can have more than one valid prior
        //producer/layout. Keep the destination precise but use a conservative
        //source scope rather than assuming a colour-attachment write happened.
        viewport.colourSSAO.transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        //Blaze3D's main targets are GENERAL-layout images. Establish attachment
        //dependencies but do not change the layout it expects.
        VkFrameHost.barrierMcImageForAttachment(cmd, rt.mcColour, false);
        VkFrameHost.barrierMcImageForAttachment(cmd, rt.mcDepth, true);

        boolean rendering = false;
        try {
            try (MemoryStack stack = stackPush()) {
                var colorAttach = VkRenderingAttachmentInfoKHR.calloc(1, stack).sType$Default()
                        .imageView(VkFrameHost.vkView(rt.mcColour))
                        .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                var depthAttach = VkRenderingAttachmentInfoKHR.calloc(stack).sType$Default()
                        .imageView(VkFrameHost.vkView(rt.mcDepth))
                        .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                var info = VkRenderingInfoKHR.calloc(stack).sType$Default()
                        .renderArea(VkRect2D.calloc(stack).extent(e -> e.width(rt.mcWidth).height(rt.mcHeight)))
                        .layerCount(1)
                        .pColorAttachments(colorAttach)
                        .pDepthAttachment(depthAttach);
                vkCmdBeginRenderingKHR(cmd, info);
                rendering = true;
            }

            this.composite.bind(cmd);
            VkCmd.setViewportScissor(cmd, rt.mcWidth, rt.mcHeight);
            try (var b = this.composite.binder()) {
                b.sampler(0, viewport.depthSampleView, this.depthSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .ubo(1, this.compositeParams)
                        .sampler(3, viewport.colourSSAO.view, this.colourSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .push(cmd);
            }
            vkCmdDraw(cmd, 4, 1, 0, 0);
        } finally {
            if (rendering) {
                vkCmdEndRenderingKHR(cmd);
                //Voxy is now handing Minecraft-owned targets back to Blaze3D.
                //The next user may be another attachment pass, a sampler, a
                //transfer, or post-processing. Publish our writes broadly while
                //preserving the GENERAL layout Minecraft tracks internally.
                VkFrameHost.barrierMcImageForExternalUse(cmd, rt.mcColour, false);
                VkFrameHost.barrierMcImageForExternalUse(cmd, rt.mcDepth, true);
            }
        }
    }

    public void free() {
        if (this.depthSetup != null) this.depthSetup.free();
        if (this.composite != null) this.composite.free();
        this.compositeParams.free();
    }
}
