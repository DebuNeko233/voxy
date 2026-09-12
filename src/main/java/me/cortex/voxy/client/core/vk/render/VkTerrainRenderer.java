package me.cortex.voxy.client.core.vk.render;

import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkCmd;
import me.cortex.voxy.client.core.vk.VkDownloadStream;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
import me.cortex.voxy.client.core.vk.VkImage2D;
import me.cortex.voxy.client.core.vk.VkShaderPipeline;
import me.cortex.voxy.client.core.vk.VkShaderSource;
import me.cortex.voxy.client.core.vk.VkUploadStream;
import me.cortex.voxy.common.Logger;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkRect2D;
import org.lwjgl.vulkan.VkRenderingAttachmentInfoKHR;
import org.lwjgl.vulkan.VkRenderingInfoKHR;

import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRDynamicRendering.vkCmdBeginRenderingKHR;
import static org.lwjgl.vulkan.KHRDynamicRendering.vkCmdEndRenderingKHR;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK12.vkCmdDrawIndexedIndirectCount;

//Pure-VK mirror of MDICSectionRenderer: prep, raster-cull, command generation,
// translucency sorting and the three terrain draw phases.
public class VkTerrainRenderer {
    private static final int TRANSLUCENT_OFFSET = VkViewport.OPAQUE_DRAW_COUNT;
    private static final int TEMPORAL_OFFSET = TRANSLUCENT_OFFSET + VkViewport.TRANSLUCENT_DRAW_COUNT;

    //cmdgen.comp can emit one double-sided + six directional opaque commands
    //per visible section. Temporal generation mirrors those opaque commands, while
    //translucency contributes at most one command per section. On Minecraft 26.2
    //drawIndirectCount is not enabled on the adopted logical device, so the
    //fixed-count fallback must use these deterministic upper bounds rather than a
    //stale CPU readback. Every unused command is pre-zeroed, making the extra
    //indirect records legal zero-draws instead of visible overdraw.
    private static final int MAX_OPAQUE_COMMANDS_PER_SECTION = 7;
    private static final int MAX_TEMPORAL_COMMANDS_PER_SECTION = 7;
    private static final int MAX_TRANSLUCENT_COMMANDS_PER_SECTION = 1;

    private final VkFrameCtx ctx;
    private final VkUploadStream uploadStream;
    private final RenderProperties properties;
    private final VkSectionGeometryData geometry;
    private final VkModelStore modelStore;

    private final VkBuffer uniform;
    private final VkBuffer distanceCountBuffer;
    private final VkBuffer indexBuffer;
    private final long depthBoundSampler;
    private final long lightmapSampler;

    private final VkShaderPipeline prep;
    private final VkShaderPipeline cmdGen;
    private final VkShaderPipeline prefixSum;
    private final VkShaderPipeline translucentGen;
    private final VkShaderPipeline cullRaster;
    private VkShaderPipeline terrainOpaque;
    private VkShaderPipeline terrainTranslucent;
    private int pipelineColorFormat = -1, pipelineDepthFormat = -1;

    public VkTerrainRenderer(VkFrameCtx ctx, VkUploadStream uploadStream, VkDownloadStream downloadStream,
                             RenderProperties properties, VkSectionGeometryData geometry, VkModelStore modelStore) {
        this.ctx = ctx;
        this.uploadStream = uploadStream;
        this.properties = properties;
        this.geometry = geometry;
        this.modelStore = modelStore;

        VkBuffer uniformBuffer = null;
        VkBuffer distanceBuffer = null;
        VkBuffer indices = null;
        VkShaderPipeline prepPipeline = null;
        VkShaderPipeline cmdPipeline = null;
        VkShaderPipeline prefixPipeline = null;
        VkShaderPipeline translucentPipeline = null;
        VkShaderPipeline cullPipeline = null;
        long depthSampler = 0;
        long lmSampler = 0;
        try {
            uniformBuffer = new VkBuffer(ctx, 1024).zero();
            distanceBuffer = new VkBuffer(ctx, 1024L * 4 + VkViewport.TRANSLUCENT_DRAW_COUNT * 4L).zero();

            indices = new VkBuffer(ctx, SharedIndexBuffer.CUBE_INDEX_OFFSET + 6 * 2 * 3 * 2L);
            var quads = SharedIndexBuffer.generateQuadIndicesShort(16380);
            try {
                long ptr = uploadStream.upload(indices, 0, indices.size());
                quads.cpyTo(ptr);
                VkCmd.writeCubeIndicesU16(ptr + SharedIndexBuffer.CUBE_INDEX_OFFSET);
                uploadStream.commit();
                ctx.flushImmediate();
            } finally {
                quads.free();
            }

            depthSampler = VkImage2D.createSampler(ctx.vk(), false, false);
            lmSampler = VkImage2D.createSampler(ctx.vk(), false, true);

            prepPipeline = new VkShaderPipeline(ctx, "prep.comp",
                    VkShaderSource.load("voxy:lod/gl46/prep.comp", VkShaderSource.defs().build()),
                    0, List.of(VkShaderPipeline.ubo(0), VkShaderPipeline.ssbo(1), VkShaderPipeline.ssbo(2)));

            cmdPipeline = new VkShaderPipeline(ctx, "cmdgen.comp",
                    VkShaderSource.load("voxy:lod/gl46/cmdgen.comp", VkShaderSource.defs()
                            .def("TRANSLUCENT_WRITE_BASE", 1024)
                            .def("TEMPORAL_OFFSET", TEMPORAL_OFFSET)
                            .def("OPAQUE_DRAW_CAP", VkViewport.OPAQUE_DRAW_COUNT)
                            .def("TRANSLUCENT_DRAW_CAP", VkViewport.TRANSLUCENT_DRAW_COUNT)
                            .def("TEMPORAL_DRAW_CAP", VkViewport.TEMPORAL_DRAW_COUNT)
                            .def("TRANSLUCENT_DISTANCE_BUFFER_BINDING", 7)
                            .build()),
                    0, List.of(VkShaderPipeline.ubo(0), VkShaderPipeline.ssbo(1), VkShaderPipeline.ssbo(2),
                            VkShaderPipeline.ssbo(3), VkShaderPipeline.ssbo(4), VkShaderPipeline.ssbo(5),
                            VkShaderPipeline.ssbo(6), VkShaderPipeline.ssbo(7)));

            boolean useSubgroup = ctx.vk().subgroupArithmetic;
            prefixPipeline = new VkShaderPipeline(ctx, "prefixsum.comp",
                    VkShaderSource.load(useSubgroup ? "voxy:util/prefixsum/inital3_vk.comp" : "voxy:util/prefixsum/simple.comp",
                            VkShaderSource.defs().def("IO_BUFFER", 0).build()),
                    0, List.of(VkShaderPipeline.ssbo(0)));

            translucentPipeline = new VkShaderPipeline(ctx, "buildtranslucents.comp",
                    VkShaderSource.load("voxy:lod/gl46/buildtranslucents.comp", VkShaderSource.defs()
                            .def("TRANSLUCENT_WRITE_BASE", 1024)
                            .def("TRANSLUCENT_DISTANCE_BUFFER_BINDING", 5)
                            .def("TRANSLUCENT_OFFSET", TRANSLUCENT_OFFSET)
                            .build()),
                    0, List.of(VkShaderPipeline.ubo(0), VkShaderPipeline.ssbo(1), VkShaderPipeline.ssbo(2),
                            VkShaderPipeline.ssbo(3), VkShaderPipeline.ssbo(4), VkShaderPipeline.ssbo(5)));

            var cullDesc = new VkShaderPipeline.GfxDesc();
            cullDesc.name = "cullraster";
            cullDesc.vertGlsl = VkShaderSource.load("voxy:lod/gl46/cull/raster.vert", VkShaderSource.defs().props(properties).build());
            cullDesc.fragGlsl = VkShaderSource.load("voxy:lod/gl46/cull/raster.frag", VkShaderSource.defs().props(properties).build());
            cullDesc.colorFormat = VK_FORMAT_UNDEFINED;
            cullDesc.depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
            cullDesc.stencilFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
            cullDesc.depthTest = true;
            cullDesc.depthWrite = false;
            cullDesc.colorWrite = false;
            cullDesc.depthCompare = VkCmd.closerEqual(this.properties);
            cullDesc.bindings = List.of(VkShaderPipeline.ubo(0), VkShaderPipeline.ssbo(1),
                    VkShaderPipeline.ssbo(2), VkShaderPipeline.ssbo(3));
            cullPipeline = new VkShaderPipeline(ctx, cullDesc);
        } catch (RuntimeException | Error failure) {
            if (cullPipeline != null) cullPipeline.free();
            if (translucentPipeline != null) translucentPipeline.free();
            if (prefixPipeline != null) prefixPipeline.free();
            if (cmdPipeline != null) cmdPipeline.free();
            if (prepPipeline != null) prepPipeline.free();
            if (indices != null) indices.free();
            if (distanceBuffer != null) distanceBuffer.free();
            if (uniformBuffer != null) uniformBuffer.free();
            ctx.waitIdleRetireAll();
            throw failure;
        }

        this.uniform = uniformBuffer;
        this.distanceCountBuffer = distanceBuffer;
        this.indexBuffer = indices;
        this.depthBoundSampler = depthSampler;
        this.lightmapSampler = lmSampler;
        this.prep = prepPipeline;
        this.cmdGen = cmdPipeline;
        this.prefixSum = prefixPipeline;
        this.translucentGen = translucentPipeline;
        this.cullRaster = cullPipeline;
    }

    private void ensureTerrainPipelines(VkViewport viewport) {
        int cf = viewport.colour.format;
        int df = viewport.depthStencil.format;
        if (this.terrainOpaque != null && cf == this.pipelineColorFormat && df == this.pipelineDepthFormat) return;

        var cardinalLight = net.minecraft.client.Minecraft.getInstance().level.cardinalLighting();
        String vert = VkShaderSource.load("voxy:lod/gl46/quads3.vert", VkShaderSource.defs().props(this.properties)
                .def("NO_SHADE_FACE_TINT", cardinalLight.up())
                .def("UP_FACE_TINT", cardinalLight.up())
                .def("DOWN_FACE_TINT", cardinalLight.down())
                .def("Z_AXIS_FACE_TINT", cardinalLight.north())
                .def("X_AXIS_FACE_TINT", cardinalLight.east())
                .build());

        var bindings = List.of(VkShaderPipeline.ubo(0), VkShaderPipeline.ssbo(1), VkShaderPipeline.ssbo(3),
                VkShaderPipeline.ssbo(4), VkShaderPipeline.ssbo(5),
                VkShaderPipeline.sampler(8), VkShaderPipeline.sampler(9), VkShaderPipeline.sampler(10));

        VkShaderPipeline newOpaque = null;
        VkShaderPipeline newTranslucent = null;
        try {
            var opaque = new VkShaderPipeline.GfxDesc();
            opaque.name = "terrain-opaque";
            opaque.vertGlsl = vert;
            opaque.fragGlsl = VkShaderSource.load("voxy:lod/gl46/quads.frag", VkShaderSource.defs().props(this.properties)
                    .defIf("VOXY_VULKAN_SAMPLE_MASK_DISCARD", this.ctx.vk().needsSampleMaskDiscard)
                    .build());
            opaque.colorFormat = cf;
            opaque.depthFormat = df;
            opaque.stencilFormat = df;
            opaque.depthTest = true;
            opaque.depthWrite = true;
            opaque.depthCompare = VkCmd.closerEqual(this.properties);
            opaque.blend = false;
            opaque.stencilTestEqual1 = true;
            opaque.bindings = bindings;
            newOpaque = new VkShaderPipeline(this.ctx, opaque);

            var translucent = new VkShaderPipeline.GfxDesc();
            translucent.name = "terrain-translucent";
            translucent.vertGlsl = vert;
            translucent.fragGlsl = VkShaderSource.load("voxy:lod/gl46/quads.frag", VkShaderSource.defs().props(this.properties)
                    .defIf("VOXY_VULKAN_SAMPLE_MASK_DISCARD", this.ctx.vk().needsSampleMaskDiscard)
                    .def("TRANSLUCENT").build());
            translucent.colorFormat = cf;
            translucent.depthFormat = df;
            translucent.stencilFormat = df;
            translucent.depthTest = true;
            translucent.depthWrite = true;
            translucent.depthCompare = VkCmd.closerEqual(this.properties);
            translucent.blend = true;
            translucent.stencilTestEqual1 = true;
            translucent.bindings = bindings;
            newTranslucent = new VkShaderPipeline(this.ctx, translucent);
        } catch (RuntimeException | Error failure) {
            if (newTranslucent != null) newTranslucent.free();
            if (newOpaque != null) newOpaque.free();
            throw failure;
        }

        VkShaderPipeline oldOpaque = this.terrainOpaque;
        VkShaderPipeline oldTranslucent = this.terrainTranslucent;
        this.terrainOpaque = newOpaque;
        this.terrainTranslucent = newTranslucent;
        this.pipelineColorFormat = cf;
        this.pipelineDepthFormat = df;
        if (oldOpaque != null) oldOpaque.free();
        if (oldTranslucent != null) oldTranslucent.free();
    }

    private final Matrix4f uniformScratch = new Matrix4f();

    public void uploadUniform(VkViewport viewport) {
        long ptr = this.uploadStream.upload(this.uniform, 0, 1024);
        var mat = this.uniformScratch.set(viewport.MVP);
        mat.translate(-viewport.innerTranslation.x, -viewport.innerTranslation.y, -viewport.innerTranslation.z);
        mat.getToAddress(ptr); ptr += 4 * 4 * 4;
        viewport.section.getToAddress(ptr); ptr += 4 * 3;
        if (viewport.frameId < 0) {
            Logger.error("Frame ID negative, this will cause things to break, wrapping around");
            viewport.frameId &= 0x7fffffff;
        }
        MemoryUtil.memPutInt(ptr, viewport.frameId & 0x7fffffff); ptr += 4;
        viewport.innerTranslation.getToAddress(ptr);
        this.uploadStream.commit();
    }

    public void buildDrawCalls(VkViewport viewport) {
        if (this.geometry.getSectionCount() == 0) return;
        var cmd = this.ctx.cmd();
        this.uploadUniform(viewport);

        if (!this.ctx.vk().hasDrawIndirectCount) {
            long stride = 5L * 4;
            vkCmdFillBuffer(cmd, viewport.drawCallBuffer.buffer, 0L, VkViewport.OPAQUE_DRAW_COUNT * stride, 0);
            vkCmdFillBuffer(cmd, viewport.drawCallBuffer.buffer, TEMPORAL_OFFSET * stride, VkViewport.TEMPORAL_DRAW_COUNT * stride, 0);
            vkCmdFillBuffer(cmd, viewport.drawCallBuffer.buffer, TRANSLUCENT_OFFSET * stride, VkViewport.TRANSLUCENT_DRAW_COUNT * stride, 0);
            this.ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        }

        this.prep.bind(cmd);
        try (var b = this.prep.binder()) {
            b.ubo(0, this.uniform)
                    .ssbo(1, viewport.drawCountCallBuffer)
                    .ssbo(2, viewport.indirectLookupBuffer)
                    .push(cmd);
        }
        vkCmdDispatch(cmd, 1, 1, 1);
        this.ctx.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                        | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        this.beginRendering(cmd, viewport, 0L);
        try {
            this.cullRaster.bind(cmd);
            VkCmd.setViewportScissor(cmd, viewport.width, viewport.height);
            try (var b = this.cullRaster.binder()) {
                b.ubo(0, this.uniform)
                        .ssbo(1, this.geometry.metadataBuffer())
                        .ssbo(2, viewport.visibilityBuffer)
                        .ssbo(3, viewport.indirectLookupBuffer)
                        .push(cmd);
            }
            vkCmdBindIndexBuffer(cmd, this.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            vkCmdDrawIndexedIndirect(cmd, viewport.drawCountCallBuffer.buffer, 6 * 4, 1, 20);
        } finally {
            vkCmdEndRenderingKHR(cmd);
        }
        this.ctx.barrier(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        vkCmdFillBuffer(cmd, this.distanceCountBuffer.buffer, 0, 1024L * 4, 0);
        this.ctx.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        this.cmdGen.bind(cmd);
        try (var b = this.cmdGen.binder()) {
            b.ubo(0, this.uniform)
                    .ssbo(1, viewport.drawCallBuffer)
                    .ssbo(2, viewport.drawCountCallBuffer)
                    .ssbo(3, this.geometry.metadataBuffer())
                    .ssbo(4, viewport.visibilityBuffer)
                    .ssbo(5, viewport.indirectLookupBuffer)
                    .ssbo(6, viewport.positionScratchBuffer)
                    .ssbo(7, this.distanceCountBuffer)
                    .push(cmd);
        }
        vkCmdDispatchIndirect(cmd, viewport.drawCountCallBuffer.buffer, 0);
        this.ctx.computeToComputeBarrier();

        this.prefixSum.bind(cmd);
        try (var b = this.prefixSum.binder()) {
            b.ssbo(0, this.distanceCountBuffer).push(cmd);
        }
        vkCmdDispatch(cmd, 1, 1, 1);
        this.ctx.computeToComputeBarrier();

        this.translucentGen.bind(cmd);
        try (var b = this.translucentGen.binder()) {
            b.ubo(0, this.uniform)
                    .ssbo(1, viewport.drawCallBuffer)
                    .ssbo(2, viewport.drawCountCallBuffer)
                    .ssbo(3, this.geometry.metadataBuffer())
                    .ssbo(4, viewport.indirectLookupBuffer)
                    .ssbo(5, this.distanceCountBuffer)
                    .push(cmd);
        }
        vkCmdDispatchIndirect(cmd, viewport.drawCountCallBuffer.buffer, 0);
    }

    private static int fixedCountUpperBound(int sectionCount, int commandsPerSection, int capacity) {
        if (sectionCount <= 0) return 0;
        long required = (long) sectionCount * commandsPerSection;
        return (int) Math.min(required, (long) capacity);
    }

    public void renderOpaque(VkViewport viewport, boolean clearTargets) {
        this.ensureTerrainPipelines(viewport);
        int sectionCount = this.geometry.getSectionCount();
        if (sectionCount == 0) return;
        this.uploadUniform(viewport);
        int maxDraw = fixedCountUpperBound(sectionCount,
                MAX_OPAQUE_COMMANDS_PER_SECTION, VkViewport.OPAQUE_DRAW_COUNT);
        this.renderTerrain(viewport, viewport.colour.view, this.terrainOpaque, 0, 4 * 3, maxDraw);
    }

    public void renderTemporal(VkViewport viewport) {
        this.ensureTerrainPipelines(viewport);
        int sectionCount = this.geometry.getSectionCount();
        if (sectionCount == 0) return;
        int maxDraw = fixedCountUpperBound(sectionCount,
                MAX_TEMPORAL_COMMANDS_PER_SECTION, VkViewport.TEMPORAL_DRAW_COUNT);
        this.renderTerrain(viewport, viewport.colour.view, this.terrainOpaque, TEMPORAL_OFFSET * 5L * 4, 4 * 5, maxDraw);
    }

    public void renderTranslucent(VkViewport viewport) {
        this.ensureTerrainPipelines(viewport);
        int sectionCount = this.geometry.getSectionCount();
        if (sectionCount == 0) return;
        int maxDraw = fixedCountUpperBound(sectionCount,
                MAX_TRANSLUCENT_COMMANDS_PER_SECTION, VkViewport.TRANSLUCENT_DRAW_COUNT);
        this.renderTerrain(viewport, viewport.colourSSAO.view, this.terrainTranslucent, TRANSLUCENT_OFFSET * 5L * 4, 4 * 4, maxDraw);
    }

    private void renderTerrain(VkViewport viewport, long colorView, VkShaderPipeline pipeline,
                               long indirectOffset, long drawCountOffset, int maxDrawCount) {
        var cmd = this.ctx.cmd();
        this.ctx.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDEX_READ_BIT);

        //Minecraft owns the lightmap image and keeps VulkanGpuTexture objects in
        //GENERAL layout. Establish the sampling dependency before entering Voxy's
        //dynamic rendering scope, but never transition the image behind Blaze3D.
        var lightmap = VkFrameHost.lightmapTextureView();
        VkFrameHost.barrierMcImageForSampling(cmd, lightmap, false);
        long lightmapView = VkFrameHost.vkView(lightmap);

        this.beginRendering(cmd, viewport, colorView);
        try {
            pipeline.bind(cmd);
            VkCmd.setViewportScissor(cmd, viewport.width, viewport.height);
            try (var b = pipeline.binder()) {
                b.ubo(0, this.uniform)
                        .ssbo(1, this.geometry.geometryBuffer())
                        .ssbo(3, this.modelStore.modelBuffer)
                        .ssbo(4, this.modelStore.modelColourBuffer)
                        .ssbo(5, viewport.positionScratchBuffer)
                        .sampler(8, this.modelStore.atlas.view, this.modelStore.atlasSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .sampler(9, lightmapView, this.lightmapSampler, VK_IMAGE_LAYOUT_GENERAL)
                        .sampler(10, viewport.depthBoundSampleView, this.depthBoundSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .push(cmd);
            }
            vkCmdBindIndexBuffer(cmd, this.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            if (this.ctx.vk().hasDrawIndirectCount) {
                vkCmdDrawIndexedIndirectCount(cmd,
                        viewport.drawCallBuffer.buffer, indirectOffset,
                        viewport.drawCountCallBuffer.buffer, drawCountOffset,
                        maxDrawCount, 5 * 4);
            } else {
                vkCmdDrawIndexedIndirect(cmd, viewport.drawCallBuffer.buffer, indirectOffset, maxDrawCount, 5 * 4);
            }
        } finally {
            vkCmdEndRenderingKHR(cmd);
        }
    }

    private void beginRendering(VkCommandBuffer cmd, VkViewport viewport, long colorView) {
        try (MemoryStack stack = stackPush()) {
            var depthAttach = VkRenderingAttachmentInfoKHR.calloc(stack).sType$Default()
                    .imageView(viewport.depthStencil.view)
                    .imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                    .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
            var stencilAttach = VkRenderingAttachmentInfoKHR.calloc(stack).sType$Default()
                    .imageView(viewport.depthStencil.view)
                    .imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                    .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
            var info = VkRenderingInfoKHR.calloc(stack).sType$Default()
                    .renderArea(VkRect2D.calloc(stack).extent(e -> e.width(viewport.width).height(viewport.height)))
                    .layerCount(1)
                    .pDepthAttachment(depthAttach)
                    .pStencilAttachment(stencilAttach);
            if (colorView != 0L) {
                var colorAttach = VkRenderingAttachmentInfoKHR.calloc(1, stack).sType$Default()
                        .imageView(colorView)
                        .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                info.pColorAttachments(colorAttach);
            }
            vkCmdBeginRenderingKHR(cmd, info);
        }
    }

    public void free() {
        this.prep.free();
        this.cmdGen.free();
        this.prefixSum.free();
        this.translucentGen.free();
        this.cullRaster.free();
        if (this.terrainOpaque != null) this.terrainOpaque.free();
        if (this.terrainTranslucent != null) this.terrainTranslucent.free();
        this.uniform.free();
        this.distanceCountBuffer.free();
        this.indexBuffer.free();
    }
}
