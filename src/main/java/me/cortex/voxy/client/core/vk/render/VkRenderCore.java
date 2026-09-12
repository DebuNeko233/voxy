package me.cortex.voxy.client.core.vk.render;

import com.mojang.blaze3d.pipeline.RenderTarget;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.model.bakery.IAtlasTextureReader;
import me.cortex.voxy.client.core.rendering.RenderDistanceTracker;
import me.cortex.voxy.client.core.rendering.ViewportSelector;
import me.cortex.voxy.client.core.rendering.bounding.StreamedBoundStore;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.util.AbstractDownloadStream;
import me.cortex.voxy.client.core.rendering.util.AbstractUploadStream;
import me.cortex.voxy.client.core.vk.MinecraftVkHost;
import me.cortex.voxy.client.core.vk.MinecraftVkHostAdapter;
import me.cortex.voxy.client.core.vk.VkAtlasTextureReader;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkDownloadStream;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
import me.cortex.voxy.client.core.vk.VkImage2D;
import me.cortex.voxy.client.core.vk.VkUploadStream;
import me.cortex.voxy.client.core.vk.VulkanBackend;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.thread.ServiceManager;
import me.cortex.voxy.common.world.WorldEngine;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.minecraft.client.Minecraft;

import java.util.Arrays;
import java.util.List;

//Pure-Vulkan render core. Every field below is owned by this object; construction
// is transactional so a failed shader/allocation cannot strand workers, global
// backend singletons, or Vulkan allocations.
public class VkRenderCore {
    private final WorldEngine worldIn;
    private final VkFrameCtx frameCtx;
    private final VkUploadStream uploadStream;
    private final VkDownloadStream downloadStream;

    private final RenderProperties properties;
    private final VkModelStore modelStore;
    private final ModelBakerySubsystem modelService;
    private final RenderGenerationService renderGen;
    private final VkSectionGeometryData geometryData;
    private final AsyncNodeManager nodeManager;
    private final VkNodeCleaner nodeCleaner;
    private final VkTraversal traversal;
    private final VkTerrainRenderer terrainRenderer;
    private final VkCompositor compositor;
    private final VkSSAO ssao;
    private final VkBoundRenderer boundRenderer;
    private final StreamedBoundStore visibleSectionStream;
    private final RenderDistanceTracker renderDistanceTracker;
    private final ViewportSelector<VkViewport> viewportSelector;
    private boolean shutDown = false;

    public VkRenderCore(WorldEngine world, ServiceManager sm) {
        world.acquireRef();
        this.worldIn = world;
        Logger.info("Creating Voxy pure-Vulkan render core");

        VkFrameCtx frame = null;
        VkUploadStream upload = null;
        VkDownloadStream download = null;
        RenderProperties props = null;
        VkModelStore models = null;
        ModelBakerySubsystem modelBakery = null;
        RenderGenerationService generation = null;
        VkSectionGeometryData geometry = null;
        AsyncNodeManager nodes = null;
        VkNodeCleaner cleaner = null;
        VkTraversal traverse = null;
        VkTerrainRenderer terrain = null;
        VkCompositor compose = null;
        VkSSAO ao = null;
        StreamedBoundStore visible = null;
        VkBoundRenderer bounds = null;
        ViewportSelector<VkViewport> viewports = null;
        RenderDistanceTracker distanceTracker = null;

        try {
            var host = MinecraftVkHost.get();
            if (host == null) throw new IllegalStateException("No Minecraft Vulkan host adapter registered");
            var vctx = VulkanBackend.context();
            frame = new VkFrameCtx(vctx);

            upload = new VkUploadStream(frame, 1 << 26);
            download = new VkDownloadStream(frame, 1 << 25);
            AbstractUploadStream.setInstance(upload);
            AbstractDownloadStream.setInstance(download);

            props = RenderProperties.getRenderProperties();
            IAtlasTextureReader.setInstance(new VkAtlasTextureReader(frame));

            models = new VkModelStore(frame, upload);
            modelBakery = new ModelBakerySubsystem(world.getMapper(), models);
            generation = new RenderGenerationService(world, modelBakery, sm, false);

            long geometryCapacity = vctx.recommendedGeometryCapacityBytes();
            Logger.info("Voxy VK geometry target: " + (geometryCapacity >> 20) + " MiB from "
                    + (vctx.deviceLocalHeapBytes >> 20) + " MiB device-local heap"
                    + (vctx.integratedGpu ? " (integrated GPU policy)" : " (discrete GPU policy)"));
            geometry = new VkSectionGeometryData(frame, 1 << 20, geometryCapacity);
            nodes = new AsyncNodeManager(1 << 21, geometry, generation, new VkNodeGpuOps(frame, upload));
            cleaner = new VkNodeCleaner(frame, upload, download, nodes);
            traverse = new VkTraversal(frame, upload, download, props, nodes, cleaner, generation);
            terrain = new VkTerrainRenderer(frame, upload, download, props, geometry, models);
            compose = new VkCompositor(frame, upload, props, VoxyConfig.CONFIG.getFogMode().hasFog);
            ao = new VkSSAO(frame, upload, props, VoxyConfig.CONFIG.getSSAOMode());
            visible = new StreamedBoundStore(size -> new VkBuffer(frame, size));
            bounds = new VkBoundRenderer(frame, upload, props);

            world.setDirtyCallback(nodes::worldEvent);
            Arrays.stream(world.getMapper().getBiomeEntries()).forEach(modelBakery::addBiome);
            world.getMapper().setBiomeCallback(modelBakery::addBiome);
            nodes.start();

            final VkFrameCtx selectedFrame = frame;
            final RenderProperties selectedProps = props;
            final VkSectionGeometryData selectedGeometry = geometry;
            viewports = new ViewportSelector<>(() ->
                    new VkViewport(selectedFrame, selectedProps, selectedGeometry.getMaxSectionCount()));

            int minSec = Minecraft.getInstance().level.getMinSectionY() >> 5;
            int maxSec = (Minecraft.getInstance().level.getMaxSectionY() - 1) >> 5;
            final AsyncNodeManager selectedNodes = nodes;
            distanceTracker = new RenderDistanceTracker(40, minSec, maxSec,
                    selectedNodes::addTopLevel, selectedNodes::removeTopLevel);
            distanceTracker.setRenderDistance((int) Math.ceil(VoxyConfig.CONFIG.sectionRenderDistance + 1));

            frame.flushImmediate();
        } catch (RuntimeException | Error failure) {
            //Stop producers and detach callbacks before touching the resources they
            // feed. Run every cleanup step even if an earlier one itself fails.
            try {
                world.setDirtyCallback(null);
                world.getMapper().setBiomeCallback(null);
                world.getMapper().setStateCallback(null);
            } catch (Throwable cleanup) {
                failure.addSuppressed(cleanup);
            }
            if (nodes != null) {
                try { nodes.stop(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (generation != null) {
                try { generation.shutdown(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (frame != null) {
                try { frame.waitIdleRetireAll(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }

            //ModelBakerySubsystem owns VkModelStore once it has been constructed.
            if (modelBakery != null) {
                try { modelBakery.shutdown(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            } else if (models != null) {
                try { models.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (bounds != null) {
                try { bounds.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (visible != null) {
                try { visible.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (traverse != null) {
                try { traverse.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (cleaner != null) {
                try { cleaner.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (terrain != null) {
                try { terrain.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (ao != null) {
                try { ao.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (compose != null) {
                try { compose.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (viewports != null) {
                try { viewports.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (geometry != null) {
                try { geometry.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (download != null) {
                try { download.flushWaitClear(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (upload != null) {
                try { upload.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (download != null) {
                try { download.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }
            if (frame != null) {
                try { frame.free(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
            }

            AbstractUploadStream.clearInstance();
            AbstractDownloadStream.clearInstance();
            IAtlasTextureReader.clearInstance();
            world.releaseRef();
            throw failure;
        }

        this.frameCtx = frame;
        this.uploadStream = upload;
        this.downloadStream = download;
        this.properties = props;
        this.modelStore = models;
        this.modelService = modelBakery;
        this.renderGen = generation;
        this.geometryData = geometry;
        this.nodeManager = nodes;
        this.nodeCleaner = cleaner;
        this.traversal = traverse;
        this.terrainRenderer = terrain;
        this.compositor = compose;
        this.ssao = ao;
        this.visibleSectionStream = visible;
        this.boundRenderer = bounds;
        this.viewportSelector = viewports;
        this.renderDistanceTracker = distanceTracker;

        Logger.info("Voxy pure-Vulkan render core created with " + this.geometryData.getMaxCapacity() + " geometry capacity");
    }

    public void renderFrame(RenderTarget target, MinecraftVkHostAdapter adapter, ChunkRenderMatrices matrices,
                            double camX, double camY, double camZ) {
        var frameCmd = adapter.frameCommandBuffer();
        if (frameCmd == null) {
            Logger.warn("Voxy VK: no frame command buffer at hook point, skipping frame");
            return;
        }
        var crs = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState.cameraRenderState;
        if (crs == null || !crs.initialized) return;

        this.frameCtx.flushImmediate();
        this.frameCtx.beginFrame(frameCmd);
        try {
            if (me.cortex.voxy.commonImpl.VoxyCommon.IS_MINE_IN_ABYSS) {
                int sector = (((int) Math.floor(camX) >> 4) + 512) >> 10;
                camX -= sector << 14;
                camY += (16 + (256 - 32 - sector * 30)) * 16;
            }

            var viewport = this.viewportSelector.getViewport();
            var voxyProjection = VoxyRenderSystem.computeProjectionMat(this.properties, matrices.projection());
            var fog = crs.fogData == null ? null : new FogParameters(
                    crs.fogData.color.x, crs.fogData.color.y, crs.fogData.color.z, crs.fogData.color.w,
                    crs.fogData.environmentalStart, crs.fogData.environmentalEnd,
                    crs.fogData.renderDistanceStart, crs.fogData.renderDistanceEnd);
            viewport.setVanillaProjection(matrices.projection())
                    .setProjection(voxyProjection)
                    .setModelView(matrices.modelView())
                    .setCamera(camX, camY, camZ)
                    .setScreenSize(target.width, target.height)
                    .setFogParameters(fog)
                    .update();
            viewport.frameId++;
            if (viewport.width <= 0 || viewport.height <= 0) return;
            viewport.ensureTargets();

            var rt = new VkCompositor.VkViewportRT(viewport,
                    target.getColorTextureView(), target.getDepthTextureView(), target.width, target.height);

            this.compositor.setupDepthStencil(rt);
            this.boundRenderer.render(viewport, this.visibleSectionStream);
            this.terrainRenderer.renderOpaque(viewport, false);

            this.compositor.offscreenToSampled(viewport);
            viewport.hiZ.buildMipChain(viewport.depthSampleView, viewport.width, viewport.height);
            this.compositor.offscreenToAttachment(viewport);

            this.downloadStream.tick();
            this.nodeManager.tick(this.traversal.getNodeBuffer(), this.nodeCleaner);
            this.nodeCleaner.tick(this.traversal.getNodeBuffer());
            this.traversal.doTraversal(viewport);

            this.terrainRenderer.buildDrawCalls(viewport);
            this.terrainRenderer.renderTemporal(viewport);
            this.ssao.compute(viewport, rt);
            this.terrainRenderer.renderTranslucent(viewport);

            this.compositor.offscreenToSampled(viewport);
            this.compositor.composite(rt);

            this.uploadStream.tick();
            this.renderDistanceTracker.setCenterAndProcess(viewport.cameraX, viewport.cameraZ);
            this.modelService.tick(900_000);
        } finally {
            this.frameCtx.endFrame();
            this.frameCtx.pollRetired();
        }
    }

    public void setRenderDistance(float renderDistance) {
        this.renderDistanceTracker.setRenderDistance((int) Math.ceil(renderDistance + 1));
    }

    public void addDebugInfo(List<String> debug) {
        debug.add("VK host mode: " + VulkanBackend.statusLine());
        debug.add("VkBuf [#/logical MiB/allocated MiB]: [" + VkBuffer.getCount() + "/"
                + (VkBuffer.getTotalSize() >> 20) + "/" + (VkBuffer.getTotalAllocationSize() >> 20) + "]");
        debug.add("VkImage [#/allocated MiB]: [" + VkImage2D.getCount() + "/"
                + (VkImage2D.getTotalAllocationSize() >> 20) + "]");
        debug.add("VkFrame [current/retired/inFlight/pendingDestroy/eventPool]: ["
                + this.frameCtx.currentFrame() + "/" + this.frameCtx.retiredFrame() + "/"
                + this.frameCtx.inFlightFrameCount() + "/" + this.frameCtx.pendingDestroyCount() + "/"
                + this.frameCtx.pooledEventCount() + "]");
        this.modelService.addDebugData(debug);
        this.renderGen.addDebugData(debug);
        this.nodeManager.addDebug(debug);
        this.ssao.addDebugInfo(debug);
    }

    public void shutdown() {
        if (this.shutDown) return;
        this.shutDown = true;
        Logger.info("Shutting down Voxy pure-Vulkan render core");

        try {
            this.worldIn.setDirtyCallback(null);
            this.worldIn.getMapper().setBiomeCallback(null);
            this.worldIn.getMapper().setStateCallback(null);
            this.nodeManager.stop();
            this.renderGen.shutdown();
        } catch (Exception e) {
            Logger.error("Error stopping VK render core CPU services", e);
        }

        boolean deviceAlive = MinecraftVkHost.get() != null;
        if (deviceAlive) {
            try {
                this.frameCtx.waitIdleRetireAll();
                this.modelService.shutdown();
                this.boundRenderer.free();
                this.visibleSectionStream.free();
                this.traversal.free();
                this.nodeCleaner.free();
                this.geometryData.free();
                this.terrainRenderer.free();
                this.ssao.free();
                this.compositor.free();
                this.viewportSelector.free();
                this.downloadStream.flushWaitClear();
                this.uploadStream.free();
                this.downloadStream.free();
                this.frameCtx.free();
            } catch (Exception e) {
                Logger.error("Error shutting down VK render core GPU resources", e);
            }
        } else {
            Logger.warn("Voxy VK: Minecraft's Vulkan device is already gone at shutdown; "
                    + "skipping GPU teardown to avoid destroying objects on a dead device");
        }

        AbstractUploadStream.clearInstance();
        AbstractDownloadStream.clearInstance();
        IAtlasTextureReader.clearInstance();
        this.worldIn.releaseRef();
        Logger.info("VK render core shutdown completed");
    }

    public StreamedBoundStore getVisibleSectionStream() {
        return this.visibleSectionStream;
    }

    public WorldEngine getEngine() {
        return this.worldIn;
    }
}
