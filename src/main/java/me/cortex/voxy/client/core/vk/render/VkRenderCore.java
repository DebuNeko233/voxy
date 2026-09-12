package me.cortex.voxy.client.core.vk.render;

import com.mojang.blaze3d.pipeline.RenderTarget;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.rendering.RenderDistanceTracker;
import me.cortex.voxy.client.core.rendering.ViewportSelector;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.model.bakery.IAtlasTextureReader;
import me.cortex.voxy.client.core.rendering.bounding.StreamedBoundStore;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.util.AbstractDownloadStream;
import me.cortex.voxy.client.core.rendering.util.AbstractUploadStream;
import me.cortex.voxy.client.core.vk.MinecraftVkHost;
import me.cortex.voxy.client.core.vk.MinecraftVkHostAdapter;
import me.cortex.voxy.client.core.vk.VkAtlasTextureReader;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkDownloadStream;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
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

//The pure-Vulkan render core: constructed instead of the GL pipeline when MC
// itself runs on Vulkan. Owns every GPU-facing subsystem (streams, geometry,
// models, traversal, terrain renderer, compositor) on MC's adopted device, and
// shares the CPU-side services (node manager, mesh generation, model bakery,
// render-distance tracking) with the GL path.
//
//Everything records into MC's frame command buffer from the render hook
// (MixinSodiumOpaqueVkFrame, TAIL of Sodium's opaque terrain draw), between
// MC's opaque terrain pass and the rest of its frame. No OpenGL is touched.
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
    private boolean shutDown = false;
    private final RenderDistanceTracker renderDistanceTracker;
    private final ViewportSelector<VkViewport> viewportSelector;

    public VkRenderCore(WorldEngine world, ServiceManager sm) {
        world.acquireRef();
        Logger.info("Creating Voxy pure-Vulkan render core");
        try {
            this.worldIn = world;
            var host = MinecraftVkHost.get();
            if (host == null) throw new IllegalStateException("No Minecraft Vulkan host adapter registered");
            var vctx = VulkanBackend.context();//adopts MC's device
            this.frameCtx = new VkFrameCtx(vctx);

            //Install the VK streams BEFORE any shared class touches the singletons
            this.uploadStream = new VkUploadStream(this.frameCtx, 1 << 26);//64 mb, same as GL
            this.downloadStream = new VkDownloadStream(this.frameCtx, 1 << 25);//32 mb, same as GL
            AbstractUploadStream.setInstance(this.uploadStream);
            AbstractDownloadStream.setInstance(this.downloadStream);

            this.properties = RenderProperties.getRenderProperties();

            //Install the VK atlas readback BEFORE the model bakery reads the block atlas
            // (its constructor does a synchronous GPU->CPU copy)
            IAtlasTextureReader.setInstance(
                    new VkAtlasTextureReader(this.frameCtx));

            this.modelStore = new VkModelStore(this.frameCtx, this.uploadStream);
            this.modelService = new ModelBakerySubsystem(world.getMapper(), this.modelStore);
            this.renderGen = new RenderGenerationService(world, this.modelService, sm, false);

            long geometryCapacity = vctx.recommendedGeometryCapacityBytes();
            Logger.info("Voxy VK geometry target: " + (geometryCapacity >> 20) + " MiB from "
                    + (vctx.deviceLocalHeapBytes >> 20) + " MiB device-local heap"
                    + (vctx.integratedGpu ? " (integrated GPU policy)" : " (discrete GPU policy)"));
            this.geometryData = new VkSectionGeometryData(this.frameCtx, 1 << 20, geometryCapacity);
            this.nodeManager = new AsyncNodeManager(1 << 21, this.geometryData, this.renderGen,
                    new VkNodeGpuOps(this.frameCtx, this.uploadStream));
            this.nodeCleaner = new VkNodeCleaner(this.frameCtx, this.uploadStream, this.downloadStream, this.nodeManager);
            this.traversal = new VkTraversal(this.frameCtx, this.uploadStream, this.downloadStream,
                    this.properties, this.nodeManager, this.nodeCleaner, this.renderGen);
            this.terrainRenderer = new VkTerrainRenderer(this.frameCtx, this.uploadStream, this.downloadStream,
                    this.properties, this.geometryData, this.modelStore);
            this.compositor = new VkCompositor(this.frameCtx, this.uploadStream, this.properties,
                    VoxyConfig.CONFIG.getFogMode().hasFog);
            this.ssao = new VkSSAO(this.frameCtx, this.uploadStream, this.properties, VoxyConfig.CONFIG.getSSAOMode());
            this.visibleSectionStream = new StreamedBoundStore(
                    size -> new VkBuffer(this.frameCtx, size));
            this.boundRenderer = new VkBoundRenderer(this.frameCtx, this.uploadStream, this.properties);

            world.setDirtyCallback(this.nodeManager::worldEvent);
            Arrays.stream(world.getMapper().getBiomeEntries()).forEach(this.modelService::addBiome);
            world.getMapper().setBiomeCallback(this.modelService::addBiome);
            this.nodeManager.start();

            this.viewportSelector = new ViewportSelector<>(() ->
                    new VkViewport(this.frameCtx, this.properties, this.geometryData.getMaxSectionCount()));

            int minSec = Minecraft.getInstance().level.getMinSectionY() >> 5;
            int maxSec = (Minecraft.getInstance().level.getMaxSectionY() - 1) >> 5;
            this.renderDistanceTracker = new RenderDistanceTracker(40, minSec, maxSec,
                    this.nodeManager::addTopLevel, this.nodeManager::removeTopLevel);
            this.setRenderDistance(VoxyConfig.CONFIG.sectionRenderDistance);

            this.frameCtx.flushImmediate();
            Logger.info("Voxy pure-Vulkan render core created with " + this.geometryData.getMaxCapacity() + " geometry capacity");
        } catch (RuntimeException e) {
            world.releaseRef();
            throw e;
        }
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
        if (this.shutDown) {
            return;
        }
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
