package me.cortex.voxy.client.core.vk.render;

import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.rendering.IRenderList;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
import me.cortex.voxy.client.core.vk.VkImage2D;
import me.cortex.voxy.common.Logger;

import static org.lwjgl.vulkan.VK10.*;

//Pure-VK viewport: the MDICViewport buffer set as VkBuffers, plus the
//offscreen render targets (colour + D32S8 depth-stencil), the depth-bound
//image (vanilla-coverage optimisation), and the HiZ pyramid.
public class VkViewport extends Viewport<VkViewport> {
    public static final int OPAQUE_DRAW_COUNT = 400_000;
    public static final int TRANSLUCENT_DRAW_COUNT = 100_000;
    public static final int TEMPORAL_DRAW_COUNT = 100_000;
    private static final int RESIZE_RETRY_FRAMES = 30;

    private final VkFrameCtx ctx;

    public final VkBuffer drawCountCallBuffer;
    public final VkBuffer drawCallBuffer;
    public final VkBuffer positionScratchBuffer;
    public final VkBuffer indirectLookupBuffer;
    public final VkBuffer visibilityBuffer;

    //Offscreen targets, lazily (re)created on resize
    public VkImage2D colour;
    public VkImage2D depthStencil;
    public long depthSampleView;//DEPTH-aspect view of depthStencil for sampling
    public VkImage2D depthBound;
    public long depthBoundSampleView;
    //SSAO output colour: the compute pass reads `colour` and writes the AO-modulated
    //result (alpha sanitized to 1/0) here; translucents then draw onto it and the
    //compositor samples it — mirroring the GL colourTex/colourSSAOTex pair.
    public VkImage2D colourSSAO;
    public final VkHiZ hiZ;

    //If a larger resize is rejected by VMA's heap budget, keep the previous
    //targets alive and retry later instead of allocating/failing every frame.
    private int failedResizeWidth = -1;
    private int failedResizeHeight = -1;
    private int resizeRetryAfterFrame;

    public VkViewport(VkFrameCtx ctx, RenderProperties properties, int maxSectionCount) {
        super(properties);
        this.ctx = ctx;

        VkBuffer createdDrawCount = null;
        VkBuffer createdDrawCalls = null;
        VkBuffer createdPositionScratch = null;
        VkBuffer createdIndirectLookup = null;
        VkBuffer createdVisibility = null;
        VkHiZ createdHiZ = null;
        try {
            createdDrawCount = new VkBuffer(ctx, 1024).zero();
            createdDrawCalls = new VkBuffer(ctx, 5L * 4 * (OPAQUE_DRAW_COUNT + TRANSLUCENT_DRAW_COUNT + TEMPORAL_DRAW_COUNT)).zero();
            createdPositionScratch = new VkBuffer(ctx, 8L * 400000).zero();
            createdIndirectLookup = new VkBuffer(ctx, HierarchicalOcclusionTraverser.MAX_QUEUE_SIZE * 4L + 4).zero();
            createdVisibility = new VkBuffer(ctx, maxSectionCount * 4L).zero();
            createdHiZ = new VkHiZ(ctx, properties);
            ctx.flushImmediate();
        } catch (RuntimeException | Error failure) {
            if (createdHiZ != null) createdHiZ.free();
            if (createdVisibility != null) createdVisibility.free();
            if (createdIndirectLookup != null) createdIndirectLookup.free();
            if (createdPositionScratch != null) createdPositionScratch.free();
            if (createdDrawCalls != null) createdDrawCalls.free();
            if (createdDrawCount != null) createdDrawCount.free();
            ctx.waitIdleRetireAll();
            throw failure;
        }

        this.drawCountCallBuffer = createdDrawCount;
        this.drawCallBuffer = createdDrawCalls;
        this.positionScratchBuffer = createdPositionScratch;
        this.indirectLookupBuffer = createdIndirectLookup;
        this.visibilityBuffer = createdVisibility;
        this.hiZ = createdHiZ;
    }

    @Override
    protected boolean useGlViewportHelpers() {
        return false;
    }

    private static void freeUnsubmitted(VkImage2D image) {
        if (image != null) image.freeUnsubmittedNow();
    }

    /** (Re)creates the offscreen targets on size change; true if recreated. */
    public boolean ensureTargets() {
        if (this.width <= 0 || this.height <= 0) return false;
        if (this.colour != null && this.colour.width == this.width && this.colour.height == this.height) return false;

        final int requestedWidth = this.width;
        final int requestedHeight = this.height;
        if (this.colour != null
                && requestedWidth == this.failedResizeWidth
                && requestedHeight == this.failedResizeHeight
                && this.frameId < this.resizeRetryAfterFrame) {
            //Render at the previous offscreen resolution for a short period.
            //The compositor already scales between Voxy and Minecraft targets.
            this.width = this.colour.width;
            this.height = this.colour.height;
            return false;
        }

        //Create the complete replacement set before touching the currently-live
        //targets. If the shared VMA budget rejects the resize, all replacement
        //resources are still unsubmitted and can be destroyed immediately while
        //the prior target set remains valid.
        VkImage2D newColour = null;
        VkImage2D newColourSSAO = null;
        VkImage2D newDepthStencil = null;
        VkImage2D newDepthBound = null;
        long newDepthSampleView = VK_NULL_HANDLE;
        try {
            newColour = new VkImage2D(this.ctx, requestedWidth, requestedHeight, 1,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, false);
            newColourSSAO = new VkImage2D(this.ctx, requestedWidth, requestedHeight, 1,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, false);
            newDepthStencil = new VkImage2D(this.ctx, requestedWidth, requestedHeight, 1,
                    VK_FORMAT_D32_SFLOAT_S8_UINT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false,
                    //VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT + view-format list lets the
                    //depth-only aspect view alias the image directly. Without this,
                    //MoltenVK may have to synthesise a separate staging texture for
                    //depth-only sampling of a packed D32S8 image. The list must
                    //contain the original format plus the depth-only format.
                    new int[]{VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT});
            newDepthSampleView = newDepthStencil.createAspectView(VK_IMAGE_ASPECT_DEPTH_BIT);
            newDepthBound = new VkImage2D(this.ctx, requestedWidth, requestedHeight, 1,
                    VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, false);
        } catch (RuntimeException failure) {
            freeUnsubmitted(newDepthBound);
            freeUnsubmitted(newDepthStencil);
            freeUnsubmitted(newColourSSAO);
            freeUnsubmitted(newColour);

            if (this.colour != null) {
                this.failedResizeWidth = requestedWidth;
                this.failedResizeHeight = requestedHeight;
                this.resizeRetryAfterFrame = this.frameId + RESIZE_RETRY_FRAMES;
                this.width = this.colour.width;
                this.height = this.colour.height;
                try {
                    var budget = this.ctx.vk().deviceLocalBudget();
                    Logger.warn("Voxy VK: keeping previous " + this.width + "x" + this.height
                            + " targets after " + requestedWidth + "x" + requestedHeight
                            + " resize exceeded/failed VMA allocation; free="
                            + (budget.availableBytes() >> 20) + " MiB, retrying later ("
                            + failure.getMessage() + ")");
                } catch (RuntimeException ignored) {
                    Logger.warn("Voxy VK: keeping previous framebuffer size after resize allocation failed: "
                            + failure.getMessage());
                }
                return false;
            }
            throw failure;
        } catch (Error failure) {
            freeUnsubmitted(newDepthBound);
            freeUnsubmitted(newDepthStencil);
            freeUnsubmitted(newColourSSAO);
            freeUnsubmitted(newColour);
            throw failure;
        }

        VkImage2D oldColour = this.colour;
        VkImage2D oldColourSSAO = this.colourSSAO;
        VkImage2D oldDepthStencil = this.depthStencil;
        VkImage2D oldDepthBound = this.depthBound;

        this.colour = newColour;
        this.colourSSAO = newColourSSAO;
        this.depthStencil = newDepthStencil;
        this.depthSampleView = newDepthSampleView;
        this.depthBound = newDepthBound;
        this.depthBoundSampleView = newDepthBound.view;
        this.width = requestedWidth;
        this.height = requestedHeight;
        this.failedResizeWidth = -1;
        this.failedResizeHeight = -1;
        this.resizeRetryAfterFrame = 0;

        if (oldColour != null) oldColour.free();
        if (oldColourSSAO != null) oldColourSSAO.free();
        if (oldDepthStencil != null) oldDepthStencil.free();
        if (oldDepthBound != null) oldDepthBound.free();
        return true;
    }

    @Override
    protected void delete0() {
        super.delete0();
        if (this.colour != null) this.colour.free();
        if (this.colourSSAO != null) this.colourSSAO.free();
        if (this.depthStencil != null) this.depthStencil.free();
        if (this.depthBound != null) this.depthBound.free();
        this.hiZ.free();
        this.visibilityBuffer.free();
        this.indirectLookupBuffer.free();
        this.drawCountCallBuffer.free();
        this.drawCallBuffer.free();
        this.positionScratchBuffer.free();
    }

    @Override
    public IRenderList getRenderList() {
        return this.indirectLookupBuffer;
    }
}
