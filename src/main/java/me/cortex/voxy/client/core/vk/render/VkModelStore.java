package me.cortex.voxy.client.core.vk.render;

import me.cortex.voxy.client.core.model.IModelStore;
import me.cortex.voxy.client.core.model.ModelFactory;
import me.cortex.voxy.client.core.rendering.util.IDeviceBuffer;
import me.cortex.voxy.client.core.vk.VkBuffer;
import me.cortex.voxy.client.core.vk.VkFrameCtx;
import me.cortex.voxy.client.core.vk.VkImage2D;
import me.cortex.voxy.client.core.vk.VkUploadStream;
import me.cortex.voxy.client.core.vk.VkUtil;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.util.MemoryBuffer;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.resources.Identifier;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

import static me.cortex.voxy.client.core.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

//Pure-VK model store: block-model data + biome colour VkBuffers and the baked
//model texture atlas as a mipped RGBA8 VkImage. Texture tiles stream through
//the upload staging buffer with vkCmdCopyBufferToImage per mip; the batch is
//bracketed by TRANSFER_DST <-> SHADER_READ_ONLY transitions.
public class VkModelStore implements IModelStore {
    private static final String ATLAS_MIP_BIAS_PROPERTY = "voxy.vk.modelAtlasMipBias";

    private final VkFrameCtx ctx;
    private final VkUploadStream uploadStream;
    final VkBuffer modelBuffer;
    final VkBuffer modelColourBuffer;
    final VkImage2D atlas;
    public final long atlasSampler;
    private final int atlasMipBias;
    private final int atlasTileSize;
    private final int atlasLevels;
    private boolean inUploadBatch;
    private boolean freed;

    public VkModelStore(VkFrameCtx ctx, VkUploadStream uploadStream) {
        this.ctx = ctx;
        this.uploadStream = uploadStream;

        VkBuffer model = null;
        VkBuffer colour = null;
        VkImage2D atlasImage = null;
        long sampler = VK_NULL_HANDLE;
        int selectedMipBias = 0;
        int selectedTileSize = ModelFactory.MODEL_TEXTURE_SIZE;
        int selectedLevels = ModelFactory.LAYERS;
        try {
            model = new VkBuffer(ctx, IModelStore.MODEL_SIZE * (1L << 16)).zero();
            colour = new VkBuffer(ctx, 4L * (1 << 16)).zero();

            int requestedMipBias = requestedAtlasMipBias();
            boolean explicitMipBias = requestedMipBias >= 0;
            selectedMipBias = explicitMipBias ? requestedMipBias : 0;
            try {
                atlasImage = createAtlas(ctx, selectedMipBias);
            } catch (VkUtil.VulkanCallException failure) {
                //The CPU baker already produced 16->8->4->2 mip data. If the
                //normal 16px atlas cannot fit inside Minecraft VMA's live heap
                //budget, reuse mip1 as a new 8px base instead of crashing. No
                //runtime resampling and no GL/Blaze3D behaviour changes.
                if (!explicitMipBias && selectedMipBias == 0 && failure.isOutOfMemory()
                        && ModelFactory.LAYERS > 1) {
                    selectedMipBias = 1;
                    Logger.warn("Voxy VK: full-resolution model atlas exceeded Vulkan memory budget; "
                            + "retrying with the existing 8px mip tail");
                    atlasImage = createAtlas(ctx, selectedMipBias);
                } else {
                    throw failure;
                }
            }

            selectedTileSize = ModelFactory.MODEL_TEXTURE_SIZE >> selectedMipBias;
            selectedLevels = ModelFactory.LAYERS - selectedMipBias;
            //Start life in shader-read so the first frame can bind it even with no uploads yet.
            atlasImage.transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            ctx.flushImmediate();

            int minecraftMaxMip = ((TextureAtlas) Minecraft.getInstance().getTextureManager()
                    .getTexture(Identifier.fromNamespaceAndPath("minecraft", "textures/atlas/blocks.png")))
                    .maxMipLevel;
            int samplerMaxLod = Math.clamp(minecraftMaxMip - selectedMipBias, 0, selectedLevels - 1);
            try (MemoryStack stack = stackPush()) {
                //Mirror the GL sampler: nearest mag, nearest-within-mip + linear-between-mips min.
                //When the low-memory atlas starts at original mip1, translate the
                //Minecraft max LOD into the new mip coordinate system as well.
                var sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                        .magFilter(VK_FILTER_NEAREST)
                        .minFilter(VK_FILTER_NEAREST)
                        .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                        .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                        .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                        .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                        .minLod(0).maxLod(samplerMaxLod);
                var pSampler = stack.mallocLong(1);
                check(vkCreateSampler(ctx.vk().device, sci, null, pSampler), "vkCreateSampler(modelAtlas)");
                sampler = pSampler.get(0);
            }

            Logger.info("Voxy VK model atlas: tile=" + selectedTileSize + "px, levels=" + selectedLevels
                    + ", maxLod=" + samplerMaxLod
                    + ", allocation=" + (atlasImage.allocationSize() >> 20) + " MiB"
                    + (selectedMipBias == 0 ? " (full resolution)" : " (mip-tail fallback)"));
        } catch (RuntimeException | Error failure) {
            //The atlas is one of Voxy's largest allocations (~510 MiB at the
            //normal 16px model texture size). Constructor failure must not
            //strand any of the already-created resources.
            if (sampler != VK_NULL_HANDLE) {
                vkDestroySampler(ctx.vk().device, sampler, null);
            }
            if (atlasImage != null) atlasImage.free();
            if (colour != null) colour.free();
            if (model != null) model.free();
            ctx.waitIdleRetireAll();
            throw failure;
        }

        this.modelBuffer = model;
        this.modelColourBuffer = colour;
        this.atlas = atlasImage;
        this.atlasSampler = sampler;
        this.atlasMipBias = selectedMipBias;
        this.atlasTileSize = selectedTileSize;
        this.atlasLevels = selectedLevels;
    }

    /**
     * -1 = automatic (full resolution, fallback to mip1 only on Vulkan OOM)
     *  0 = force normal 16px atlas
     *  1 = force existing mip1 as an 8px base for low-memory testing.
     */
    private static int requestedAtlasMipBias() {
        String value = System.getProperty(ATLAS_MIP_BIAS_PROPERTY, "auto").trim();
        if (value.isEmpty() || value.equalsIgnoreCase("auto")) return -1;
        try {
            int bias = Integer.parseInt(value);
            if (bias < 0 || bias > 1 || bias >= ModelFactory.LAYERS) {
                Logger.warn("Ignoring invalid -D" + ATLAS_MIP_BIAS_PROPERTY + "=" + value
                        + "; expected auto, 0, or 1");
                return -1;
            }
            return bias;
        } catch (NumberFormatException ignored) {
            Logger.warn("Ignoring invalid -D" + ATLAS_MIP_BIAS_PROPERTY + "=" + value
                    + "; expected auto, 0, or 1");
            return -1;
        }
    }

    private static VkImage2D createAtlas(VkFrameCtx ctx, int mipBias) {
        int tileSize = ModelFactory.MODEL_TEXTURE_SIZE >> mipBias;
        int levels = ModelFactory.LAYERS - mipBias;
        return new VkImage2D(ctx,
                tileSize * 3 * 256,
                tileSize * 2 * 256,
                levels,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, false);
    }

    private static long mipBytes(int sourceLevel) {
        long ts = ModelFactory.MODEL_TEXTURE_SIZE;
        return (ts * ts * 3L * 2L * 4L) >> (sourceLevel << 1);
    }

    private static long sourceMipOffset(int firstSourceLevel) {
        long offset = 0;
        for (int level = 0; level < firstSourceLevel; level++) {
            offset += mipBytes(level);
        }
        return offset;
    }

    @Override
    public IDeviceBuffer modelBufferHandle() {
        return this.modelBuffer;
    }

    @Override
    public IDeviceBuffer colourBufferHandle() {
        return this.modelColourBuffer;
    }

    @Override
    public void beginTextureUploads() {
        this.atlas.transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        this.inUploadBatch = true;
    }

    @Override
    public void uploadModelTexture(int modelId, MemoryBuffer texture) {
        if (!this.inUploadBatch) throw new IllegalStateException("Texture upload outside batch");
        final int TS = ModelFactory.MODEL_TEXTURE_SIZE;
        int X = (modelId & 0xFF) * this.atlasTileSize * 3;
        int Y = ((modelId >> 8) & 0xFF) * this.atlasTileSize * 2;

        long sourceOffset = sourceMipOffset(this.atlasMipBias);
        long totalBytesLong = 0;
        for (int dstLevel = 0; dstLevel < this.atlasLevels; dstLevel++) {
            totalBytesLong += mipBytes(dstLevel + this.atlasMipBias);
        }
        if (sourceOffset > texture.size - totalBytesLong) {
            throw new IllegalStateException("Model mip-tail upload exceeds baked texture buffer: bias="
                    + this.atlasMipBias + ", sourceOffset=" + sourceOffset + ", bytes=" + totalBytesLong
                    + ", textureBytes=" + texture.size);
        }
        if (totalBytesLong > Integer.MAX_VALUE) throw new IllegalStateException("Model atlas upload is too large");
        int totalBytes = (int) totalBytesLong;
        long stageOff = this.uploadStream.rawUploadAddress(totalBytes);
        MemoryUtil.memCopy(texture.address + sourceOffset, this.uploadStream.getBaseAddress() + stageOff, totalBytes);

        var cmd = this.ctx.cmd();
        try (MemoryStack stack = stackPush()) {
            var regions = VkBufferImageCopy.calloc(this.atlasLevels, stack);
            long srcOff = stageOff;
            for (int dstLevel = 0; dstLevel < this.atlasLevels; dstLevel++) {
                final int fDstLevel = dstLevel;
                int sourceLevel = dstLevel + this.atlasMipBias;
                var r = regions.get(dstLevel);
                r.bufferOffset(srcOff).bufferRowLength(0).bufferImageHeight(0);
                r.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(dstLevel).baseArrayLayer(0).layerCount(1);
                r.imageOffset(o -> o.x(X >> fDstLevel).y(Y >> fDstLevel).z(0));
                r.imageExtent(e -> e.width((TS * 3) >> sourceLevel).height((TS * 2) >> sourceLevel).depth(1));
                srcOff += mipBytes(sourceLevel);
            }
            vkCmdCopyBufferToImage(cmd, this.uploadStream.stagingBufferHandle(), this.atlas.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions);
        }
    }

    @Override
    public void endTextureUploads() {
        this.atlas.transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        this.inUploadBatch = false;
    }

    @Override
    public void free() {
        if (this.freed) return;
        this.freed = true;
        final long sampler = this.atlasSampler;
        if (sampler != VK_NULL_HANDLE) {
            //The sampler is referenced by submitted terrain descriptors, so its
            //lifetime follows Minecraft's graphics submission just like buffers
            //and images. Never destroy it immediately during renderer shutdown.
            this.ctx.vk().deferUntilSubmissionComplete(() -> {
                try {
                    vkDestroySampler(this.ctx.vk().device, sampler, null);
                } catch (RuntimeException | Error failure) {
                    Logger.error("Failed to destroy retired Vulkan model-atlas sampler", failure);
                }
            });
        }
        this.modelBuffer.free();
        this.modelColourBuffer.free();
        this.atlas.free();
    }
}
