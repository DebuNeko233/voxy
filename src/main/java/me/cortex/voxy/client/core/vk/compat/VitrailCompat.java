package me.cortex.voxy.client.core.vk.compat;

import com.mojang.blaze3d.textures.GpuTextureView;
import me.cortex.voxy.common.Logger;
import net.fabricmc.loader.api.FabricLoader;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Optional runtime bridge between Voxy's native Vulkan renderer and Vitrail.
 *
 * <p>Vitrail is intentionally not a compile dependency. The bridge resolves the
 * small runtime surface Voxy needs through reflection and fails closed if that
 * surface changes: native Voxy keeps its ordinary renderer, but it will not try
 * to write into unknown shader-pack targets.</p>
 *
 * <p>Texture views are never cached. Vitrail recreates its colour targets on a
 * resize and explicitly requires callers to look the current view up again at
 * every use.</p>
 */
public final class VitrailCompat {
    private static final Supplier<String> FLUSH_CAUSE = () -> "Voxy native Vulkan frame";

    private static final boolean LOADED;
    private static final Method DRAWING_PACK;
    private static final Method DRAWING_SHADOW;
    private static final Method FLUSH_GEOMETRY_HOLD;
    private static final Field ACTIVE_CHAIN;
    private static final Field CHAIN_TERRAIN;
    private static final Field TERRAIN_TARGETS;
    private static final Field TERRAIN_LOADED;
    private static final Method TARGETS_SCHEDULE;
    private static final Method TARGETS_VIEW;
    private static final Method SCHEDULE_STEP;
    private static final Method BOUND_READ;
    private static final Method LOADED_PATH;

    private static volatile boolean bridgeFailed;
    private static volatile boolean announcedPhotonBridge;

    static {
        boolean loaded = FabricLoader.getInstance().isModLoaded("vitrail");
        Method drawingPack = null;
        Method drawingShadow = null;
        Method flushGeometryHold = null;
        Field activeChain = null;
        Field chainTerrain = null;
        Field terrainTargets = null;
        Field terrainLoaded = null;
        Method targetsSchedule = null;
        Method targetsView = null;
        Method scheduleStep = null;
        Method boundRead = null;
        Method loadedPath = null;
        boolean failed = false;

        if (loaded) {
            try {
                ClassLoader loader = VitrailCompat.class.getClassLoader();
                Class<?> packChain = Class.forName("dev.vitrail.render.PackChain", false, loader);
                Class<?> terrainDraw = Class.forName("dev.vitrail.render.TerrainDraw", false, loader);
                Class<?> geometryHold = Class.forName("dev.vitrail.render.GeometryHold", false, loader);
                Class<?> colorTargets = Class.forName("dev.vitrail.render.ColorTargets", false, loader);
                Class<?> targetSchedule = Class.forName("dev.vitrail.pack.target.TargetSchedule", false, loader);
                Class<?> targetSide = Class.forName("dev.vitrail.pack.target.TargetSchedule$Side", false, loader);
                Class<?> targetBound = Class.forName("dev.vitrail.pack.target.TargetSchedule$Bound", false, loader);
                Class<?> loadedProgram = Class.forName("dev.vitrail.glsl.PackProgram$Loaded", false, loader);

                drawingPack = packChain.getMethod("drawingPack");
                drawingShadow = terrainDraw.getMethod("drawingShadow");
                flushGeometryHold = geometryHold.getMethod("flush", Supplier.class);

                activeChain = packChain.getDeclaredField("active");
                activeChain.setAccessible(true);
                chainTerrain = packChain.getDeclaredField("terrain");
                chainTerrain.setAccessible(true);
                terrainTargets = terrainDraw.getDeclaredField("targets");
                terrainTargets.setAccessible(true);
                terrainLoaded = terrainDraw.getDeclaredField("loaded");
                terrainLoaded.setAccessible(true);

                targetsSchedule = colorTargets.getDeclaredMethod("schedule");
                targetsSchedule.setAccessible(true);
                targetsView = colorTargets.getDeclaredMethod("view", int.class, targetSide);
                targetsView.setAccessible(true);
                scheduleStep = targetSchedule.getMethod("step", String.class);
                boundRead = targetBound.getMethod("read", int.class);
                loadedPath = loadedProgram.getMethod("path");

                Logger.info("Voxy VK: Vitrail frame/target compatibility bridge enabled");
            } catch (Throwable t) {
                failed = true;
                Logger.error("Voxy VK: Vitrail is loaded but its compatibility surface could not be resolved; shader-pack target export is disabled", t);
            }
        }

        LOADED = loaded;
        DRAWING_PACK = drawingPack;
        DRAWING_SHADOW = drawingShadow;
        FLUSH_GEOMETRY_HOLD = flushGeometryHold;
        ACTIVE_CHAIN = activeChain;
        CHAIN_TERRAIN = chainTerrain;
        TERRAIN_TARGETS = terrainTargets;
        TERRAIN_LOADED = terrainLoaded;
        TARGETS_SCHEDULE = targetsSchedule;
        TARGETS_VIEW = targetsView;
        SCHEDULE_STEP = scheduleStep;
        BOUND_READ = boundRead;
        LOADED_PATH = loadedPath;
        bridgeFailed = failed;
    }

    private VitrailCompat() {
    }

    /**
     * Ends Vitrail's held geometry pass before Voxy records raw Vulkan dynamic
     * rendering, and rejects Vitrail's shadow-map re-entry of Sodium's opaque
     * terrain hook.
     */
    public static boolean prepareForWorldFrame() {
        if (!LOADED) return true;
        if (bridgeFailed || DRAWING_PACK == null || DRAWING_SHADOW == null || FLUSH_GEOMETRY_HOLD == null) {
            return false;
        }

        try {
            if ((boolean) DRAWING_SHADOW.invoke(null)) return false;
            if ((boolean) DRAWING_PACK.invoke(null)) {
                FLUSH_GEOMETRY_HOLD.invoke(null, FLUSH_CAUSE);
            }
            return true;
        } catch (Throwable t) {
            fail("Vitrail frame handoff failed", t);
            return false;
        }
    }

    /** Current Photon/Vitrail targets used by native Voxy. Never retain this record across frames. */
    public record PhotonTargets(GpuTextureView gbuffer, GpuTextureView lodBridge) {
    }

    /**
     * Returns the current colortex1 and colortex17 views on the same target side
     * as Vitrail's solid terrain program. Null means the active pack cannot safely
     * accept native Voxy G-buffer output this frame.
     */
    public static PhotonTargets photonTargets() {
        if (!LOADED || bridgeFailed || DRAWING_PACK == null) return null;

        try {
            if (!(boolean) DRAWING_PACK.invoke(null)) return null;

            Object chain = ACTIVE_CHAIN.get(null);
            if (chain == null) return null;
            Object terrain = CHAIN_TERRAIN.get(chain);
            if (terrain == null) return null;

            Object targets = TERRAIN_TARGETS.get(terrain);
            if (targets == null) return null;

            Object solidLoaded = solidProgram(TERRAIN_LOADED.get(terrain));
            if (solidLoaded == null) return null;
            String path = (String) LOADED_PATH.invoke(solidLoaded);
            if (path == null || path.isEmpty()) return null;
            String servedBy = path.substring(path.lastIndexOf('/') + 1);

            Object schedule = TARGETS_SCHEDULE.invoke(targets);
            @SuppressWarnings("unchecked")
            Optional<Object> step = (Optional<Object>) SCHEDULE_STEP.invoke(schedule, servedBy);
            if (step.isEmpty()) return null;

            Object bound = step.get();
            Object gbufferSide = BOUND_READ.invoke(bound, 1);
            Object lodSide = BOUND_READ.invoke(bound, 17);
            GpuTextureView gbuffer = (GpuTextureView) TARGETS_VIEW.invoke(targets, 1, gbufferSide);
            GpuTextureView lod = (GpuTextureView) TARGETS_VIEW.invoke(targets, 17, lodSide);
            if (gbuffer == null || lod == null) return null;

            if (!announcedPhotonBridge) {
                announcedPhotonBridge = true;
                Logger.info("Voxy VK: exporting native terrain into Vitrail shader-pack colortex1/17");
            }
            return new PhotonTargets(gbuffer, lod);
        } catch (Throwable t) {
            fail("Vitrail shader-pack target lookup failed", t);
            return null;
        }
    }

    private static Object solidProgram(Object loadedPrograms) {
        if (!(loadedPrograms instanceof Map<?, ?> map)) return null;
        for (var entry : map.entrySet()) {
            if (entry.getKey() != null && "SOLID".equals(entry.getKey().toString())) {
                return entry.getValue();
            }
        }
        return null;
    }

    private static void fail(String what, Throwable t) {
        if (!bridgeFailed) {
            bridgeFailed = true;
            Logger.error("Voxy VK: " + what + "; native shader-pack export is disabled to avoid corrupt Vulkan state", t);
        }
    }
}
