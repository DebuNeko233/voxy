package me.cortex.voxy.client.core.vk.compat;

import me.cortex.voxy.common.Logger;
import net.fabricmc.loader.api.FabricLoader;

import java.lang.reflect.Method;
import java.util.function.Supplier;

/**
 * Optional bridge to Vitrail's Vulkan frame ownership.
 *
 * Vitrail keeps one dynamic-rendering pass open across consecutive world geometry
 * passes and also calls Sodium's opaque terrain path a second time for its shadow
 * map. Voxy records native Vulkan commands directly into Minecraft's command
 * buffer, so it must neither run during that shadow re-entry nor begin its own
 * dynamic-rendering scopes while Vitrail still has a held pass open.
 *
 * This bridge deliberately uses reflection so Vitrail remains an optional runtime
 * dependency. If its internal compatibility surface changes, Voxy skips native
 * world frames rather than risking invalid nested rendering commands and a native
 * driver crash.
 */
public final class VitrailCompat {
    private static final Supplier<String> FLUSH_CAUSE = () -> "Voxy native Vulkan frame";
    private static final StackWalker STACK_WALKER = StackWalker.getInstance();

    private static final boolean LOADED;
    private static final Method DRAWING_SHADOW;
    private static final Method FLUSH_GEOMETRY_HOLD;
    private static volatile boolean bridgeFailed;

    static {
        boolean loaded = FabricLoader.getInstance().isModLoaded("vitrail");
        Method drawingShadow = null;
        Method flushGeometryHold = null;
        boolean failed = false;

        if (loaded) {
            try {
                ClassLoader loader = VitrailCompat.class.getClassLoader();
                Class<?> terrainDraw = Class.forName("dev.vitrail.render.TerrainDraw", false, loader);
                Class<?> geometryHold = Class.forName("dev.vitrail.render.GeometryHold", false, loader);
                drawingShadow = terrainDraw.getMethod("drawingShadow");
                flushGeometryHold = geometryHold.getMethod("flush", Supplier.class);
                Logger.info("Voxy VK: Vitrail Vulkan frame compatibility bridge enabled");
            } catch (Throwable t) {
                failed = true;
                Logger.error("Voxy VK: Vitrail is loaded but its frame compatibility API could not be resolved; native Voxy world drawing will be skipped for safety", t);
            }
        }

        LOADED = loaded;
        DRAWING_SHADOW = drawingShadow;
        FLUSH_GEOMETRY_HOLD = flushGeometryHold;
        bridgeFailed = failed;
    }

    private VitrailCompat() {
    }

    /**
     * Vitrail deliberately uses LevelExtractor.allChanged() as a safe frame-boundary
     * door when its own Sodium terrain mesh format, face shading, block-state IDs or
     * shader-pack terrain requirements change. None of those rebuilds changes Voxy's
     * native Vulkan geometry format or model atlas. Detect those direct Vitrail calls
     * so Voxy can keep its multi-gigabyte VkRenderCore alive while Sodium rebuilds.
     *
     * Ordinary Minecraft/resource-pack/F3+A allChanged calls do not have a
     * dev.vitrail.* frame and therefore still rebuild Voxy, which is required when
     * the actual block model/texture atlas may have changed.
     */
    public static boolean isVitrailDrivenWorldRebuild() {
        if (!LOADED) return false;
        return STACK_WALKER.walk(frames -> frames
                .map(StackWalker.StackFrame::getClassName)
                .anyMatch(name -> name.startsWith("dev.vitrail.")));
    }

    /**
     * Prepares the shared Minecraft Vulkan command buffer for one main-world Voxy
     * frame.
     *
     * @return true when Voxy may record its native frame; false for Vitrail's
     * shadow-terrain re-entry or when the optional bridge is unavailable/broken.
     */
    public static boolean prepareForWorldFrame() {
        if (!LOADED) return true;
        if (bridgeFailed || DRAWING_SHADOW == null || FLUSH_GEOMETRY_HOLD == null) return false;

        try {
            // Vitrail deliberately re-enters Sodium's OPAQUE path while building
            // its shadow map. That is not a second camera frame and Voxy must not
            // write its terrain/depth/composite into it.
            if ((boolean) DRAWING_SHADOW.invoke(null)) {
                return false;
            }

            // Vitrail may keep vkCmdBeginRendering open after Sodium's opaque
            // chunk work. End that hold through Vitrail's own RenderPass object
            // before Voxy starts any raw Vulkan dynamic-rendering scope.
            FLUSH_GEOMETRY_HOLD.invoke(null, FLUSH_CAUSE);
            return true;
        } catch (Throwable t) {
            bridgeFailed = true;
            Logger.error("Voxy VK: Vitrail frame handoff failed; native Voxy world drawing is disabled to avoid invalid Vulkan command nesting", t);
            return false;
        }
    }
}
