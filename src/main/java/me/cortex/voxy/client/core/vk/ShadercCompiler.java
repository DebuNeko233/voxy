package me.cortex.voxy.client.core.vk;

import me.cortex.voxy.client.core.gl.shader.ShaderType;

import java.nio.ByteBuffer;

import static org.lwjgl.util.shaderc.Shaderc.*;

/** Runtime GLSL -> SPIR-V compiler for the native Vulkan backend. */
public final class ShadercCompiler {
    public static ByteBuffer compile(String source, ShaderType type, String name) {
        long compiler = shaderc_compiler_initialize();
        if (compiler == 0L) {
            throw new IllegalStateException("shaderc_compiler_initialize failed for " + name);
        }

        long options = 0L;
        try {
            options = shaderc_compile_options_initialize();
            if (options == 0L) {
                throw new IllegalStateException("shaderc_compile_options_initialize failed for " + name);
            }

            shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
            shaderc_compile_options_set_auto_bind_uniforms(options, true);
            shaderc_compile_options_set_auto_map_locations(options, true);
            shaderc_compile_options_add_macro_definition(options, "VOXY_VULKAN", "1");

            int kind = switch (type) {
                case VERTEX -> shaderc_vertex_shader;
                case FRAGMENT -> shaderc_fragment_shader;
                case COMPUTE -> shaderc_compute_shader;
                default -> throw new IllegalArgumentException("Unsupported stage for VK: " + type);
            };

            long result = shaderc_compile_into_spv(compiler, source, kind, name, "main", options);
            if (result == 0L) {
                throw new IllegalStateException("shaderc_compile_into_spv returned null for " + name);
            }
            try {
                int status = shaderc_result_get_compilation_status(result);
                if (status != shaderc_compilation_status_success) {
                    throw new IllegalStateException("SPIR-V compile failed for " + name + ":\n"
                            + shaderc_result_get_error_message(result));
                }
                ByteBuffer spv = shaderc_result_get_bytes(result);
                if (spv == null || !spv.hasRemaining()) {
                    throw new IllegalStateException("shaderc returned empty SPIR-V for " + name);
                }
                //Result-owned storage becomes invalid at shaderc_result_release;
                //copy it into JVM-owned direct memory first.
                ByteBuffer copy = ByteBuffer.allocateDirect(spv.remaining());
                copy.put(spv).flip();
                return copy;
            } finally {
                shaderc_result_release(result);
            }
        } finally {
            if (options != 0L) shaderc_compile_options_release(options);
            shaderc_compiler_release(compiler);
        }
    }

    private ShadercCompiler() {}
}
