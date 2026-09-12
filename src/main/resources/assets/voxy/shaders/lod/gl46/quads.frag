#version 460 core
//Use quad shuffling to compute fragment mip
//#extension GL_KHR_shader_subgroup_quad: enable
#ifdef USE_SINGLE_TRI
#define USE_NV_BARRY
#endif

#ifdef USE_NV_BARRY
#extension GL_NV_fragment_shader_barycentric: require
#endif

#ifdef VOXY_VULKAN
layout(binding = 8) uniform sampler2D blockModelAtlas;
layout(binding = 10) uniform sampler2D depthTex;
#else
layout(binding = 0) uniform sampler2D blockModelAtlas;
layout(binding = 2) uniform sampler2D depthTex;
#endif

//#define DEBUG_RENDER

//TODO: need to fix when merged quads have discardAlpha set to false but they span multiple tiles
// however they are not a full block

layout(location = 0) in flat uvec4 interData;
#ifndef USE_NV_BARRY
layout(location = 1) in vec2 uv;
#endif

#ifdef DEBUG_RENDER
layout(location = 7) in flat uint quadDebug;
#endif

//On non-Apple macOS GPUs, older MoltenVK/SPIRV-Cross builds can misclassify
//covered fragments while guarding stores after OpKill.  The result is a depth
//write without the matching colour write, which the SSAO pass deliberately
//visualises as red.  A zero sample mask has the same framebuffer semantics as
//discard for this single-sample pipeline without entering MoltenVK's broken
//discard/helper-thread path.  Other backends retain the original OpKill path.
#ifdef VOXY_VULKAN_SAMPLE_MASK_DISCARD
#define VOXY_DISCARD_FRAGMENT() { gl_SampleMask[0] = 0; return; }
#else
#define VOXY_DISCARD_FRAGMENT() { discard; return; }
#endif


#if defined(VOXY_VULKAN_PHOTON)
layout(location = 0) out vec4 outColour;
layout(location = 1) out vec4 photonGbuffer;
layout(location = 2) out vec2 photonLod;
#define MODEL_BUFFER_BINDING 3
#import <voxy:lod/block_model.glsl>
#elif !defined(PATCHED_SHADER)
layout(location = 0) out vec4 outColour;
#else
//Bind the model buffer and import the model system as we need it
#define MODEL_BUFFER_BINDING 3
#import <voxy:lod/block_model.glsl>
#endif

#import <voxy:lod/gl46/bindings.glsl>
#import <voxy:lod/lighting.glsl>
#import <voxy:util/depthutils.glsl>

vec4 uint2vec4RGBA(uint colour) {
    return vec4((uvec4(colour)>>uvec4(24,16,8,0))&uvec4(0xFF))/255.0;
}

uint tintingState() {
    return (interData.x>>2)&3u;
}

bool useDiscard() {
    return (interData.x&1u)==1u;
}

uint getFace() {
    return (interData.x>>4)&7u;
}

uint getModelId() {
    return interData.x>>16;
}

vec2 getBaseUV() {
    uint face = getFace();
    uint modelId = interData.x>>16;
    vec2 modelUV = vec2(modelId&0xFFu, (modelId>>8)&0xFFu)*(1.0/(256.0));
    return modelUV + (vec2(face>>1, face&1u) * (1.0/(vec2(3.0, 2.0)*256.0)));
}

#ifdef PATCHED_SHADER
struct VoxyFragmentParameters {
    vec4 sampledColour;
    vec2 tile;
    vec2 uv;
    uint face;
    uint modelId;
    vec2 lightMap;
    vec4 tinting;
    uint customId;
};

void voxy_emitFragment(VoxyFragmentParameters parameters);
#else
vec4 computeColour(vec2 texturePos, vec4 colour) {
    uint tintingFunction = tintingState();
    bool doTint = tintingFunction==2;
    if (tintingFunction == 1) {
        vec4 tintTest = textureLod(blockModelAtlas, texturePos, 0);
        if (abs(tintTest.r-tintTest.g) < 0.02f && abs(tintTest.g-tintTest.b) < 0.02f) {
            doTint = true;
        }
    }
    if (doTint) {
        colour *= uint2vec4RGBA(interData.z).yzwx;
    }
    return (colour * uint2vec4RGBA(interData.y)) + vec4(0,0,0,float(interData.w&0xFFu)/255);
}
#endif

#ifdef VOXY_VULKAN_PHOTON
vec2 photonSignNonZero(vec2 v) {
    return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

vec2 photonEncodeUnitVector(vec3 v) {
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    p = v.z <= 0.0 ? ((1.0 - abs(p.yx)) * photonSignNonZero(p)) : p;
    return 0.5 * p + 0.5;
}

float photonPackUnorm2x8(vec2 v) {
    return dot(floor(255.0 * v + 0.5), vec2(1.0 / 65535.0, 256.0 / 65535.0));
}

void emitPhotonOpaque(vec4 sampledColour) {
    uint modelId = getModelId();
    BlockModel model = modelData[modelId];

    uint tintingFunction = tintingState();
    bool doTint = tintingFunction == 2u;
    if (tintingFunction == 1u) {
        vec4 tintTest = texture(blockModelAtlas, getBaseUV(), -2.0);
        if (abs(tintTest.r - tintTest.g) < 0.02 && abs(tintTest.g - tintTest.b) < 0.02) {
            doTint = true;
        }
    }
    vec4 tint = doTint ? uint2vec4RGBA(interData.z).yzwx : vec4(1.0);

    uint face = getFace();
    face ^= uint((face & 1u) != uint(gl_FrontFacing != ((face >> 1u) != 0u)));
    vec3 normal = vec3(
        uint((face >> 1u) == 2u),
        uint((face >> 1u) == 0u),
        uint((face >> 1u) == 1u)
    ) * (float(int(face) & 1) * 2.0 - 1.0);

    vec3 baseColor = sampledColour.rgb * tint.rgb;
    uint materialMask = max(model.customId - 10000u, 0u);
    photonGbuffer.x = photonPackUnorm2x8(baseColor.rg);
    photonGbuffer.y = photonPackUnorm2x8(vec2(
        baseColor.b,
        clamp(float(materialMask) * (1.0 / 255.0), 0.0, 1.0)
    ));
    photonGbuffer.z = photonPackUnorm2x8(photonEncodeUnitVector(normal));
    photonGbuffer.w = photonPackUnorm2x8(getLightmapUv(interData.y));

#ifdef USE_REVERSE_Z
    float forwardDepth = 1.0 - gl_FragCoord.z;
#else
    float forwardDepth = gl_FragCoord.z;
#endif
    photonLod = vec2(forwardDepth, vxRenderDistance);
}
#endif

void main() {
#ifdef VOXY_VULKAN_SAMPLE_MASK_DISCARD
    gl_SampleMask[0] = 1;
#endif

    vec2 tile;
#ifdef USE_NV_BARRY
#ifdef USE_SINGLE_TRI
    if (gl_BaryCoordNV.x>=0.5||gl_BaryCoordNV.y>=0.5) VOXY_DISCARD_FRAGMENT();
    vec2 uv = gl_BaryCoordNV.yx*(vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)+1)*2;
#else
    vec2 uv = mix(gl_BaryCoordNV.yx, 1-gl_BaryCoordNV.xz, gl_PrimitiveID&1)*(vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)+1);
#endif
#endif

    vec2 uv2 = modf(uv, tile)*(1.0/(vec2(3.0,2.0)*256.0));
    vec4 colour;
    vec2 texPos = uv2 + getBaseUV();
    {
        vec2 uvSmol = uv*(1.0/(vec2(3.0,2.0)*256.0));
        vec2 dx = dFdx(uvSmol);
        vec2 dy = dFdy(uvSmol);
        colour = textureGrad(blockModelAtlas, texPos, dx, dy);
    }

#if !defined(PATCHED_SHADER_ALLOW_DERIVATIVES) && !defined(VOXY_VULKAN_SAMPLE_MASK_DISCARD)
    if (gl_HelperInvocation) {
        return;
    }
#endif

    if (any(notEqual(clamp(tile, vec2(0), vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)), tile))) {
        VOXY_DISCARD_FRAGMENT();
    }

    if (DEPTH_SCALAR_COMPARE(gl_FragCoord.z, texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r)) {
        VOXY_DISCARD_FRAGMENT();
    }

#ifndef TRANSLUCENT
    colour.a = 1.0f;
    if (useDiscard() && (textureLod(blockModelAtlas, texPos, 0).a <= 0.1f)) {
#else
    if (textureLod(blockModelAtlas, texPos, 0).a == 0.0f) {
#endif
#ifndef DEBUG_RENDER
        VOXY_DISCARD_FRAGMENT();
#endif
    }

#if !defined(PATCHED_SHADER_ALLOW_DERIVATIVES) && !defined(VOXY_VULKAN_SAMPLE_MASK_DISCARD)
    if (gl_HelperInvocation) {
        return;
    }
#endif

#ifdef VOXY_VULKAN_PHOTON
    vec4 sampledColour = colour;
    outColour = computeColour(texPos, colour);
    // Use the already-resolved texel colour; the pack contract applies the tint
    // separately and does not multiply Voxy's fallback face/light colour.
    uint modelId = getModelId();
    BlockModel model = modelData[modelId];
    uint tintingFunction = tintingState();
    bool doTint = tintingFunction == 2u;
    if (tintingFunction == 1u) {
        vec4 tintTest = texture(blockModelAtlas, texPos, -2.0);
        if (abs(tintTest.r - tintTest.g) < 0.02 && abs(tintTest.g - tintTest.b) < 0.02) doTint = true;
    }
    vec4 tint = doTint ? uint2vec4RGBA(interData.z).yzwx : vec4(1.0);
    uint face = getFace();
    face ^= uint((face & 1u) != uint(gl_FrontFacing != ((face >> 1u) != 0u)));
    vec3 normal = vec3(uint((face >> 1u) == 2u), uint((face >> 1u) == 0u), uint((face >> 1u) == 1u))
        * (float(int(face) & 1) * 2.0 - 1.0);
    vec3 baseColor = sampledColour.rgb * tint.rgb;
    uint materialMask = max(model.customId - 10000u, 0u);
    photonGbuffer.x = photonPackUnorm2x8(baseColor.rg);
    photonGbuffer.y = photonPackUnorm2x8(vec2(baseColor.b, clamp(float(materialMask) / 255.0, 0.0, 1.0)));
    photonGbuffer.z = photonPackUnorm2x8(photonEncodeUnitVector(normal));
    photonGbuffer.w = photonPackUnorm2x8(getLightmapUv(interData.y));
#ifdef USE_REVERSE_Z
    photonLod = vec2(1.0 - gl_FragCoord.z, vxRenderDistance);
#else
    photonLod = vec2(gl_FragCoord.z, vxRenderDistance);
#endif

#ifdef DEBUG_RENDER
    uint hash = quadDebug*1231421+123141;
    hash ^= hash>>16;
    hash = hash*1231421+123141;
    hash ^= hash>>16;
    hash = hash * 1827364925 + 123325621;
    outColour = vec4(float(hash&15u)/15, float((hash>>4)&15u)/15, float((hash>>8)&15u)/15, 0);
#endif

#elif !defined(PATCHED_SHADER)
    colour = computeColour(texPos, colour);
    outColour = colour;

#ifdef DEBUG_RENDER
    uint hash = quadDebug*1231421+123141;
    hash ^= hash>>16;
    hash = hash*1231421+123141;
    hash ^= hash>>16;
    hash = hash * 1827364925 + 123325621;
    outColour = vec4(float(hash&15u)/15, float((hash>>4)&15u)/15, float((hash>>8)&15u)/15, 0);
#endif

#else
    uint modelId = getModelId();
    BlockModel model = modelData[modelId];
    uint tintingFunction = tintingState();
    bool doTint = tintingFunction==2;
    if (tintingFunction==1) {
        vec4 tintTest = texture(blockModelAtlas, texPos, -2);
        if (abs(tintTest.r-tintTest.g) < 0.02f && abs(tintTest.g-tintTest.b) < 0.02f) {
            doTint = true;
        }
    }
    vec4 tint = vec4(1);
    if (doTint) {
        tint = uint2vec4RGBA(interData.z).yzwx;
    }

    uint face = getFace();
    face ^= uint((face&1u)!=uint(gl_FrontFacing!=((face>>1)!=0u)));
    voxy_emitFragment(VoxyFragmentParameters(colour, tile, texPos, face, modelId, getLightmapUv(interData.y), tint, model.customId));
#endif
}

#import <voxy:util/depthutils.glsl>
