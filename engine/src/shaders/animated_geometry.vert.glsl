#version 450

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_position;
layout(location = 3) out vec2 out_uv;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout (location = 3) in uvec4 in_boneIDs;
layout (location = 4) in  vec4 in_weights;

// the std430 is a identifier that tells the shader how the input is padded/alligned in memory, 430 means tightly packed, 140 means 16 byte alignment
layout(std430,set = 3, binding = 0) readonly buffer globalBoneTransform{
    mat4 transforms[];
} bone_transforms;

layout(std430,set = 3, binding = 1) readonly buffer globalBoneNormalTransform{
    mat3 transforms[];
} bone_normal_transforms;

layout(set = 0, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normals;
} model;

// per frame so updated at start of command buffer
layout(set = 1, binding = 0) uniform ViewProjUBO{
    mat4 view;
    mat4 projection;
} vp_uniforms;

vec3 applyBoneTransform(vec4 p) {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        mat4 boneTransform = bone_transforms.transforms[in_boneIDs[i]];
        result += in_weights[i] * (boneTransform * p).xyz;
    }
    return result;
}
// Normals represent the orientation of a surface (a plane), not a vector attached to a point.
// They must be transformed so that they remain perpendicular to the surface after transformation.
// For non-uniform scaling, transforming normals with the same matrix as positions breaks this perpendicularity,
// so normals must be transformed using the inverse-transpose of the linear part of the transform.
vec3 skinNormal(vec3 n) {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        mat3 normalTransform = bone_normal_transforms.transforms[in_boneIDs[i]];
        result += in_weights[i] * (normalTransform * n);
    }
    return normalize(result);
}

void main() {
    vec3 position = applyBoneTransform(vec4(in_position, 1.0));
    vec3 normal = skinNormal(in_normal);

    gl_Position =
    vp_uniforms.projection *
    vp_uniforms.view *
    model.model *
    vec4(position, 1.0);

    out_uv = in_uv;
    out_normal = normalize((model.normals * vec4(normal, 0.0)).xyz);
    out_position = model.model * vec4(position, 1.0);
    out_color = vec3(1.0);

}
