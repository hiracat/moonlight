#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D height_map;

layout(rgba16f, set = 1, binding = 0) uniform image2D radiance_field;
layout(rgba16f, set = 1, binding = 1) uniform image2D above_radiance_field;

struct MeshInfo {
    uint vertex_offset;
    uint index_offset;
    uint index_count;
    uint _pad;
};

layout(set = 2, binding = 0) uniform RadianceConfigUBO {
    vec4     start_position;
    uint     count_x;
    uint     count_y;
    uint     count_z;
    uint     z_cols;
    uint     xy_cols;
    uint     xy_rows;
    float    probe_spacing;
    float    interval_start;
    float    interval_end;
    uint     is_top_cascade;
    uint     sqrt_ray_count;
    uint     mesh_count;
    MeshInfo meshes[64];
}
config;

layout(set = 3, binding = 0) readonly buffer PositionBuffer {
    vec3 positions[];
}
pos;
layout(set = 3, binding = 1) readonly buffer IndexBuffer {
    uint indices[];
}
idx;

void main() {
    uint flat_idx   = gl_GlobalInvocationID.x;
    uint ray_count  = config.sqrt_ray_count * config.sqrt_ray_count;
    uint ray_flat   = flat_idx % ray_count;
    uint probe_flat = flat_idx / ray_count;

    uint probe_x = probe_flat % config.count_x;
    uint probe_y = (probe_flat / config.count_x) % config.count_y;
    uint probe_z = probe_flat / (config.count_x * config.count_y);

    uint ray_col = ray_flat % config.sqrt_ray_count;
    uint ray_row = ray_flat / config.sqrt_ray_count;

    if (probe_x >= config.count_x || probe_y >= config.count_y || probe_z >= config.count_z) {
        return;
    }
    vec3 probe_world_pos = config.start_position.xyz + vec3(probe_x, probe_y, probe_z) * config.probe_spacing;
    // --- z level ---
    uint z_col = probe_z % config.z_cols;
    uint z_row = probe_z / config.z_cols;

    // --- xy level ---
    uint xy_idx = probe_y * config.count_x + probe_x; // flatten x,y into one index
    uint xy_col = xy_idx % config.xy_cols;
    uint xy_row = xy_idx / config.xy_cols;

    uint texel_x = z_col * (config.xy_cols * config.sqrt_ray_count) + xy_col * config.sqrt_ray_count + ray_col;
    uint texel_y = z_row * (config.xy_rows * config.sqrt_ray_count) + xy_row * config.sqrt_ray_count + ray_row;

    float r = float(probe_x) / float(config.count_x - 1u);
    float g = float(probe_y) / float(config.count_y - 1u);
    float b = float(probe_z) / float(config.count_z - 1u);

    vec4 result = vec4(r, g, b, 1.0);
    imageStore(radiance_field, ivec2(texel_x, texel_y), result);
}
