#version 450
layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_normals;
layout(set = 0, binding = 2) uniform sampler2D u_position;

layout(set = 0, binding = 3) uniform sampler2D cascade_0;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform RadianceInfo {
    vec4  start_position; // world space origin of the probe grid
    uint  probe_x_count;
    uint  probe_y_count;
    uint  probe_z_count;
    uint  z_cols;
    uint  xy_cols;
    uint  xy_rows;
    uint  sqrt_ray_count;
    float probe_spacing;
}
info;
vec3 unitVectorFrom2d(float x, float y, float range) {
    vec3 v;
    v.x = -1 + 2 * (x / range);
    v.y = -1 + 2 * (y / range);
    v.z = 1 - (abs(v.x) + abs(v.y));
    if (v.z < 0) {
        float xo = v.x;
        v.x      = (1 - abs(v.y)) * sign(xo);
        v.y      = (1 - abs(xo)) * sign(v.y);
    }
    return normalize(v);
}

void main() {
    vec3 position = texture(u_position, in_uv).rgb;
    vec3 normal   = texture(u_normals, in_uv).rgb;
    vec3 albedo   = texture(u_color, in_uv).rgb;

    // find which probe this pixel is in
    vec3 local_pos = position - info.start_position.xyz;

    vec3 volume_size = vec3(float(info.probe_x_count) * info.probe_spacing,
                            float(info.probe_y_count) * info.probe_spacing,
                            float(info.probe_z_count) * info.probe_spacing);

    if (any(lessThan(local_pos, vec3(0.0))) || any(greaterThan(local_pos, volume_size))) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint probe_x = uint(local_pos.x / info.probe_spacing);
    uint probe_y = uint(local_pos.y / info.probe_spacing);
    uint probe_z = uint(local_pos.z / info.probe_spacing);

    // clamp to grid bounds
    probe_x = clamp(probe_x, 0u, info.probe_x_count - 1u);
    probe_y = clamp(probe_y, 0u, info.probe_y_count - 1u);
    probe_z = clamp(probe_z, 0u, info.probe_z_count - 1u);

    uint z_col  = probe_z % info.z_cols;
    uint z_row  = probe_z / info.z_cols;
    uint xy_idx = probe_y * info.probe_x_count + probe_x;
    uint xy_col = xy_idx % info.xy_cols;
    uint xy_row = xy_idx / info.xy_cols;

    // integrate over all rays
    vec3  total_radiance = vec3(0.0);
    float total_weight   = 0.0;

    for (uint i = 0; i < info.sqrt_ray_count; i++) {
        for (uint j = 0; j < info.sqrt_ray_count; j++) {
            uint texel_x = z_col * (info.xy_cols * info.sqrt_ray_count) + xy_col * info.sqrt_ray_count + i;
            uint texel_y = z_row * (info.xy_rows * info.sqrt_ray_count) + xy_row * info.sqrt_ray_count + j;

            vec4 radiance = texelFetch(cascade_0, ivec2(texel_x, texel_y), 0);

            // weight by cosine with surface normal
            vec3  ray_dir = normalize(unitVectorFrom2d(j, i, float(info.sqrt_ray_count)));
            float weight  = max(0.0, dot(ray_dir, normal));

            total_radiance += radiance.rgb * weight;
            total_weight += weight;
        }
    }
    f_color = vec4(albedo * total_radiance * 1.0, 0.7);
    // f_color = vec4(total_radiance, 1.0);
    // f_color = vec4(local_pos, 1.0);
    vec3 normalized_local = local_pos / vec3(float(info.probe_x_count) * info.probe_spacing,
                                             float(info.probe_y_count) * info.probe_spacing,
                                             float(info.probe_z_count) * info.probe_spacing);
    // f_color               = vec4(normalized_local, 1.0);
}
