#version 450
#extension GL_EXT_debug_printf : enable
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

struct DirectionalLight {
    vec4  sun_position;
    vec4  sun_color;
    vec4  sky_zenith_color;
    vec4  sky_horizon_color;
    float sky_gradient_sharpness;
    float pad1;
    float pad2;
    float pad3;
};

layout(set = 1, binding = 1) uniform LightData {
    DirectionalLight sky;
    uint             point_light_count;
    uint             _pad0;
    uint             _pad1;
    uint             _pad2;
    vec4             point_light_positions[32]; // xyz = pos, w = radius
    vec4             point_light_colors[32];    // xyz = color (intensity baked in), w = unused
}
lights;

layout(set = 1, binding = 2) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 inverse_view;
    mat4 inverse_proj;
}
camera;
vec3 easedMix(vec3 a, vec3 b, float t, float power) {
    float eased = t < 0.5 ? 0.5 * pow(2.0 * t, power) : 1.0 - 0.5 * pow(2.0 - 2.0 * t, power);
    return mix(a, b, eased);
}

#define NO_HIT 1e30
#define EPS .00002

struct Ray {
    vec3 direction;
    vec3 origin;
};
float raySphereIntersect(Ray ray, vec3 s0, float sr) {
    // float a = dot(ray.direction, ray.direction);
    // always 1 if ray.direction is normalized, which it is
    float a     = 1.0;
    vec3  s0_r0 = ray.origin - s0;
    float b     = 2.0 * dot(ray.direction, s0_r0);
    float c     = dot(s0_r0, s0_r0) - (sr * sr);
    float disc  = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return NO_HIT;
    }
    float t = (-b - sqrt(disc)) / (2.0 * a);
    return t > 0.0 ? t : NO_HIT;
}

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
    vec4 position_with_hit = texture(u_position, in_uv).rgba;
    vec3 normal            = texture(u_normals, in_uv).rgb;
    vec3 albedo            = texture(u_color, in_uv).rgb;

    vec3 position     = position_with_hit.rgb;
    bool hit_geometry = position_with_hit.a == 0.0 ? false : true;

    // find which probe this pixel is in
    vec3 local_pos = position - info.start_position.xyz;

    vec3 volume_size = vec3(float(info.probe_x_count) * info.probe_spacing,
                            float(info.probe_y_count) * info.probe_spacing,
                            float(info.probe_z_count) * info.probe_spacing);

    float closest = NO_HIT;
    {
        // in uv is 0..1, ndc is -1..1
        vec2 ndc = in_uv * 2.0 - 1.0;
        // projection goes from ndc/clip space to view/camera space
        vec4 view = camera.inverse_proj * vec4(ndc, 1.0, 1.0);
        // undo perspective distortion
        view /= view.w;
        // inverse view goes from camera space to world space
        vec3 world_dir = normalize((camera.inverse_view * vec4(view.xyz, 0.0)).xyz);

        // the last collum on the right is always the transform coordinates
        vec3 ray_origin = camera.inverse_view[3].xyz;
        bool hit_light  = false;
        vec3 color      = vec3(0.0);
        for (int i = 0; i < lights.point_light_count; i++) {
            vec4 light_pos = lights.point_light_positions[i];
            vec4 light_col = lights.point_light_colors[i];
            Ray  ray       = Ray(world_dir, ray_origin);

            float d = raySphereIntersect(ray, light_pos.xyz, light_pos.w);
            if (d < closest) {
                closest   = d;
                color     = normalize(light_col.xyz);
                hit_light = true;
            }
        }
        if (hit_light) {
            f_color = vec4(color, 1.0);
            return;
        }
        float t       = max(world_dir.y, 0.0);
        vec3  sky     = easedMix(lights.sky.sky_zenith_color.xyz, lights.sky.sky_horizon_color.xyz, t, lights.sky.sky_gradient_sharpness);
        float sun_dot = dot(world_dir, normalize(lights.sky.sun_position.xyz));
        if (sun_dot > 0.99) {
            sky = lights.sky.sun_color.xyz;
            // sky = vec3(0.0, 1.0, 0.0);
        }
        if (!hit_geometry) {
            f_color = vec4(sky, 1.0);
            return;
        }
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
    float N = float(info.sqrt_ray_count * info.sqrt_ray_count);
    f_color = vec4(albedo * total_radiance * (4.0 / N), 0.7);
    // f_color = vec4(1.0, 0.0, 1.0, 1.0);

    // f_color = vec4(N);
    // f_color = vec4(local_pos, 1.0);
    vec3 normalized_local = local_pos / vec3(float(info.probe_x_count) * info.probe_spacing,
                                             float(info.probe_y_count) * info.probe_spacing,
                                             float(info.probe_z_count) * info.probe_spacing);
    // f_color               = vec4(normalized_local, 1.0);
}
