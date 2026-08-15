#version 450
#extension GL_EXT_debug_printf : enable
layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_normals;
layout(set = 0, binding = 2) uniform sampler2D u_position;

layout(set = 0, binding = 3) uniform sampler2D cascade_0;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 f_color;

layout(location = 1) out vec4 f_tmp1;
layout(location = 2) out vec4 f_tmp2;
layout(location = 3) out vec4 f_tmp3;
layout(location = 4) out vec4 f_tmp4;

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

/* fractional position within current cell made by 8 nearest probes(a cube) to trilinear interpolation weights */
float[8] trilinear_weights(vec3 ratio) {
    float w[8];
    // corners of the unit cell from (0,0,0) to (1,1,1), z outermost, x innermost
    // if all are 0, ie in the 0,0,0 corner, then this is 1, and all others end up 0
    w[0] = (1 - ratio.x) * (1 - ratio.y) * (1 - ratio.z);
    // this is similar but instead of the 1.0 place being (0,0,0) its 1, 0,0, becaause the goal is to prevent all pieces from zeroing out
    w[1] = (ratio.x) * (1 - ratio.y) * (1 - ratio.z);
    w[2] = (1 - ratio.x) * (ratio.y) * (1 - ratio.z);
    w[3] = (ratio.x) * (ratio.y) * (1 - ratio.z);
    w[4] = (1 - ratio.x) * (1 - ratio.y) * (ratio.z);
    w[5] = (ratio.x) * (1 - ratio.y) * (ratio.z);
    w[6] = (1 - ratio.x) * (ratio.y) * (ratio.z);
    w[7] = (ratio.x) * (ratio.y) * (ratio.z);

    return w;
}

// this function supports non uniform cell spacing(ie different lenghs on different axis)
void trilinear_samples(vec3 worldspace_position, vec3 worldspace_cell_spacing, out float weights[8], out ivec3 min_corner_probe_index) {
    /* Coordinate of the top-left trilinear probe when floored */

    const vec3 probe_cell_coordinate = (worldspace_position / worldspace_cell_spacing);

    const vec3 sub_probe_cell_coordinate = fract(probe_cell_coordinate); /* Sub-trilinear probe position */
    weights                              = trilinear_weights(sub_probe_cell_coordinate);
    min_corner_probe_index               = ivec3(floor(probe_cell_coordinate)); /* Top-left trilinear probe coordinate */
}
struct Ray {
    vec3 direction;
    vec3 origin;
};
float raySphereIntersect(Ray ray, vec3 s0, float sr) {
    // always 1 if ray.direction is normalized, which it is
    float a     = 1.0;
    vec3  s0_r0 = ray.origin - s0;
    float b     = 2.0 * dot(ray.direction, s0_r0);
    float c     = dot(s0_r0, s0_r0) - (sr * sr);
    float disc  = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return NO_HIT;
    }

    float sqrt_disc = sqrt(disc);
    float t_near    = (-b - sqrt_disc) / (2.0 * a);
    float t_far     = (-b + sqrt_disc) / (2.0 * a);

    if (t_near > 0.0) {
        return t_near;
    }
    if (t_far > 0.0) {
        return t_far;
    }
    return NO_HIT;
}

vec3 unitVectorFrom2d(float x, float y, float range) {
    vec3 v;
    v.x = -1 + 2 * ((x + 0.5) / range);
    v.z = -1 + 2 * ((y + 0.5) / range);
    v.y = 1 - (abs(v.x) + abs(v.z));
    if (v.y < 0) {
        float xo = v.x;
        v.x      = (1 - abs(v.z)) * sign(xo);
        v.z      = (1 - abs(xo)) * sign(v.z);
    }
    return normalize(v);
}

void main() {
    f_tmp1 = vec4(0.0, 0.0, 0.0, 0.0);
    f_tmp2 = vec4(0.0, 0.0, 0.0, 0.0);
    f_tmp3 = vec4(0.0, 0.0, 0.0, 0.0);
    f_tmp4 = vec4(0.0, 0.0, 0.0, 0.0);

    vec4 position_with_hit = texture(u_position, in_uv).rgba;
    vec3 normal            = texture(u_normals, in_uv).rgb;
    vec3 albedo            = texture(u_color, in_uv).rgb;
    vec3 position          = position_with_hit.xyz;
    bool hit_geometry      = position_with_hit.a == 0.0 ? false : true;

    vec2 ndc      = in_uv * 2.0 - 1.0;
    vec4 view_pos = camera.inverse_proj * vec4(ndc, 1.0, 1.0);
    view_pos /= view_pos.w;
    vec3  world_dir  = normalize((camera.inverse_view * vec4(view_pos.xyz, 0.0)).xyz);
    vec3  ray_origin = camera.inverse_view[3].xyz;
    vec3  color      = vec3(0.0);
    float closest    = NO_HIT;

    float surface_dist = hit_geometry ? length(position - ray_origin) : NO_HIT;
    for (uint i = 0; i < lights.point_light_count; i++) {
        Ray   r = Ray(world_dir, ray_origin);
        float d = raySphereIntersect(r, lights.point_light_positions[i].xyz, lights.point_light_positions[i].w);
        if (d < closest) {
            closest = d;
            color   = lights.point_light_colors[i].xyz;
        }
    }
    if (surface_dist > closest) {
        // sky and lights first
        f_color = vec4(color, 1.0);
        return;
    }
    if (!hit_geometry) {

        float t       = max(world_dir.y, 0.0);
        vec3  sky     = easedMix(lights.sky.sky_horizon_color.xyz, lights.sky.sky_zenith_color.xyz, t, lights.sky.sky_gradient_sharpness);
        float sun_dot = dot(world_dir, normalize(lights.sky.sun_position.xyz));
        if (sun_dot > (1.0 - clamp(lights.sky.sun_color.w, 0.0, 1.0) * 0.05)) {
            sky = lights.sky.sun_color.xyz;
        }
        f_color = vec4(sky, 1.0);
        return;
    }

    // only geometry pixels reach here

    // the position of the current pixel transformed so that the corner of the probe grid is at 0,0,0
    vec3 local_pos = position - info.start_position.xyz;
    // the total size of the grid
    vec3 grid_extent = vec3(float(info.probe_x_count - 1) * info.probe_spacing,
                            float(info.probe_y_count - 1) * info.probe_spacing,
                            float(info.probe_z_count - 1) * info.probe_spacing);

    // if any geometry is outside the probe, just return gray
    // TODO: i should return color of sky + sun mixed based on normal
    if (any(lessThan(local_pos, vec3(0.0))) || any(greaterThan(local_pos, grid_extent))) {
        f_color = vec4(vec3(0.3), 1.0);
        return;
    }
    float distance_weights[8];
    ivec3 min_probe_index;
    trilinear_samples(local_pos, vec3(info.probe_spacing), distance_weights, min_probe_index);

    f_tmp1 = vec4(local_pos, 1.0);
    f_tmp2 = vec4(min_probe_index, 1.0);

    vec3 total_radiance = vec3(0.0);
    for (int probe_idx = 0; probe_idx < 8; probe_idx++) {
        int dx = probe_idx & 1;
        int dy = (probe_idx >> 1) & 1;
        int dz = (probe_idx >> 2) & 1;
        // you have to clamp for things that are exactly on the boundary of the probe volume cause sometimes it would go ouside the probe
        // volume, but its weight will be zero anyway so doesnt matter
        uint probe_x = uint(clamp(min_probe_index.x + dx, 0, int(info.probe_x_count) - 1));
        uint probe_y = uint(clamp(min_probe_index.y + dy, 0, int(info.probe_y_count) - 1));
        uint probe_z = uint(clamp(min_probe_index.z + dz, 0, int(info.probe_z_count) - 1));

        uint z_col  = probe_z % info.z_cols;
        uint z_row  = probe_z / info.z_cols;
        uint xy_idx = probe_y * info.probe_x_count + probe_x;
        uint xy_col = xy_idx % info.xy_cols;
        uint xy_row = xy_idx / info.xy_cols;

        vec3 probe_radiance = vec3(0.0);
        for (uint i = 0; i < info.sqrt_ray_count; i++) {
            for (uint j = 0; j < info.sqrt_ray_count; j++) {
                vec3 ray_dir = normalize(unitVectorFrom2d(float(i), float(j), float(info.sqrt_ray_count)));

                uint texel_x = z_col * (info.xy_cols * info.sqrt_ray_count) + xy_col * info.sqrt_ray_count + i;
                uint texel_y = z_row * (info.xy_rows * info.sqrt_ray_count) + xy_row * info.sqrt_ray_count + j;

                float similarity_weight = max(0.0, dot(ray_dir, normal));

                vec4 rad = texelFetch(cascade_0, ivec2(texel_x, texel_y), 0);
                f_tmp3.r += similarity_weight;
                f_tmp3.g += rad.a;

                f_tmp3.b += rad.a * similarity_weight;
                f_tmp3.a = 1.0;

                probe_radiance += rad.rgb * similarity_weight * rad.a;
            }
        }
        total_radiance += probe_radiance * distance_weights[probe_idx];
    }

    float ray_count = float(info.sqrt_ray_count * info.sqrt_ray_count);
    f_color         = vec4(albedo * total_radiance * (4.0 / ray_count), 1.0);
}
