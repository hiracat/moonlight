#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D height_map;
layout(set = 0, binding = 1) uniform HeightMapUBO {
    float size;
    float height;
    float resolution;
}
ubo;

layout(rgba16f, set = 1, binding = 0) uniform image2D radiance_field;
layout(rgba16f, set = 1, binding = 1) uniform image2D above_radiance_field;

struct MeshInfo {
    uint vertex_offset;
    uint index_offset;
    uint index_count;
    uint _pad;
    vec4 aabb_local_min;
    vec4 aabb_local_max;
    mat4 local_to_world;
};

layout(set = 2, binding = 0) uniform RadianceConfigUBO {
    vec4     start_position;
    uint     count_x;
    uint     count_y;
    uint     count_z;
    uint     z_cols;
    uint     xy_cols;
    uint     xy_rows;
    uint     above_z_cols;
    uint     above_xy_cols;
    uint     above_xy_rows;
    uint     _pad;
    float    probe_spacing;
    float    interval_start;
    float    interval_end;
    uint     is_top_cascade;
    uint     sqrt_ray_count;
    uint     mesh_count;
    MeshInfo meshes[64];
}
config;

layout(set = 2, binding = 1) readonly buffer PositionBuffer {
    vec4 positions[];
}
pos;
layout(set = 2, binding = 2) readonly buffer IndexBuffer {
    uint indices[];
}
idx;
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

layout(set = 3, binding = 0) uniform LightData {
    DirectionalLight sky;
    uint             point_light_count;
    uint             _pad0;
    uint             _pad1;
    uint             _pad2;
    vec4             point_light_positions[32]; // xyz = pos, w = radius
    vec4             point_light_colors[32];    // xyz = color (intensity baked in), w = unused
}
lights;
vec3 easedMix(vec3 a, vec3 b, float t, float power) {
    float eased = t < 0.5 ? 0.5 * pow(2.0 * t, power) : 1.0 - 0.5 * pow(2.0 - 2.0 * t, power);
    return mix(a, b, eased);
}

#define NO_HIT 1e30
#define EPS .002

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
    float t = (-b - sqrt(disc)) / (2.0 * a);
    return t > 0.0 ? t : NO_HIT;
}

/**
Tomas Möller & Ben Trumbore (1997) Fast, Minimum Storage Ray-Triangle Intersection,
Journal of Graphics Tools
*/
float rayTriangleIntersect(Ray ray, vec3 point1, vec3 point2, vec3 point3) {
    vec3 edge1 = point2 - point1;
    vec3 edge2 = point3 - point1;

    vec3 directionxedge2 = cross(ray.direction, edge2);

    float determinant = dot(directionxedge2, edge1);

    // ray parallel to triangle plane
    if (abs(determinant) < EPS) {
        return -1.0;
    }
    float invdeterminant = 1.0 / determinant;
    vec3  origin_to_p1   = ray.origin - point1;
    float bary_1         = dot(directionxedge2, origin_to_p1) * invdeterminant;
    vec3  txedge1        = cross(origin_to_p1, edge1);
    float bary_2         = dot(txedge1, ray.direction) * invdeterminant;
    if (bary_1 < 0.0 || bary_2 < 0.0 || bary_1 + bary_2 > 1.0) {
        return -1.0;
    }

    float distance_along_ray = dot(txedge1, edge2) * invdeterminant;
    if (distance_along_ray < EPS) {
        return -1.0;
    }

    return distance_along_ray;
}

float sampleHeight(sampler2D hmap, vec2 xz, float hmap_size, float hmap_height) {
    vec2 uv = (xz / hmap_size) + 0.5; // world xz to uv, centered on 0
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return -1e10;
    }
    return texture(hmap, uv).r * hmap_height;
}

float rayHeightmapIntersect(Ray ray, sampler2D hmap, float hmap_size, float hmap_height, float start, float end) {
    int   steps     = 16; // coarse march steps
    float step_size = (end - start) / float(steps);

    float t_prev = start;

    for (int i = 1; i <= steps; i++) {
        float t    = start + float(i) * step_size;
        float y    = ray.origin.y + ray.direction.y * t;
        float h    = sampleHeight(hmap, ray.origin.xz + ray.direction.xz * t, hmap_size, hmap_height);
        float diff = y - h;

        if (diff < 0.0) {
            // crossed the surface — binary search between t_prev and t
            float t_lo = t_prev;
            float t_hi = t;
            for (int b = 0; b < 8; b++) {
                float t_mid = (t_lo + t_hi) * 0.5;
                float y_mid = ray.origin.y + ray.direction.y * t_mid;
                float h_mid = sampleHeight(hmap, ray.origin.xz + ray.direction.xz * t_mid, hmap_size, hmap_height);
                if (y_mid < h_mid) {
                    t_hi = t_mid;
                } else {
                    t_lo = t_mid;
                }
            }
            return (t_lo + t_hi) * 0.5;
        }

        t_prev = t;
    }
    return NO_HIT;
}

float segmentTriangleIntersect(Ray ray, vec3 A, vec3 B, vec3 C, float start, float end) {
    float d = rayTriangleIntersect(ray, A, B, C);
    if (d > start && d < end) {
        return d;
    }
    return NO_HIT;
}

float segmentSphereIntersect(Ray ray, vec3 pos, float radius, float start, float end) {
    float distance = raySphereIntersect(ray, pos, radius);
    if (distance > start && distance < end) {
        return distance;
    }
    return NO_HIT;
}

// adapted from intersectCube in https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
// compute the near and far intersections of the cube (stored in the x and y components) using the slab method
// no intersection means (really tNear > tFar)
float intersectAABB(Ray ray, vec3 boxMin, vec3 boxMax) {
    vec3  tMin  = (boxMin - ray.origin) / ray.direction;
    vec3  tMax  = (boxMax - ray.origin) / ray.direction;
    vec3  t1    = min(tMin, tMax);
    vec3  t2    = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    if (tNear > tFar) {
        return NO_HIT;
    } else {
        return tNear;
    }
}

// https://pbr-book.org/4ed/Geometry_and_Transformations/Spherical_Geometry#fragment-Reparameterizedirectionsinthez0portionoftheoctahedron-0
// modified because the book uses z as up
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

vec4 merge_intervals(vec4 near, vec4 far) {
    /* Far radiance can get occluded by near visibility term */
    const vec3 radiance = near.rgb + (far.rgb * near.a);

    return vec4(radiance, near.a * far.a);
}

ivec2 get_texel(uvec3 probe, uvec2 ray) {
    // --- z level ---
    uint z_col = probe.z % config.z_cols;
    uint z_row = probe.z / config.z_cols;

    // --- xy level ---
    uint xy_idx = probe.y * config.count_x + probe.x; // flatten x,y into one index
    uint xy_col = xy_idx % config.xy_cols;
    uint xy_row = xy_idx / config.xy_cols;

    uint texel_x = z_col * (config.xy_cols * config.sqrt_ray_count) + xy_col * config.sqrt_ray_count + ray.x;
    uint texel_y = z_row * (config.xy_rows * config.sqrt_ray_count) + xy_row * config.sqrt_ray_count + ray.y;

    return ivec2(texel_x, texel_y);
}

ivec2 get_above_texel(uvec3 probe, uvec2 ray) {
    uint z_col = probe.z % config.above_z_cols;
    uint z_row = probe.z / config.above_z_cols;

    uint xy_idx = probe.y * (config.count_x / 2) + probe.x;
    uint xy_col = xy_idx % config.above_xy_cols;
    uint xy_row = xy_idx / config.above_xy_cols;

    uint above_sqrt_ray = config.sqrt_ray_count * 2;

    uint texel_x = z_col * (config.above_xy_cols * above_sqrt_ray) + xy_col * above_sqrt_ray + ray.x;
    uint texel_y = z_row * (config.above_xy_rows * above_sqrt_ray) + xy_row * above_sqrt_ray + ray.y;

    return ivec2(texel_x, texel_y);
}

/* Sub-texel offset to bilinear interpolation weights */
float[8] trilinear_weights(vec3 ratio) {
    float w[8];
    // go from corner ( -1, -1, -1) with z outermost and x innermost, to (+1,+1,+1)
    w[0] = (1 - ratio.x) * (1 - ratio.y) * (1 - ratio.z);
    w[1] = (ratio.x) * (1 - ratio.y) * (1 - ratio.z);
    w[2] = (1 - ratio.x) * (ratio.y) * (1 - ratio.z);
    w[3] = (ratio.x) * (ratio.y) * (1 - ratio.z);
    w[4] = (1 - ratio.x) * (1 - ratio.y) * (ratio.z);
    w[5] = (ratio.x) * (1 - ratio.y) * (ratio.z);
    w[6] = (1 - ratio.x) * (ratio.y) * (ratio.z);
    w[7] = (ratio.x) * (ratio.y) * (ratio.z);

    return w;
}

void trilinear_samples(vec3 dest_center, vec3 trilinear_size, out float weights[8], out ivec3 base_index) {
    /* Coordinate of the top-left trilinear probe when floored */

    const vec3 base_coord = (dest_center / trilinear_size);

    const vec3 ratio = fract(base_coord); /* Sub-trilinear probe position */
    weights          = trilinear_weights(ratio);
    base_index       = ivec3(floor(base_coord)); /* Top-left trilinear probe coordinate */
}

void main() {
    uint flat_idx   = gl_GlobalInvocationID.x;
    uint ray_count  = config.sqrt_ray_count * config.sqrt_ray_count;
    uint ray_flat   = flat_idx % ray_count;
    uint probe_flat = flat_idx / ray_count;

    uint probe_x = probe_flat % config.count_x;
    uint probe_y = (probe_flat / config.count_x) % config.count_y;
    uint probe_z = probe_flat / (config.count_x * config.count_y);

    if (probe_x >= config.count_x || probe_y >= config.count_y || probe_z >= config.count_z) {
        return;
    }

    uint ray_row = ray_flat / config.sqrt_ray_count;
    uint ray_col = ray_flat % config.sqrt_ray_count;

    vec3 probe_world_pos = config.start_position.xyz + vec3(probe_x, probe_y, probe_z) * config.probe_spacing;

    vec3 direction = unitVectorFrom2d(float(ray_col), float(ray_row), float(config.sqrt_ray_count));

    ivec2 texel = get_texel(uvec3(probe_x, probe_y, probe_z), uvec2(ray_col, ray_row));

    // CHECK RAY FOR EACH TRIANGLE
    Ray   ray     = Ray(direction, probe_world_pos);
    float closest = NO_HIT;
    vec4  color   = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < config.mesh_count; i++) {
        MeshInfo mesh_info = config.meshes[i];

        vec4 world_aabb_max = mesh_info.local_to_world * mesh_info.aabb_local_max;
        vec4 world_aabb_min = mesh_info.local_to_world * mesh_info.aabb_local_min;

        // IF THE RAY MISSES THE AABB FOR THE MESH  SKIP THE MESH
        // float aabb_hit = intersectAABB(ray, world_aabb_min.xyz, world_aabb_max.xyz);
        // if (aabb_hit >= closest) {
        //     continue;
        // }

        // CHECK THE RAY AGAINST EVERY TRIANGLE, (should replace with a bvh probably, am lazy)
        for (int j = 0; j < mesh_info.index_count; j += 3) {
            vec4 p1 = vec4(pos.positions[idx.indices[j + 0 + mesh_info.index_offset] + mesh_info.vertex_offset]);
            vec4 p2 = vec4(pos.positions[idx.indices[j + 1 + mesh_info.index_offset] + mesh_info.vertex_offset]);
            vec4 p3 = vec4(pos.positions[idx.indices[j + 2 + mesh_info.index_offset] + mesh_info.vertex_offset]);

            vec3 worldp1 = (mesh_info.local_to_world * p1).xyz;
            vec3 worldp2 = (mesh_info.local_to_world * p2).xyz;
            vec3 worldp3 = (mesh_info.local_to_world * p3).xyz;

            float d = segmentTriangleIntersect(ray, worldp1, worldp2, worldp3, config.interval_start, config.interval_end);
            // if the distance is closer than the other then use this ray
            if (d < closest) {
                closest = d;
                color   = vec4(0.0, 0.0, 0.0, 0.0); // opaque, no emission, no transmittance
            }
        }
    }
    for (int i = 0; i < lights.point_light_count; i++) {
        float d = segmentSphereIntersect(
            ray, lights.point_light_positions[i].xyz, lights.point_light_positions[i].w, config.interval_start, config.interval_end);
        if (d < closest) {
            closest = d;
            // no early out here because different colors per light, and if two lights line up then we need to apply the closer light
            color = vec4(lights.point_light_colors[i].xyz, 0.0);
        }
    }
    float d = rayHeightmapIntersect(ray, height_map, ubo.size, ubo.height, config.interval_start, config.interval_end);

    if (d < closest) {
        closest = d;
        color   = vec4(0.0, 0.0, 0.0, 0.0); // opaque, no emission, no transmittance
    }

    uint above_sqrt_ray = config.sqrt_ray_count * 2;

    vec4 merged = vec4(0.0);
    if (config.is_top_cascade == 0) {
        // for each nearby ray, because we need to average the 4 surrounding rays
        float weights[8];
        ivec3 base_index;
        // find the spacing of the above probes, to get the weights and base index for what probes to use
        vec3 above_local_pos = probe_world_pos - config.start_position.xyz;
        trilinear_samples(above_local_pos, vec3(config.probe_spacing * 2.0), weights, base_index);

        for (int ray_idx = 0; ray_idx < 4; ray_idx++) {
            int ray_row_offset = ray_idx / 2;
            int ray_col_offset = ray_idx % 2;

            uint above_ray_row =
                clamp(uint(float(ray_row) / float(config.sqrt_ray_count) * float(above_sqrt_ray)) + ray_row_offset, 0, above_sqrt_ray - 1);
            uint above_ray_col =
                clamp(uint(float(ray_col) / float(config.sqrt_ray_count) * float(above_sqrt_ray)) + ray_col_offset, 0, above_sqrt_ray - 1);

            vec4 radiance = vec4(0.0);
            for (int probe_idx = 0; probe_idx < 8; probe_idx++) {
                // alternates
                int dx = probe_idx & 1;
                // every 2 switches
                int dy = (probe_idx >> 1) & 1;
                // every 4 switches
                int dz = (probe_idx >> 2) & 1;

                // ------------------- above
                // above cascade has half the probes per axis
                // its a count not an index so subtract one

                uint trilinear_probe_x = uint(clamp(base_index.x + dx, 0, int(config.count_x / 2) - 1));
                uint trilinear_probe_y = uint(clamp(base_index.y + dy, 0, int(config.count_y / 2) - 1));
                uint trilinear_probe_z = uint(clamp(base_index.z + dz, 0, int(config.count_z / 2) - 1));

                ivec2 above_texel =
                    get_above_texel(uvec3(trilinear_probe_x, trilinear_probe_y, trilinear_probe_z), uvec2(above_ray_col, above_ray_row));

                // -------------------  above
                radiance += merge_intervals(color, imageLoad(above_radiance_field, above_texel)) * weights[probe_idx];
            }
            merged += radiance;
        }
        merged /= 4.0;
    } else {
        float sun_dot   = dot(direction, normalize(lights.sky.sun_position.xyz));
        float t         = max(direction.y, 0.0);
        vec3  sky_color = easedMix(lights.sky.sky_horizon_color.xyz, lights.sky.sky_zenith_color.xyz, t, lights.sky.sky_gradient_sharpness);

        vec3 incoming_color = sky_color;
        if (sun_dot > (1.0 - clamp(lights.sky.sun_color.w, 0.0, 1.0) * 0.05)) {
            incoming_color = lights.sky.sun_color.xyz;
        }
        vec4 above_radiance = vec4(incoming_color, 1.0);
        merged              = merge_intervals(color, above_radiance);
    }

    // debug border
    // if (ray_col == 0 || ray_row == 0 || ray_col == config.sqrt_ray_count - 1 || ray_row == config.sqrt_ray_count - 1) {
    //     merged = vec4(0.0, 0.0, 0.0, 1.0);
    // }

    imageStore(radiance_field, texel, merged);
}
