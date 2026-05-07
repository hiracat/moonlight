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

layout(set = 3, binding = 0) readonly buffer PositionBuffer {
    vec3 positions[];
}
pos;
layout(set = 3, binding = 1) readonly buffer IndexBuffer {
    uint indices[];
}
idx;

layout(set = 4, binding = 0) uniform LightData {
    vec4 sun_direction; // xyz = direction, w = unused
    vec4 sun_color;     // xyz = color (intensity baked in), w = unused
    uint point_light_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    vec4 point_light_positions[32]; // xyz = pos, w = radius
    vec4 point_light_colors[32];    // xyz = color (intensity baked in), w = unused
}
lights;

#define EPS .00002

struct Ray {
    vec3 direction;
    vec3 origin;
};
float raySphereIntersect(Ray ray, vec3 s0, float sr) {
    // - r0: ray origin
    // - rd: normalized ray direction
    // - s0: sphere center
    // - sr: sphere radius
    // - Returns distance from r0 to first intersecion with sphere,
    //   or -1.0 if no intersection.
    // float a     = dot(ray.direction, ray.direction);
    // always 1 if ray.direction is normalized, which it is
    float a     = 1.0;
    vec3  s0_r0 = ray.origin - s0;
    float b     = 2.0 * dot(ray.direction, s0_r0);
    float c     = dot(s0_r0, s0_r0) - (sr * sr);
    if (b * b - 4.0 * a * c < 0.0) {
        return -1.0;
    }
    return (-b - sqrt((b * b) - 4.0 * a * c)) / (2.0 * a);
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

#define NO_HIT 1e30

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

// https://pbr-book.org/4ed/Geometry_and_Transformations/Spherical_Geometry#fragment-Reparameterizedirectionsinthez0portionoftheoctahedron-0
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
vec4 merge_intervals(vec4 near, vec4 far) {
    /* Far radiance can get occluded by near visibility term */
    const vec3 radiance = near.rgb + (far.rgb * near.a);

    return vec4(radiance, near.a * far.a);
}

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

    // ------------------- above
    // above cascade has half the probes per axis
    uint above_probe_x = probe_x / 2;
    uint above_probe_y = probe_y / 2;
    uint above_probe_z = probe_z / 2;

    uint above_xy_idx = above_probe_y * (config.count_x / 2) + above_probe_x;
    uint above_xy_col = above_xy_idx % config.above_xy_cols;
    uint above_xy_row = above_xy_idx / config.above_xy_cols;
    uint above_z_col  = above_probe_z % config.above_z_cols;
    uint above_z_row  = above_probe_z / config.above_z_cols;

    uint above_sqrt_ray = config.sqrt_ray_count * 4;
    uint above_ray_col  = uint(float(ray_col) / float(config.sqrt_ray_count) * float(above_sqrt_ray));
    uint above_ray_row  = uint(float(ray_row) / float(config.sqrt_ray_count) * float(above_sqrt_ray));
    uint above_texel_x  = above_z_col * (config.above_xy_cols * above_sqrt_ray) + above_xy_col * above_sqrt_ray + above_ray_col;
    uint above_texel_y  = above_z_row * (config.above_xy_rows * above_sqrt_ray) + above_xy_row * above_sqrt_ray + above_ray_row;
    // -------------------  above

    vec3 direction = unitVectorFrom2d(ray_row, ray_col, config.sqrt_ray_count);

    Ray   ray     = Ray(direction, probe_world_pos);
    float closest = NO_HIT;
    vec4  color   = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < config.mesh_count; i++) {
        MeshInfo mesh_info = config.meshes[i];
        for (int j = 0; j < mesh_info.index_count; j += 3) {
            vec4 p1 = vec4(pos.positions[idx.indices[j + 0 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);
            vec4 p2 = vec4(pos.positions[idx.indices[j + 1 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);
            vec4 p3 = vec4(pos.positions[idx.indices[j + 2 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);

            vec3 worldp1 = (mesh_info.local_to_world * p1).xyz;
            vec3 worldp2 = (mesh_info.local_to_world * p2).xyz;
            vec3 worldp3 = (mesh_info.local_to_world * p3).xyz;

            float d = segmentTriangleIntersect(ray, worldp1, worldp2, worldp3, config.interval_start, config.interval_end);
            if (d < closest) {
                closest = d;
                color   = vec4(0.0, 0.0, 0.0, 0.0); // opaque, no emission
            }
        }
    }
    for (int i = 0; i < lights.point_light_count; i++) {
        float d = segmentSphereIntersect(
            ray, lights.point_light_positions[i].xyz, lights.point_light_positions[i].w, config.interval_start, config.interval_end);
        if (d < closest) {
            closest = d;
            color   = vec4(lights.point_light_colors[i].xyz, 0.0);
        }
    }
    vec4 above_radiance;

    if (config.is_top_cascade == 0) {
        above_radiance = imageLoad(above_radiance_field, ivec2(above_texel_x, above_texel_y)).rgba;
    } else {
        float sun_dot        = dot(direction, lights.sun_direction.xyz);
        vec3  sky_color      = vec3(0.2, 0.4, 0.8) * max(0.0, direction.y);
        vec3  incoming_color = sky_color;
        if (sun_dot > 0.99) {
            incoming_color = lights.sun_color.xyz;
        }
        above_radiance = vec4(incoming_color, 1.0);
    }
    vec4 merged = merge_intervals(color, above_radiance);

    imageStore(radiance_field, ivec2(texel_x, texel_y), merged);
}
