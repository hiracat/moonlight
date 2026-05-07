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
#define EPS .00002

struct Ray {
    vec3 direction;
    vec3 origin;
};

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

bool segmentTriangleIntersect(Ray ray, vec3 A, vec3 B, vec3 C, float start, float end) {
    float distance = rayTriangleIntersect(ray, A, B, C);
    if (distance > start && distance < end) {
        return true;
    }
    return false;
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

    vec3 direction = unitVectorFrom2d(ray_row, ray_col, config.sqrt_ray_count);

    Ray  ray       = Ray(direction, probe_world_pos);
    bool collision = false;
    vec3 color;
    for (int i = 0; i < config.mesh_count; i++) {
        MeshInfo mesh_info = config.meshes[i];
        for (int j = 0; j < mesh_info.index_count; j += 3) {
            vec4 p1 = vec4(pos.positions[idx.indices[j + 0 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);
            vec4 p2 = vec4(pos.positions[idx.indices[j + 1 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);
            vec4 p3 = vec4(pos.positions[idx.indices[j + 2 + mesh_info.index_offset] + mesh_info.vertex_offset], 1.0);

            vec3 worldp1 = (mesh_info.local_to_world * p1).xyz;
            vec3 worldp2 = (mesh_info.local_to_world * p2).xyz;
            vec3 worldp3 = (mesh_info.local_to_world * p3).xyz;

            collision = segmentTriangleIntersect(ray, worldp1, worldp2, worldp3, config.interval_start, config.interval_end);

            if (collision) {
                break;
            }
        }
        if (collision) {
            break;
        }
    }
    if (collision) {
        color = vec3(0.0, 0.0, 0.0);
    } else {
        color = vec3(1.0, 1.0, 1.0);
    }
    imageStore(radiance_field, ivec2(texel_x, texel_y), vec4(color, 1.0));
}
