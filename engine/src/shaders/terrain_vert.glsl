#version 450

layout(set = 0, binding = 0) uniform TerrainData {
    float size;        // world size of terrain
    float height;      // max height scale
    int resolution;    // how many vertices per side e.g. 256
} terrain;

layout(set = 1, binding = 0) uniform ViewProjUBO{
    mat4 view;
    mat4 projection;
} vp_uniforms;
layout(set = 2, binding = 0) uniform sampler2D heightmap;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_position;
layout(location = 3) out vec2 out_uv;



void main() {
    // the index of which quad in all the quads
    int quad_index = gl_VertexIndex / 6;
    // the vertex within that quad
    int vert_index = gl_VertexIndex % 6;

    // which quad in the grid (row, col)
    int col = quad_index % (terrain.resolution - 1);
    int row = quad_index / (terrain.resolution - 1);

    // vertex index 0 has no offset, 1 does, 2 doesnt, 3 ...
    int offsets_x[6] = int[](0, 1, 0, 1, 0, 1);
    // then this goes through all the combinations to make a proper quad
    int offsets_y[6] = int[](0, 0, 1, 0, 1, 1);

    // the x coordinate, + the row/col
    int x = col + offsets_x[vert_index];
    int y = row + offsets_y[vert_index];

    // one massive texture for the whole terrain, might need to make this tile later, just add an extra modulo
    vec2 uv = vec2(x, y) / float(terrain.resolution - 1);
    out_uv = uv;

    // sample height from red channel
    float height = texture(heightmap, uv).r * terrain.height;

    // convert to world position
    // center terrain, * by world size
    vec2 xz = (uv - 0.5) * terrain.size;
    // convert to world coordinates
    vec4 world_pos = vec4(xz.x, height, xz.y, 1.0);

    float texel = 1.0 / float(terrain.resolution - 1);
    float hL = texture(heightmap, uv + vec2(-texel, 0.0)).r * terrain.height;
    float hR = texture(heightmap, uv + vec2( texel, 0.0)).r * terrain.height;
    float hD = texture(heightmap, uv + vec2(0.0, -texel)).r * terrain.height;
    float hU = texture(heightmap, uv + vec2(0.0,  texel)).r * terrain.height;

    float world_texel = terrain.size / float(terrain.resolution - 1);
    out_normal = normalize(vec3(hL - hR, 2.0 * world_texel, hD - hU));
    out_uv = uv;
    out_position = world_pos;



    float t = texture(heightmap, uv).r; // 0..1 normalized height
    float slope = out_normal.y;         // 1=flat, 0=vertical cliff

    // Biome colours
    vec3 deep_water    = vec3(0.04, 0.12, 0.28);
    vec3 shallow_water = vec3(0.10, 0.28, 0.48);
    vec3 sand          = vec3(0.72, 0.65, 0.44);
    vec3 grass_lo      = vec3(0.22, 0.48, 0.14);
    vec3 grass_hi      = vec3(0.16, 0.36, 0.10);
    vec3 rock          = vec3(0.40, 0.36, 0.30);
    vec3 rock_dark     = vec3(0.28, 0.25, 0.20);
    vec3 snow          = vec3(0.88, 0.90, 0.94);

    vec3 color;
    if (t < 0.28) {
        color = mix(deep_water, shallow_water, smoothstep(0.0, 0.22, t));
        color = mix(color, sand,               smoothstep(0.22, 0.28, t));
    } else if (t < 0.50) {
        color = mix(sand,     grass_lo,        smoothstep(0.28, 0.36, t));
        color = mix(color,    grass_hi,        smoothstep(0.40, 0.50, t));
    } else if (t < 0.72) {
        color = mix(grass_hi, rock,            smoothstep(0.50, 0.62, t));
        color = mix(color,    rock_dark,       smoothstep(0.62, 0.72, t));
    } else {
        color = mix(rock_dark, snow,           smoothstep(0.72, 0.88, t));
    }

    // Steep slopes override with bare rock regardless of elevation
    color = mix(rock_dark, color, smoothstep(0.45, 0.80, slope));

    out_color = color;

    // camera project
    gl_Position = vp_uniforms.projection * vp_uniforms.view * world_pos;
}
