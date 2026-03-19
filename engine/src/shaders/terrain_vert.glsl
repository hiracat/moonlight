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

    out_normal = normalize(vec3(hL - hR, 2.0, hD - hU));
    out_uv = uv;
    out_position = world_pos;



    float real_height = (texture(heightmap, uv).r - .5) * 2;
    float red = real_height * 0.4;
    float green = real_height;
    float blue = -real_height;

    out_color = vec3(red, green, blue);

    // camera project
    gl_Position = vp_uniforms.projection * vp_uniforms.view * world_pos;
}
