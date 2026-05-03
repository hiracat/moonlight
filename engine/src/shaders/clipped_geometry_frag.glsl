#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_position;
layout(location = 3) in vec2 in_tex_coord;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_position;

layout(set = 2, binding = 0) uniform sampler2D tex1;
layout(set = 2, binding = 1) uniform MaterialUBO {
    float clip_cutoff;
}
material;

void main() {
    vec4 sampled = texture(tex1, in_tex_coord);

    if (sampled.a < material.clip_cutoff) {
        discard;
    }

    f_color    = sampled;
    f_normal   = in_normal;
    f_position = in_position;
}
