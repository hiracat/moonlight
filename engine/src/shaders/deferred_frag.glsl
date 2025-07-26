#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_position;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_position;


void main() {
    f_color = vec4(in_color, 1.0);
    f_normal = in_normal;
    f_position = in_position;
    // f_color = vec4(0.2, 0.3, 0.5, 1.0);
}
