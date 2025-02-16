#version 450

layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 2) uniform subpassInput u_normals;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(subpassLoad(u_normals).rgb, 1.0);
}
