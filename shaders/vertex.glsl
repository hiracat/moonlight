#version 450
layout(set = 0, binding = 0) uniform TransformationUBO {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 frag_pos;

void main() {
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);
    out_color = color;
    out_normal = mat3(ubo.model) * normal;
    frag_pos = vec3(ubo.model * vec4(position, 1.0));
}
