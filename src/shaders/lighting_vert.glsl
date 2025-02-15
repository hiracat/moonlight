#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform TransformationUBO{
    mat4 model;
    mat4 view;
    mat4 projection;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.model;
    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
}
