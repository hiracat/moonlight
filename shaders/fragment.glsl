#version 450
layout(location = 0) in vec3 in_color;
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 1) uniform AmbientLightUBO {
    vec3 color;
    float intensity;
} ambient;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 combined_color = ambient_color * in_color;
    f_color = vec4(combined_color, 1.0);
}
