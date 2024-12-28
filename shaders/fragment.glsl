#version 450
layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 frag_pos;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform AmbientLightUBO{
    vec3 color;
    float intensity;
} ambient;

layout(set = 0, binding = 2) uniform DirectionalLightUBO{
    vec3 position;
    vec3 color;
} directional;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 light_direction = normalize(directional.position - frag_pos);
    float directional_intensity = max(dot(in_normal, light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = (ambient_color + directional_color) * in_color;
    f_color = vec4(combined_color, 1.0);
}
