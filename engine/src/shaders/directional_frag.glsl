#version 450
// can use the same set number since this is a different pipeline that uses a different pipeline layout
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_position;
layout(set = 1, binding = 0) uniform DirectionalLight {
    vec4 position;
    vec3 color;
} directional;
layout(location = 0) out vec4 f_color;

void main() {
    vec3 light_direction = normalize(directional.position.xyz - subpassLoad(u_normals).xyz);

    float directional_intensity = max(dot(normalize(subpassLoad(u_normals).rgb), light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = directional_color * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
    // f_color = vec3(gl_FragDepth, 0,0);
    // f_color = vec4(1.0, 0.2, 2.6, 1.0);
}
