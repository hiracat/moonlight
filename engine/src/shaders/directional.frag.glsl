#version 450
// can use the same set number since this is a different pipeline that uses a different pipeline layout
// had to replace subpassinput with sampler2d for dynamic rendering
layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_normals;
layout(set = 0, binding = 2) uniform sampler2D u_position;
layout(set = 1, binding = 0) uniform DirectionalLight {
    vec4 position;
    vec3 color;
} directional;
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 f_color;

void main() {
    vec3 light_direction = normalize(directional.position.xyz - texture(u_normals, in_uv).xyz);

    float directional_intensity = max(dot(normalize(texture(u_normals, in_uv).rgb), light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = directional_color * texture(u_color, in_uv).rgb;
    f_color = vec4(combined_color, 1.0);
    // f_color = vec3(gl_FragDepth, 0,0);
    // f_color = vec4(1.0, 0.2, 2.6, 1.0);
}
