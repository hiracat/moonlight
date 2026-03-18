#version 450
// can use the same set number since this is a different pipeline that uses a different pipeline layout
// had to replace subpassinput with sampler2d for dynamic rendering
layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_normals;
layout(set = 0, binding = 2) uniform sampler2D u_position;
// these can also stay since they are static but may change from frame to frame, same set as input attachments
layout(set = 1, binding = 0) uniform AmbientLightUBO{
    vec3 color;
    float intensity;
} ambient;
layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 combined_color = ambient_color * texture(u_color, in_uv).rgb;
    f_color = vec4(combined_color, 1.0);
    // f_color = vec4(subpassLoad(u_normals).rgb, 1.0);
    // f_color = vec4(1.0, 0.2, 2.6, 1.0);

}
