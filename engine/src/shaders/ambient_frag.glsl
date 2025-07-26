#version 450
// can use the same set number since this is a different pipeline that uses a different pipeline layout
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_position;
// these can also stay since they are static but may change from frame to frame, same set as input attachments
layout(set = 1, binding = 0) uniform AmbientLightUBO{
    vec3 color;
    float intensity;
} ambient;
layout(set = 1, binding = 1) uniform DirectionalLight {
    vec4 position;
    vec3 color;
} directional;
// this needs a different set since it is rebound per object, since it is in a different grahpics pipeline it still uses set index one because different pipeline layout
layout(set = 2, binding = 0) uniform PointLight {
    vec3 position;
    float _padding;
    vec3 color;
    float brightness;
    float linear;      // Controls linear distance falloff
    float quadratic;   // Controls quadratic distance falloff
} point;
layout(location = 0) out vec4 f_color;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 combined_color = ambient_color * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
    // f_color = vec4(subpassLoad(u_normals).rgb, 1.0);
    // f_color = vec4(1.0, 0.2, 2.6, 1.0);

}
