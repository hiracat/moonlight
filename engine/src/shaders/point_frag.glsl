#version 450
// can use the same set number since this is a different pipeline that uses a different pipeline layout
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_position;
// this needs a different set since it is rebound per object, since it is in a different grahpics pipeline it still uses set index one because different pipeline layout
layout(set = 1, binding = 0) uniform PointLight {
    vec3 position;
    float _padding;
    vec3 color;
    float brightness;
    float linear;      // Controls linear distance falloff
    float quadratic;   // Controls quadratic distance falloff
} point;
layout(location = 0) out vec4 f_color;
// Simple Reinhard tonemapping function
vec3 toneMap(vec3 color, float exposure) {
    // Apply exposure adjustment
    color *= exposure;

    // Apply Reinhard tonemapping formula
    color = color / (vec3(1.0) + color);

    // Optional gamma correction
    color = pow(color, vec3(1.0/2.2));

    return color;
}


void main() {
    vec3 fragPos = subpassLoad(u_position).xyz;
    vec3 lightDirection = normalize(point.position - fragPos);
    float distance = length(point.position - fragPos);
    float attenuation = 1.0 / (1 + point.linear * distance + point.quadratic * distance * distance);

    float diffuseIntensity = max(dot(normalize(subpassLoad(u_normals).rgb), lightDirection), 0.0);
    vec3 diffuseColor = diffuseIntensity * point.color;

    diffuseColor *= attenuation * point.brightness;

    vec3 combinedColor = diffuseColor * subpassLoad(u_color).rgb;
    f_color = vec4(combinedColor, 1);
    // f_color = vec4(toneMap(combinedColor, 1.0), 1.0); // Adjust exposure value as needed
}
