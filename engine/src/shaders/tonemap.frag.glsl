#version 450
layout(set = 0, binding = 0) uniform sampler2D hdr_color;
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 f_color;

void main() {
    vec3 hdr = texture(hdr_color, in_uv).rgb;
    // Reinhard tonemap
    vec3 ldr = hdr / (hdr + vec3(1.0));
    f_color = vec4(ldr, 1.0);
}
