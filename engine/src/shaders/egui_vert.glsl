#version 450

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} push_constants;


void main() {
    vec2 ndc_pos;
    ndc_pos.x = (a_pos.x / push_constants.screen_size.x) * 2.0 - 1.0;
    ndc_pos.y = (a_pos.y / push_constants.screen_size.y) * 2.0 - 1.0;

    gl_Position = vec4(ndc_pos, 0.0, 1.0);
    v_uv = a_uv;
    v_color = a_color;
}
