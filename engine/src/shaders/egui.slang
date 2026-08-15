#language slang 2026

struct VertexInput {
    float2 position : POSITION;
    float2 uv : UV;
    uint8_t4 color : COLOR;
}

struct VertexOutput {
    uint8_t4 color : COLOR;
    float2 uv : UV;
    float4 position : SV_Position;
}

struct PushConstants {
    float2 screen_size : SCREEN_SIZE;
}

[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uniform PushConstants pc) {
    float2 ndc_pos;
    ndc_pos.x = (input.position.x / pc.screen_size.x) * 2.0 - 1.0;
    ndc_pos.y = (input.position.y / pc.screen_size.y) * 2.0 - 1.0;

    float4 position = float4(ndc_pos, 0.0, 1.0);

    VertexOutput output;
    output.position = position;
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

struct FragmentInput {
    uint8_t4 color : COLOR;
    float2 uv : UV;
}

struct FragmentOutput {
    float4 color : SV_Target;
}
Sampler2D texture;

[shader("fragment")]
FragmentOutput fragmentMain(FragmentInput input) {
    float4 tex_color = texture.Sample(input.uv);
    FragmentOutput output;
    output.color = tex_color * input.color;
    return output;
}
