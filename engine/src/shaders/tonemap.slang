#language slang 2026
struct FragmentInput {
    float2 uv : UV;
};

struct FragmentOutput {
    float4 color : SV_Target;
};

Sampler2D hdr_color;
Sampler2D other_texture3;

[shader("fragment")]
FragmentOutput fragmentMain(FragmentInput input) {
    float4 color = hdr_color.Sample(input.uv);
    // Reinhard tonemap, TODO: use a less  crappy tonemapping algo like aces
    float3 low_dynamic_range = color.xyz / (color.xyz + float3(1.0, 1.0, 1.0));

    FragmentOutput output;
    output.color = float4(low_dynamic_range, 1.0);
    return output;
}

struct VertexInput {
    uint vertex_index : SV_VertexID;
};
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : UV;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input) {
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };
    float2 position = positions[input.vertex_index];
    VertexOutput output;
    output.uv = position * 0.5 + 0.5;
    output.position = float4(position, 0.0, 1.0);
    return output;
}
