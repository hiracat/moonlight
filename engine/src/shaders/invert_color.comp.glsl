#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgb10_a2, set = 0, binding = 0) uniform image2D image;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = imageSize(image);

    if (coord.x >= size.x || coord.y >= size.y) {
        return;
    }

    vec4 color = imageLoad(image, coord);
    imageStore(image, coord, vec4(1.0 - color.rgb, color.a));
}
