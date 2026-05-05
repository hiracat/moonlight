#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba8, binding = 0) readonly uniform image2D inputImage;
layout(rgba8, binding = 1) writeonly uniform image2D outputImage;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = imageSize(inputImage);

    if (coord.x >= size.x || coord.y >= size.y) {
        return;
    }

    vec4 color = imageLoad(inputImage, coord);
    imageStore(outputImage, coord, vec4(1.0 - color.rgb, color.a));
}
