#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 inverse_view;
    mat4 inverse_proj;
} camera;
layout(set = 1, binding = 0) uniform samplerCube skybox;

// this shader only runs for depth values which are equal to the max depth
// as a direction to point into the cubemap
layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 finalColor;

void main() {
    // Convert from screen space (0,0 to 1,1) to NDC (-1,-1 to 1,1)
    vec4 clip_space = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    // Unproject from clip space to view space (still in homogeneous coords which are coordinates with a w component that is not 1,
    //projected into the camera cube, with w encoding the amount of projection)
    vec4 view_homogeneous = camera.inverse_proj * clip_space;
    // Perspective division: convert from homogeneous to Cartesian coordinates relative to the camera(in camera space)
    vec4 view_space = view_homogeneous / view_homogeneous.w;
    // gives us direction in world space by undooing the view matrix( go from camera space to world space)
    // only using the mat3 discards translation info, leaving behind only rotation
    vec3 world_dir = mat3(camera.inverse_view) * normalize(view_space.xyz);

    finalColor = vec4(texture(skybox, world_dir).rgb, 1.0);
}
