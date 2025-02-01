use vulkano::pipeline::graphics::vertex_input;

#[derive(vulkano::buffer::BufferContents, vertex_input::Vertex)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
pub struct TransformationUBO {
    pub model: ultraviolet::Mat4,
    pub view: ultraviolet::Mat4,
    pub proj: ultraviolet::Mat4,
}
//#[derive(Pod, Zeroable, Copy, Debug, Clone)]
//#[repr(C)]
//struct AmbientLightUBO {
//    color: [f32; 3],
//    intensity: f32,
//}
//#[derive(Pod, Zeroable, Copy, Debug, Clone)]
//#[repr(C)]
//struct DirectionalLightUBO {
//    position: [f32; 3],
//    color: [f32; 3],
//}
