pub mod defered_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/defered_vert.glsl",
    }
}
pub mod defered_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/defered_frag.glsl"
    }
}
pub mod lighting_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/directional_vert.glsl"
    }
}
pub mod lighting_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/directional_frag.glsl"
    }
}
pub mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/ambient_vert.glsl",
    }
}
pub mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ambient_frag.glsl",
    }
}

pub mod point_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/point_vert.glsl",
    }
}
pub mod point_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/point_frag.glsl"
    }
}
