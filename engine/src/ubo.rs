#![allow(dead_code)]
#![allow(unreachable_patterns)]

use bytemuck as bm;
use ultraviolet as uv;

use crate::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    core::TerrainMap,
    resources::Material,
};

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct ModelUBO {
    pub model: uv::Mat4,
    pub normal: uv::Mat4,
}

impl ModelUBO {
    pub(crate) fn new(position: uv::Vec3, rotation: uv::Rotor3) -> ModelUBO {
        let rotation_mat = rotation.into_matrix().into_homogeneous();
        let translation_mat = uv::Mat4::from_translation(position);
        let model_mat = translation_mat * rotation_mat;

        ModelUBO {
            model: model_mat,
            normal: model_mat.inversed().transposed(),
        }
    }
}

impl From<&Transform> for ModelUBO {
    fn from(transform: &Transform) -> Self {
        let rotation = transform.rotation.into_matrix().into_homogeneous();
        let scale = uv::Mat4::from_nonuniform_scale(transform.scale);
        let position = uv::Mat4::from_translation(transform.position);
        let model = position * rotation * scale;

        ModelUBO {
            model,
            normal: model.inversed().transposed(),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct MeshInfo {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub _pad: u32,
    pub aabb_local_min: uv::Vec4,
    pub aabb_local_max: uv::Vec4,
    pub local_to_world: uv::Mat4,
}

#[derive(Debug, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct RadianceConfigUBO {
    pub start_position: uv::Vec4, // 16 bytes
    pub count_x: u32,
    pub count_y: u32,
    pub count_z: u32,
    pub z_cols: u32,
    pub xy_cols: u32,
    pub xy_rows: u32,
    pub above_z_cols: u32,
    pub above_xy_cols: u32,
    pub above_xy_rows: u32,
    pub _pad: u32,
    pub probe_spacing: f32,
    pub interval_start: f32,
    pub interval_end: f32,
    pub is_top_cascade: u32,
    pub sqrt_ray_count: u32,
    pub mesh_count: u32,
    // offset 64 here, meshes is naturally aligned, no padding needed
    pub meshes: [MeshInfo; 64],
}

#[derive(Default, Debug, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct RadianceInfoUBO {
    pub start_position: uv::Vec4, // world space origin of the probe grid
    pub probe_x_count: u32,
    pub probe_y_count: u32,
    pub probe_z_count: u32,
    pub z_cols: u32,
    pub xy_cols: u32,
    pub xy_rows: u32,
    pub sqrt_ray_count: u32,
    pub probe_spacing: f32,
}

#[derive(Debug, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct LightDataUBO {
    pub sky_light: DirectionalLightUBO,
    pub point_light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub point_light_positions: [uv::Vec4; 32], // xyz = pos, w = radius
    pub point_light_colors: [uv::Vec4; 32],    // xyz = color (intensity baked in), w = unused
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct MaterialUBO {
    alpha_clip: f32,
}

impl From<&Material> for MaterialUBO {
    fn from(material: &Material) -> Self {
        Self {
            alpha_clip: material.alpha_clip.unwrap_or(0.0),
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct CameraUBO {
    view: uv::Mat4,
    proj: uv::Mat4,
}

impl From<&Camera> for CameraUBO {
    fn from(camera: &Camera) -> Self {
        let rotation_matrix = camera.rotation.reversed().into_matrix().into_homogeneous();
        let translation_matrix = uv::Mat4::from_translation(-camera.position);
        let view = rotation_matrix * translation_matrix;

        CameraUBO {
            view,
            proj: uv::projection::perspective_reversed_infinite_z_vk(
                camera.fov_rads,
                camera.aspect_ratio,
                camera.near,
            ),
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct TerrainUBO {
    size: f32,
    height: f32,
    resolution: u32,
}

impl From<&TerrainMap> for TerrainUBO {
    fn from(map: &TerrainMap) -> Self {
        Self {
            resolution: map.resolution,
            height: map.height,
            size: map.size,
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct CameraInverseUBO {
    view: uv::Mat4,
    proj: uv::Mat4,
    inverse_view: uv::Mat4,
    inverse_proj: uv::Mat4,
}

impl From<&Camera> for CameraInverseUBO {
    fn from(camera: &Camera) -> Self {
        let rotation_matrix = camera.rotation.reversed().into_matrix().into_homogeneous();
        let translation_matrix = uv::Mat4::from_translation(-camera.position);
        let view = rotation_matrix * translation_matrix;
        let proj = uv::projection::perspective_vk(
            camera.fov_rads,
            camera.aspect_ratio,
            camera.near,
            camera.far,
        );

        CameraInverseUBO {
            view,
            proj,
            inverse_view: view.inversed(),
            inverse_proj: proj.inversed(),
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct AmbientLightUBO {
    pub(crate) color: uv::Vec3,
    pub(crate) intensity: f32,
}

impl From<&AmbientLight> for AmbientLightUBO {
    fn from(light: &AmbientLight) -> Self {
        AmbientLightUBO {
            color: light.color,
            intensity: light.intensity,
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct PointLightUBO {
    pub(crate) position: uv::Vec3,
    _padding: f32,
    pub(crate) color: uv::Vec3,
    pub(crate) brightness: f32,
    pub(crate) linear: f32,
    pub(crate) quadratic: f32,
}

impl PointLightUBO {
    pub(crate) fn new() -> Self {
        PointLightUBO {
            position: uv::Vec3::zero(),
            _padding: 0.0,
            color: uv::Vec3::zero(),
            brightness: 0.0,
            linear: 0.0,
            quadratic: 0.0,
        }
    }
}

impl From<(&PointLight, &Transform)> for PointLightUBO {
    fn from((light, transform): (&PointLight, &Transform)) -> Self {
        PointLightUBO {
            position: transform.position,
            _padding: 0.0,
            color: light.color,
            brightness: light.brightness,
            linear: light.linear,
            quadratic: light.quadratic,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub sun_position: uv::Vec4,
    pub sun_color: uv::Vec4,
    pub sky_zenith_color: uv::Vec4,
    pub sky_horizon_color: uv::Vec4,
    pub sky_gradient_sharpness: f32,
    pub _pad: [u32; 3],
}

impl From<&DirectionalLight> for DirectionalLightUBO {
    fn from(light: &DirectionalLight) -> Self {
        DirectionalLightUBO {
            sun_position: light.sun_position.into_homogeneous_point(),
            sun_color: light.sun_color.into_homogeneous_point(),
            sky_zenith_color: light.sky_zenith_color.into_homogeneous_point(),
            sky_horizon_color: light.sky_horizon_color.into_homogeneous_point(),
            sky_gradient_sharpness: light.sky_gradient_sharpness,
            _pad: [0; 3],
        }
    }
}
