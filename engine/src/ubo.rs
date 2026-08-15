#![allow(dead_code)]
#![allow(unreachable_patterns)]

use bytemuck as bm;
use glam as gl;

use crate::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    core::TerrainMap,
    resources::Material,
};

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct ModelUBO {
    pub model: gl::Mat4,
    pub normal: gl::Mat4,
}

impl ModelUBO {
    pub(crate) fn new(position: gl::Vec3, rotation: gl::Quat) -> ModelUBO {
        let rotation_mat = gl::Mat4::from_quat(rotation);
        let translation_mat = gl::Mat4::from_translation(position);
        let model_mat = translation_mat * rotation_mat;

        ModelUBO {
            model: model_mat,
            normal: model_mat.inverse().transpose(),
        }
    }
}

impl From<&Transform> for ModelUBO {
    fn from(transform: &Transform) -> Self {
        let rotation = gl::Mat4::from_quat(transform.rotation);
        let scale = gl::Mat4::from_scale(transform.scale);
        let position = gl::Mat4::from_translation(transform.position);
        let model = position * rotation * scale;

        ModelUBO {
            model,
            normal: model.inverse().transpose(),
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
    pub aabb_local_min: gl::Vec4,
    pub aabb_local_max: gl::Vec4,
    pub local_to_world: gl::Mat4,
    pub world_to_local: gl::Mat4,
}
#[derive(Debug, Clone, Copy, bm::Pod, bm::Zeroable)]
#[repr(C)]
pub struct RadianceLevelConfigUBO {
    pub grid_size: gl::UVec3,
    pub grid_gap: f32,
    /// the grid origin, varies between each level because how how each probe level is offset from
    /// the previous
    pub grid_origin: gl::Vec3,
    pub sqrt_ray_count: u32,
    pub interval_start: f32,
    pub interval_end: f32,
    pub is_top_cascade: u32, // zero for false, anything else for true
    pub _pad0: u32,
}

#[derive(Debug, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct ComputeRadianceUBO {
    pub level_config: RadianceLevelConfigUBO,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,

    pub mesh_count: u32,
    // offset 64 here, meshes is naturally aligned, no padding needed
    pub meshes: [MeshInfo; 64],
}

#[derive(Debug, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct LightDataUBO {
    pub sky_light: DirectionalLightUBO,
    pub point_light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub point_light_positions: [gl::Vec4; 32], // xyz = pos, w = radius
    pub point_light_colors: [gl::Vec4; 32],    // xyz = color (intensity baked in), w = unused
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
    view: gl::Mat4,
    proj: gl::Mat4,
}

impl From<&Camera> for CameraUBO {
    fn from(camera: &Camera) -> Self {
        let rotation_matrix = gl::Mat4::from_quat(camera.rotation.inverse());
        let translation_matrix = gl::Mat4::from_translation(-camera.position);
        let view = rotation_matrix * translation_matrix;

        CameraUBO {
            view,
            proj: gl::camera::rh::proj::vulkan::perspective_infinite_reverse(
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
    view: gl::Mat4,
    proj: gl::Mat4,
    inverse_view: gl::Mat4,
    inverse_proj: gl::Mat4,
}

impl From<&Camera> for CameraInverseUBO {
    fn from(camera: &Camera) -> Self {
        let rotation_matrix = gl::Mat4::from_quat(camera.rotation.inverse());
        let translation_matrix = gl::Mat4::from_translation(-camera.position);
        let view = rotation_matrix * translation_matrix;
        let proj = gl::camera::rh::proj::vulkan::perspective_infinite_reverse(
            camera.fov_rads,
            camera.aspect_ratio,
            camera.near,
        );

        CameraInverseUBO {
            view,
            proj,
            inverse_view: view.inverse(),
            inverse_proj: proj.inverse(),
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct AmbientLightUBO {
    pub(crate) color: gl::Vec3,
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
    color: gl::Vec4, // w = size
    position: gl::Vec4,
}

impl PointLightUBO {
    pub(crate) fn new() -> Self {
        PointLightUBO {
            position: gl::Vec4::ZERO,
            color: gl::Vec4::ZERO,
        }
    }
}

impl From<(&PointLight, &Transform)> for PointLightUBO {
    fn from((light, transform): (&PointLight, &Transform)) -> Self {
        PointLightUBO {
            position: transform.position.extend(1.0),
            color: gl::Vec4::from_array([light.color.x, light.color.y, light.color.z, light.size]),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub sun_position: gl::Vec4,
    pub sun_color: gl::Vec4, // w = size, between 0 and 1
    pub sky_zenith_color: gl::Vec4,
    pub sky_horizon_color: gl::Vec4,
    pub sky_gradient_sharpness: f32,
    pub _pad: [u32; 3],
}

impl From<&DirectionalLight> for DirectionalLightUBO {
    fn from(light: &DirectionalLight) -> Self {
        DirectionalLightUBO {
            sun_position: light.sun_position.normalize().extend(1.0),
            sun_color: gl::Vec4::new(
                light.sun_color.x,
                light.sun_color.y,
                light.sun_color.z,
                light.sun_size,
            ),
            sky_zenith_color: light.sky_zenith_color.extend(1.0),
            sky_horizon_color: light.sky_horizon_color.extend(1.0),
            sky_gradient_sharpness: light.sky_gradient_sharpness,
            _pad: [0; 3],
        }
    }
}
