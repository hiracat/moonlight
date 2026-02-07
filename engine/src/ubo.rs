#![allow(dead_code)]
#![allow(unreachable_patterns)]

use bytemuck as bm;
use ultraviolet as uv;

use crate::components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform};

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct ModelUBO {
    pub(crate) model: uv::Mat4,
    normal: uv::Mat4,
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
            model: model,
            normal: model.inversed().transposed(),
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
            view: view,
            proj: uv::projection::perspective_vk(
                camera.fov_rads,
                camera.aspect_ratio,
                camera.near,
                camera.far,
            ),
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

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub(crate) position: uv::Vec4,
    pub(crate) color: uv::Vec3,
    _padding: f32,
}

impl From<&DirectionalLight> for DirectionalLightUBO {
    fn from(light: &DirectionalLight) -> Self {
        DirectionalLightUBO {
            position: light.from_position,
            color: light.color,
            _padding: 0.0,
        }
    }
}
