#![allow(dead_code)]
#![allow(unreachable_patterns)]

use bytemuck as bm;
use ultraviolet as uv;

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: uv::Vec3,
    pub rotation: uv::Rotor3, // rotation from -unit_z
    pub scale: uv::Vec3,
}

impl Transform {
    pub fn as_model_ubo(&self) -> ModelUBO {
        let rotation = self.rotation.into_matrix().into_homogeneous();
        let scale = uv::Mat4::from_nonuniform_scale(self.scale);
        let position = uv::Mat4::from_translation(self.position);
        let model = position * rotation * scale;

        ModelUBO {
            model: model,
            normal: model.inversed().transposed(),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    pub position: uv::Vec3,
    pub rotation: uv::Rotor3,
    pub fov_rads: f32,
    pub near: f32,
    pub far: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub(crate) aspect_ratio: f32,
}
#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct CameraUBO {
    view: uv::Mat4,
    proj: uv::Mat4,
}
#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct CameraInverseUBO {
    view: uv::Mat4,
    proj: uv::Mat4,
    inverse_view: uv::Mat4,
    inverse_proj: uv::Mat4,
}

pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl AmbientLight {
    pub fn as_ubo(&self) -> AmbientLightUBO {
        AmbientLightUBO {
            color: self.color,
            intensity: self.intensity,
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct AmbientLightUBO {
    pub(crate) color: [f32; 3],
    pub(crate) intensity: f32,
}
pub struct PointLight {
    pub color: uv::Vec3,
    pub brightness: f32,
    pub linear: f32,
    pub quadratic: f32,
    pub dirty: bool,
}

impl PointLight {
    pub fn as_ubo(&self, transform: &Transform) -> PointLightUBO {
        PointLightUBO {
            position: transform.position,
            _padding: 0.0,
            color: self.color,
            brightness: self.brightness,
            linear: self.linear,
            quadratic: self.quadratic,
        }
    }
}
pub struct DirectionalLight {
    pub from_position: [f32; 4],
    pub color: [f32; 3],
}

impl DirectionalLight {
    pub fn as_ubo(&self) -> DirectionalLightUBO {
        DirectionalLightUBO {
            position: self.from_position,
            color: self.color,
        }
    }
}

#[derive(Default, Copy, Clone, bm::Zeroable, bm::Pod)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub(crate) position: [f32; 4],
    pub(crate) color: [f32; 3],
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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bm::Zeroable, bm::Pod)]
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
impl Transform {
    pub fn new() -> Self {
        Self {
            rotation: uv::Rotor3::identity(),
            scale: uv::Vec3::one(),
            position: uv::Vec3::zero(),
        }
    }
    pub fn from(
        position: Option<uv::Vec3>,
        rotation: Option<uv::Rotor3>,
        scale: Option<uv::Vec3>,
    ) -> Self {
        Self {
            rotation: rotation.unwrap_or(uv::Rotor3::identity()),
            scale: scale.unwrap_or(uv::Vec3::one()),
            position: position.unwrap_or(uv::Vec3::zero()),
        }
    }
}

impl AmbientLight {
    pub fn create(color: [f32; 3], intensity: f32) -> Self {
        Self { color, intensity }
    }
}

impl PointLight {
    pub fn new(
        color: uv::Vec3,
        brightness: f32,
        linear: Option<f32>,
        quadratic: Option<f32>,
    ) -> Self {
        Self {
            color,
            brightness,
            linear: linear.unwrap_or(3.00),
            quadratic: quadratic.unwrap_or(0.00),
            dirty: true,
        }
    }
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

impl DirectionalLight {
    pub fn create(position: [f32; 4], color: [f32; 3]) -> Self {
        Self {
            from_position: position,
            color,
        }
    }
}

impl Camera {
    pub fn create(position: uv::Vec3, fov: f32, near: f32, far: f32, aspect_ratio: f32) -> Self {
        let fov_rads = fov * (std::f32::consts::PI / 180.0);
        let rotation = uv::Rotor3::identity();
        Camera {
            pitch: 0.0,
            yaw: 0.0,
            position,
            rotation,
            fov_rads,
            near,
            far,
            aspect_ratio,
        }
    }
    pub fn as_ubo(&self) -> CameraUBO {
        let rotation_matrix = self.rotation.reversed().into_matrix().into_homogeneous();
        let translation_matrix = uv::Mat4::from_translation(-self.position);
        let view = rotation_matrix * translation_matrix;

        CameraUBO {
            view: view,
            proj: uv::projection::perspective_vk(
                self.fov_rads,
                self.aspect_ratio,
                self.near,
                self.far,
            ),
        }
    }
    pub fn as_inverse_ubo(&self) -> CameraInverseUBO {
        let rotation_matrix = self.rotation.reversed().into_matrix().into_homogeneous();
        let translation_matrix = uv::Mat4::from_translation(-self.position);
        let view = rotation_matrix * translation_matrix;
        let proj =
            uv::projection::perspective_vk(self.fov_rads, self.aspect_ratio, self.near, self.far);

        CameraInverseUBO {
            view,
            proj,
            inverse_view: view.inversed(),
            inverse_proj: proj.inversed(),
        }
    }
}
