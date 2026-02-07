#![allow(dead_code)]

use ultraviolet as uv;

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: uv::Vec3,
    pub rotation: uv::Rotor3,
    pub scale: uv::Vec3,
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
}

pub struct AmbientLight {
    pub color: uv::Vec3,
    pub intensity: f32,
}

impl AmbientLight {
    pub fn create(color: uv::Vec3, intensity: f32) -> Self {
        Self { color, intensity }
    }
}

pub struct PointLight {
    pub color: uv::Vec3,
    pub brightness: f32,
    pub linear: f32,
    pub quadratic: f32,
    pub dirty: bool,
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

pub struct DirectionalLight {
    pub from_position: uv::Vec4,
    pub color: uv::Vec3,
}

impl DirectionalLight {
    pub fn create(position: uv::Vec4, color: uv::Vec3) -> Self {
        Self {
            from_position: position,
            color,
        }
    }
}
