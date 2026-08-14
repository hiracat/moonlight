#![allow(dead_code)]

use glam::{self as gl, Vec3};
use proc_macros::LuaRef;

#[derive(LuaRef, Debug, Clone, Copy, Default)]
pub struct Time {
    pub delta_time: f32,
}

#[derive(LuaRef, Debug, Clone, Copy)]
pub struct Transform {
    pub position: gl::Vec3,
    pub rotation: gl::Quat,
    pub scale: gl::Vec3,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Default::default(),
            rotation: Default::default(),
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new() -> Self {
        Self {
            rotation: gl::Quat::IDENTITY,
            scale: gl::Vec3::ONE,
            position: gl::Vec3::ZERO,
        }
    }

    pub fn from(
        position: Option<gl::Vec3>,
        rotation: Option<gl::Quat>,
        scale: Option<gl::Vec3>,
    ) -> Self {
        Self {
            rotation: rotation.unwrap_or(gl::Quat::IDENTITY),
            scale: scale.unwrap_or(gl::Vec3::ONE),
            position: position.unwrap_or(gl::Vec3::ZERO),
        }
    }
}

#[derive(LuaRef, Debug, Clone, Copy, Default)]
pub struct Camera {
    pub position: gl::Vec3,
    pub rotation: gl::Quat,
    pub fov_rads: f32,
    pub near: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn create(position: gl::Vec3, fov: f32, near: f32, aspect_ratio: f32) -> Self {
        let fov_rads = fov * (std::f32::consts::PI / 180.0);
        let rotation = gl::Quat::IDENTITY;
        Camera {
            pitch: 0.0,
            yaw: 0.0,
            position,
            rotation,
            fov_rads,
            near,
            aspect_ratio,
        }
    }
}

#[derive(LuaRef, Debug, Clone, Copy, Default)]
pub struct AmbientLight {
    pub color: gl::Vec3,
    pub intensity: f32,
}

impl AmbientLight {
    pub fn create(color: gl::Vec3, intensity: f32) -> Self {
        Self { color, intensity }
    }
}

#[derive(LuaRef, Debug, Clone, Copy, Default)]
pub struct PointLight {
    pub color: gl::Vec3,
    pub size: f32,
}

impl PointLight {
    pub fn new(color: gl::Vec3, size: f32) -> Self {
        Self { color, size }
    }
}

#[derive(LuaRef, Debug, Clone, Copy, Default)]
pub struct DirectionalLight {
    pub sun_position: gl::Vec3,
    pub sun_color: gl::Vec3,
    pub sun_size: f32,

    pub sky_zenith_color: gl::Vec3,
    pub sky_horizon_color: gl::Vec3,
    pub sky_gradient_sharpness: f32,
}

impl DirectionalLight {
    pub fn create(
        sun_pos: gl::Vec3,
        sun_color: gl::Vec3,
        sky_zenith_color: gl::Vec3,
        sky_horizon_color: gl::Vec3,
        sky_gradient_sharpness: f32,
        sun_size: f32,
    ) -> Self {
        Self {
            sun_position: sun_pos.normalize(),
            sun_color,
            sky_zenith_color,
            sky_horizon_color,
            sky_gradient_sharpness,
            sun_size,
        }
    }
}
