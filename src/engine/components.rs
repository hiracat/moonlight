use std::sync::Arc;

use ultraviolet::{Mat4, Rotor3, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::Pipeline,
};

use crate::renderer::{Renderer, Vertex, FRAMES_IN_FLIGHT};

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Rotor3, // rotation from -unit_z
    pub scale: Vec3,
    pub dirty: bool,
    matrix_cache: Mat4,
    inv_trans_cache: Mat4,
}
impl Transform {
    pub fn new() -> Self {
        Self {
            matrix_cache: Mat4::identity(),
            inv_trans_cache: Mat4::identity(),
            rotation: Rotor3::identity(),
            scale: Vec3::zero(),
            dirty: true,
            position: Vec3::zero(),
        }
    }
    pub fn from(position: Option<Vec3>, rotation: Option<Rotor3>, scale: Option<Vec3>) -> Self {
        Self {
            matrix_cache: Mat4::identity(),
            inv_trans_cache: Mat4::identity(),
            rotation: rotation.unwrap_or(Rotor3::identity()),
            scale: scale.unwrap_or(Vec3::one()),
            dirty: true,
            position: position.unwrap_or(Vec3::zero()),
        }
    }
    pub(in crate::engine) fn as_model_ubo(&mut self) -> ModelUBO {
        if self.dirty {
            let rotation_mat = self.rotation.into_matrix().into_homogeneous();
            let translation_mat = Mat4::from_translation(self.position);
            self.matrix_cache = translation_mat * rotation_mat;
            self.inv_trans_cache = self.matrix_cache.inversed().transposed();
        }

        ModelUBO {
            model: self.matrix_cache,
            normal: self.inv_trans_cache,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RigidBody {
    pub velocity: Vec3,
}
impl RigidBody {
    pub fn new() -> Self {
        Self {
            velocity: Vec3::zero(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Aabb(Aabb),
}

pub struct Dynamic;

pub struct Model {
    pub(in crate::engine) u_buffer: Vec<Subbuffer<ModelUBO>>,
    pub(in crate::engine) descriptor_set: Vec<Arc<PersistentDescriptorSet>>,
    pub(in crate::engine) vertex_buffer: Subbuffer<[Vertex]>,
    pub(in crate::engine) index_buffer: Subbuffer<[u32]>,
}
impl Model {
    pub fn create(renderer: &Renderer, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let vertex_buffer = Buffer::from_iter(
            renderer.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        let index_buffer = Buffer::from_iter(
            renderer.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let mut model_buffers: Vec<Subbuffer<ModelUBO>> = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            model_buffers.push(
                Buffer::new_sized(
                    renderer.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }

        //TODO: this is bullshit, it needs to be completely changed by im lost
        let model_layout = renderer.pipelines[0].layout().set_layouts().get(1).unwrap();

        let mut model_sets = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for i in 0..FRAMES_IN_FLIGHT {
            let model_set = PersistentDescriptorSet::new(
                &renderer.descriptor_set_allocator,
                model_layout.clone(),
                [WriteDescriptorSet::buffer(0, model_buffers[i].clone())],
                [],
            )
            .unwrap();
            model_sets.push(model_set);
        }
        Model {
            u_buffer: model_buffers,
            descriptor_set: model_sets,
            vertex_buffer,
            index_buffer,
        }
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct ModelUBO {
    model: Mat4,
    normal: Mat4,
}
impl ModelUBO {
    pub(in crate::engine) fn new(position: Vec3, rotation: Rotor3) -> ModelUBO {
        let rotation_mat = rotation.into_matrix().into_homogeneous();
        let translation_mat = Mat4::from_translation(position);
        let model_mat = translation_mat * rotation_mat;

        ModelUBO {
            model: model_mat,
            normal: model_mat.inversed().transposed(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}
impl Aabb {
    pub fn new(half_extent: Vec3, position: Vec3) -> Aabb {
        let global_max = position + half_extent;
        let global_min = position - half_extent;

        Aabb {
            min: global_min,
            max: global_max,
        }
    }
}

impl Collider {
    pub fn penetration_vector(
        from: &Collider,
        to: &Collider,
        from_tr: &Transform,
        to_tr: &Transform,
    ) -> Vec3 {
        match (from, to) {
            (Collider::Aabb(a), Collider::Aabb(b)) => {
                let a_max_world = a.max + from_tr.position;
                let a_min_world = a.min + from_tr.position;

                let b_max_world = b.max + to_tr.position;
                let b_min_world = b.min + to_tr.position;

                let dx1 = a_max_world.x - b_min_world.x;
                let dx2 = a_min_world.x - b_max_world.x;

                let dy1 = a_max_world.y - b_min_world.y;
                let dy2 = a_min_world.y - b_max_world.y;

                let dz1 = a_max_world.z - b_min_world.z;
                let dz2 = a_min_world.z - b_max_world.z;

                let dx = dx1.close_to_zero(dx2);
                let dy = dy1.close_to_zero(dy2);
                let dz = dz1.close_to_zero(dz2);

                Vec3 {
                    x: dx,
                    y: dy,
                    z: dz,
                }
            }
        }
    }
    pub fn intersects(a: &Collider, b: &Collider, a_tr: &Transform, b_tr: &Transform) -> bool {
        match (a, b) {
            (Collider::Aabb(a), Collider::Aabb(b)) => {
                let a_max_world = a.max + a_tr.position;
                let a_min_world = a.min + a_tr.position;

                let b_max_world = b.max + b_tr.position;
                let b_min_world = b.min + b_tr.position;

                !(a_max_world.x < b_min_world.x
                    || a_min_world.x > b_max_world.x
                    || a_max_world.y < b_min_world.y
                    || a_min_world.y > b_max_world.y
                    || a_max_world.z < b_min_world.z
                    || a_min_world.z > b_max_world.z)
            }
        }
    }
}

trait CloseToZero {
    fn close_to_zero(self, other: Self) -> Self;
}

impl CloseToZero for f32 {
    fn close_to_zero(self, other: f32) -> f32 {
        if self.abs() < other.abs() {
            self
        } else {
            other
        }
    }
}

#[derive(Default)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,

    pub(in crate::engine) u_buffer: Option<Vec<Subbuffer<AmbientLightUBO>>>,
    pub(in crate::engine) descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct AmbientLightUBO {
    pub(in crate::engine) color: [f32; 3],
    pub(in crate::engine) intensity: f32,
}
impl AmbientLight {
    pub fn new(color: [f32; 3], intensity: f32) -> Self {
        Self {
            color,
            intensity,
            u_buffer: None,
            descriptor_set: None,
        }
    }
}

pub struct PointLight {
    pub color: Vec3,
    pub brightness: f32,
    pub linear: f32,
    pub quadratic: f32,
    pub dirty: bool,

    pub(in crate::engine) u_buffers: Vec<Subbuffer<PointLightUBO>>,
    pub(in crate::engine) descriptor_set: Vec<Arc<PersistentDescriptorSet>>,
}
impl PointLight {
    pub fn create(
        renderer: &Renderer,
        color: Vec3,
        brightness: f32,
        linear: Option<f32>,
        quadratic: Option<f32>,
    ) -> Self {
        let mut u_buffers: Vec<Subbuffer<PointLightUBO>> = vec![];
        for _ in 0..renderer.swapchain.image_count() {
            u_buffers.push(
                Buffer::new_sized(
                    renderer.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
        //TODO: this is even more bullshit, ill get to it eventually
        let point_layout = renderer.pipelines[3]
            .layout()
            .set_layouts()
            .first()
            .unwrap();

        let mut descriptor_set = vec![];
        for i in 0..renderer.swapchain.image_count() as usize {
            let dir_set = PersistentDescriptorSet::new(
                &renderer.descriptor_set_allocator,
                point_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, renderer.color_buffers[i].clone()),
                    WriteDescriptorSet::image_view(1, renderer.normal_buffers[i].clone()),
                    WriteDescriptorSet::image_view(2, renderer.position_buffers[i].clone()),
                    WriteDescriptorSet::buffer(3, u_buffers[i].clone()),
                ],
                [],
            )
            .unwrap();
            descriptor_set.push(dir_set);
        }
        PointLight {
            color,
            brightness,
            linear: linear.unwrap_or(3.00),
            quadratic: quadratic.unwrap_or(0.00),
            dirty: true,
            u_buffers,
            descriptor_set,
        }
    }

    pub(in crate::engine) fn as_point_ubo(&self, transform: Transform) -> PointLightUBO {
        PointLightUBO {
            position: transform.position,
            color: self.color,
            brightness: self.brightness,
            linear: self.linear,
            quadratic: self.quadratic,
            _padding: 0.0,
        }
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub(in crate::engine) struct PointLightUBO {
    pub(in crate::engine) position: Vec3,
    _padding: f32,
    pub(in crate::engine) color: Vec3,
    pub(in crate::engine) brightness: f32,
    pub(in crate::engine) linear: f32,
    pub(in crate::engine) quadratic: f32,
}

pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],

    pub(in crate::engine) u_buffer: Option<Vec<Subbuffer<DirectionalLightUBO>>>,
    pub(in crate::engine) descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub(in crate::engine) position: [f32; 4],
    pub(in crate::engine) color: [f32; 3],
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> Self {
        Self {
            position,
            color,
            descriptor_set: None,
            u_buffer: None,
        }
    }
}
