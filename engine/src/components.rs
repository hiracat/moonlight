#![allow(dead_code)]
#![allow(unreachable_patterns)]
use core::fmt;
use std::sync::Arc;

use image::DynamicImage;
use ultraviolet::{Mat4, Rotor3, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    image::{Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::Pipeline,
    sync::{self, GpuFuture},
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
            scale: Vec3::one(),
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
    pub(crate) fn as_model_ubo(&mut self) -> ModelUBO {
        if self.dirty {
            let rotation_mat = self.rotation.into_matrix().into_homogeneous();
            let translation_mat = Mat4::from_translation(self.position);
            let scale_mat = Mat4::from_nonuniform_scale(self.scale);
            self.matrix_cache = translation_mat * rotation_mat * scale_mat;
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
    pub grounded: bool,
}
impl RigidBody {
    pub fn new() -> Self {
        Self {
            velocity: Vec3::zero(),
            grounded: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Aabb(Aabb),
}

impl Collider {
    pub fn penetration_vector(
        from: &Collider,
        to: &Collider,
        from_tr: &Transform,
        to_tr: &Transform,
    ) -> Option<Vec3> {
        match (from, to) {
            (Collider::Aabb(from), Collider::Aabb(to)) => {
                let from_max = (from.max * from_tr.scale) + from_tr.position;
                let from_min = (from.min * from_tr.scale) + from_tr.position;
                let to_max = (to.max * to_tr.scale) + to_tr.position;
                let to_min = (to.min * to_tr.scale) + to_tr.position;

                let overlap_pos_x = from_max.x - to_min.x;
                let overlap_neg_x = to_max.x - from_min.x;

                let overlap_pos_y = from_max.y - to_min.y;
                let overlap_neg_y = to_max.y - from_min.y;

                let overlap_pos_z = from_max.z - to_min.z;
                let overlap_neg_z = to_max.z - from_min.z;

                if overlap_pos_x <= 0.0 || overlap_pos_y <= 0.0 || overlap_pos_z <= 0.0 {
                    return None;
                }
                if overlap_neg_x <= 0.0 || overlap_neg_y <= 0.0 || overlap_neg_z <= 0.0 {
                    return None;
                }

                let x;
                let y;
                let z;
                if overlap_pos_x >= overlap_neg_x {
                    x = overlap_neg_x
                } else {
                    x = -overlap_pos_x
                }
                if overlap_pos_y >= overlap_neg_y {
                    y = overlap_neg_y
                } else {
                    y = -overlap_pos_y
                }
                if overlap_pos_z >= overlap_neg_z {
                    z = overlap_neg_z
                } else {
                    z = -overlap_pos_z
                }
                if x.abs() < y.abs() && x.abs() < z.abs() {
                    Some(Vec3::new(x, 0.0, 0.0))
                } else if y.abs() < z.abs() {
                    Some(Vec3::new(0.0, y, 0.0))
                } else {
                    Some(Vec3::new(0.0, 0.0, z))
                }
            }
            _ => unimplemented!(),
        }
    }
}

/// marker/cache component
pub struct Model {
    pub(crate) u_buffer: Vec<Subbuffer<ModelUBO>>,
    pub(crate) descriptor_set: Vec<Arc<PersistentDescriptorSet>>,
    pub(crate) vertex_buffer: Subbuffer<[Vertex]>,
    pub(crate) index_buffer: Subbuffer<[u32]>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Marker, no debug information available yet")
    }
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

        //HACK: this is bullshit, it needs to be completely changed by im lost
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

/// marker/cache component
pub struct Texture {
    pub(crate) image: Arc<Image>,
}
impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Marker, no debug information available yet")
    }
}
impl Texture {
    pub fn create(renderer: &Renderer, image: DynamicImage) -> Self {
        let image = image.as_rgba8().expect("image is invalid");

        let dst_image = Image::new(
            renderer.memory_allocator.clone(),
            ImageCreateInfo {
                format: vulkano::format::Format::R8G8B8A8_SRGB,
                extent: [image.width(), image.height(), 1],
                initial_layout: vulkano::image::ImageLayout::Undefined,
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let staging_buffer = Buffer::new_slice::<u8>(
            renderer.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            image.len() as u64,
        )
        .unwrap();

        let mut staging_buffer_write = staging_buffer.write().unwrap();
        staging_buffer_write.copy_from_slice(image.as_raw());
        drop(staging_buffer_write);
        let mut builder = AutoCommandBufferBuilder::primary(
            &renderer.command_buffer_allocator,
            renderer.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer_to_image(
                vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                    staging_buffer,
                    dst_image.clone(),
                ),
            )
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(renderer.device.clone())
            .then_execute(renderer.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        Texture { image: dst_image }
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
    pub(crate) fn new(position: Vec3, rotation: Rotor3) -> ModelUBO {
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

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, target: Vec3) -> Self {
        Self {
            origin,
            dir: (target - origin).normalized(),
        }
    }
    pub fn from_direction(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalized(),
        }
    }
    pub fn ray_box(ray: &Ray, collider: &Collider, collider_position: Vec3) -> Option<f32> {
        fn calculate_t_values(origin: f32, dir: f32, min: f32, max: f32) -> (f32, f32) {
            if dir.abs() < f32::EPSILON {
                if origin < min || origin > max {
                    return (f32::INFINITY, f32::NEG_INFINITY);
                } else {
                    return (f32::NEG_INFINITY, f32::INFINITY);
                }
            } else {
                let t1 = (min - origin) / dir;
                let t2 = (max - origin) / dir;
                return (t1.min(t2), t1.max(t2));
            }
        }
        match collider {
            Collider::Aabb(aabb) => {
                let aabb = aabb.at_position(collider_position);
                let (t_min_x, t_max_x) =
                    calculate_t_values(ray.origin.x, ray.dir.x, aabb.min.x, aabb.max.x);
                let (t_min_y, t_max_y) =
                    calculate_t_values(ray.origin.y, ray.dir.y, aabb.min.y, aabb.max.y);
                let (t_min_z, t_max_z) =
                    calculate_t_values(ray.origin.z, ray.dir.z, aabb.min.z, aabb.max.z);

                let t_enter = t_min_x.max(t_min_y).max(t_min_z);
                let t_exit = t_max_x.min(t_max_y).min(t_max_z);

                if t_exit < 0.0 || t_enter > t_exit {
                    return None;
                }

                let t_hit = if t_enter < 0.0 { t_exit } else { t_enter };
                Some(t_hit)
            }
            _ => unimplemented!(),
        }
    }
}

impl Aabb {
    pub fn new(half_extent: Vec3, offset: Vec3) -> Aabb {
        let local_max = offset + half_extent;
        let local_min = offset - half_extent;

        Aabb {
            min: local_min,
            max: local_max,
        }
    }

    pub fn at_position(&self, position: Vec3) -> Aabb {
        let global_max = position + self.max;
        let global_min = position + self.min;
        Aabb {
            min: global_min,
            max: global_max,
        }
    }
}

#[derive(Default)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,

    pub(crate) u_buffer: Option<Vec<Subbuffer<AmbientLightUBO>>>,
    pub(crate) descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct AmbientLightUBO {
    pub(crate) color: [f32; 3],
    pub(crate) intensity: f32,
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

    pub(crate) u_buffers: Vec<Subbuffer<PointLightUBO>>,
    pub(crate) descriptor_set: Vec<Arc<PersistentDescriptorSet>>,
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

    pub(crate) fn as_point_ubo(&self, transform: Transform) -> PointLightUBO {
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
pub(crate) struct PointLightUBO {
    pub(crate) position: Vec3,
    _padding: f32,
    pub(crate) color: Vec3,
    pub(crate) brightness: f32,
    pub(crate) linear: f32,
    pub(crate) quadratic: f32,
}

pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],

    pub(crate) u_buffer: Option<Vec<Subbuffer<DirectionalLightUBO>>>,
    pub(crate) descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub(crate) position: [f32; 4],
    pub(crate) color: [f32; 3],
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
#[test]
fn ray_hits_aabb_center() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 0.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_some(), "Expected ray to hit the box");
    let t = result.unwrap();
    assert!(t > 0.0, "Expected hit to be in front of ray origin");
}

#[test]
fn ray_misses_aabb() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(5.0, 0.0, 0.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_none(), "Expected ray to miss the box");
}

#[test]
fn ray_starts_inside_aabb() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_some(), "Expected ray to exit the box");
    let t = result.unwrap();
    assert!(t > 0.0, "Expected hit to be forward from the origin");
}
