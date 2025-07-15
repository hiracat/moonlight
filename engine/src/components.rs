#![allow(dead_code)]
#![allow(unreachable_patterns)]
use core::fmt;
use gpu_allocator::vulkan::*;
use std::sync::Arc;

use ash::vk::{self, DescriptorSet};

use ultraviolet::{Mat4, Rotor3, Vec3};

use crate::{
    layouts::{self, BINDINGS},
    renderer::{alloc_buffer, HasUBO, Renderer, Vertex, FRAMES_IN_FLIGHT},
};

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

trait DescriptorPerObject {
    fn set_descriptor_sets(&mut self, sets: Vec<DescriptorSet>);
    fn allocate_descriptor_sets(
        &mut self,
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        count: u32,
    ) {
        let set_layouts = vec![layout; count as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            p_set_layouts: set_layouts.as_ptr(),
            descriptor_set_count: set_layouts.len() as u32,
            ..Default::default()
        };
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };
        self.set_descriptor_sets(sets);
    }
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
#[derive(Debug)]
pub struct Model {
    pub(crate) u_buffer: Option<Vec<vk::Buffer>>,
    // write to this to upload values, only need to update desriptors to point at this once
    pub(crate) allocations: Option<Vec<Allocation>>,

    pub(crate) descriptor_set: Option<Vec<vk::DescriptorSet>>,

    pub(crate) vertex_buffer: vk::Buffer,
    pub(crate) vertex_count: usize,
    pub(crate) vertex_buffer_memory: Allocation,

    pub(crate) index_buffer: vk::Buffer,
    pub(crate) index_count: usize,
    pub(crate) index_buffer_memory: Allocation,
}

impl HasUBO for Model {
    type Context = Transform;
    type UBOData = ModelUBO;
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocation: Vec<Allocation>) {
        self.u_buffer = Some(buffer);
        self.allocations = Some(allocation);
    }
    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData {
        if context.dirty {
            let rotation_mat = context.rotation.into_matrix().into_homogeneous();
            let translation_mat = Mat4::from_translation(context.position);
            let scale_mat = Mat4::from_nonuniform_scale(context.scale);
            context.matrix_cache = translation_mat * rotation_mat * scale_mat;
            context.inv_trans_cache = context.matrix_cache.inversed().transposed();
        }

        ModelUBO {
            model: context.matrix_cache,
            normal: context.inv_trans_cache,
        }
    }
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>> {
        self.u_buffer.as_ref()
    }
}

impl DescriptorPerObject for Model {
    fn set_descriptor_sets(&mut self, sets: Vec<DescriptorSet>) {
        self.descriptor_set = Some(sets);
    }
}

impl Model {
    pub fn create(renderer: &mut Renderer, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let (vertex_buffer, mut vertex_alloc) = alloc_buffer(
            renderer.allocator.clone(),
            1,
            vertices.len() as u64 * size_of::<Vertex>() as u64,
            &renderer.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(vertices.as_ref()),
            "vertex buffer",
        );

        let (index_buffer, mut index_alloc) = alloc_buffer(
            renderer.allocator.clone(),
            1,
            indices.len() as u64 * 4,
            &renderer.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(indices.as_ref()),
            "index buffer",
        );

        let mut model = Model {
            u_buffer: None,
            allocations: None,
            descriptor_set: None,
            index_count: indices.len(),
            vertex_count: vertices.len(),
            vertex_buffer: vertex_buffer[0],
            index_buffer: index_buffer[0],
            index_buffer_memory: index_alloc.pop().expect("should have 1 element"),
            vertex_buffer_memory: vertex_alloc.pop().expect("should have 1 element"),
        };
        model.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            &mut Transform::new(),
            FRAMES_IN_FLIGHT,
        );
        model.allocate_descriptor_sets(
            &renderer.device,
            renderer.model_pool_0,
            renderer.descriptor_layouts.geometry_per_model_layout_0,
            FRAMES_IN_FLIGHT as u32,
        );
        let descriptor_set = model.descriptor_set.as_ref().unwrap();

        // Step 2: now borrow model to call method
        model.write_descriptor_sets(&renderer.device, descriptor_set, BINDINGS.model);

        model
    }
}

/// marker/cache component
pub struct Texture {
    pub(crate) image: Arc<image::DynamicImage>,
}
impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Marker, no debug information available yet")
    }
}
// impl Texture {
//     pub fn create(renderer: &Renderer, image: DynamicImage) -> Self {
//         let image = image.as_rgba8().expect("image is invalid");
//
//         let dst_image = Image::new(
//             renderer.memory_allocator.clone(),
//             ImageCreateInfo {
//                 format: vulkano::format::Format::R8G8B8A8_SRGB,
//                 extent: [image.width(), image.height(), 1],
//                 initial_layout: vulkano::image::ImageLayout::Undefined,
//                 usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
//                 ..Default::default()
//             },
//             AllocationCreateInfo {
//                 memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
//                 ..Default::default()
//             },
//         )
//         .unwrap();
//
//         let staging_buffer = Buffer::new_slice::<u8>(
//             renderer.memory_allocator.clone(),
//             BufferCreateInfo {
//                 usage: BufferUsage::TRANSFER_SRC,
//                 ..Default::default()
//             },
//             AllocationCreateInfo {
//                 memory_type_filter: MemoryTypeFilter::PREFER_HOST
//                     | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
//                 ..Default::default()
//             },
//             image.len() as u64,
//         )
//         .unwrap();
//
//         let mut staging_buffer_write = staging_buffer.write().unwrap();
//         staging_buffer_write.copy_from_slice(image.as_raw());
//         drop(staging_buffer_write);
//         let mut builder = AutoCommandBufferBuilder::primary(
//             &renderer.command_pool,
//             renderer.queue.queue_family_index(),
//             CommandBufferUsage::OneTimeSubmit,
//         )
//         .unwrap();
//
//         builder
//             .copy_buffer_to_image(
//                 vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
//                     staging_buffer,
//                     dst_image.clone(),
//                 ),
//             )
//             .unwrap();
//
//         let command_buffer = builder.build().unwrap();
//
//         let future = sync::now(renderer.device.clone())
//             .then_execute(renderer.queue.clone(), command_buffer)
//             .unwrap()
//             .then_signal_fence_and_flush()
//             .unwrap();
//         future.wait(None).unwrap();
//         Texture { image: dst_image }
//     }
// }

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

    //ambientlightubo
    pub(crate) u_buffer: Option<Vec<vk::Buffer>>,
    pub(crate) allocations: Option<Vec<Allocation>>,
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct AmbientLightUBO {
    pub(crate) color: [f32; 3],
    pub(crate) intensity: f32,
}
impl AmbientLight {
    pub fn create(renderer: &mut Renderer, color: [f32; 3], intensity: f32) -> Self {
        let mut light = Self {
            color,
            intensity,
            u_buffer: None,
            allocations: None,
        };
        light.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            &mut (),
            FRAMES_IN_FLIGHT,
        );
        light.write_descriptor_sets(
            &renderer.device,
            &mut renderer.lighting_per_frame_sets_1,
            layouts::BINDINGS.ambient,
        );
        light
    }
}

impl HasUBO for AmbientLight {
    type Context = ();
    type UBOData = AmbientLightUBO;
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocation: Vec<Allocation>) {
        self.u_buffer = Some(buffer);
        self.allocations = Some(allocation);
    }
    #[allow(unused)]
    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData {
        Self::UBOData {
            color: self.color,
            intensity: self.intensity,
        }
    }
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>> {
        self.u_buffer.as_ref()
    }
}

pub struct PointLight {
    pub color: Vec3,
    pub brightness: f32,
    pub linear: f32,
    pub quadratic: f32,
    pub dirty: bool,

    // point light ubo
    pub(crate) u_buffers: Option<Vec<vk::Buffer>>,
    // write to this to upload values, only need to update desriptors to point at this once
    pub(crate) allocations: Option<Vec<Allocation>>,
    pub(crate) descriptor_set: Option<Vec<vk::DescriptorSet>>,
}

impl HasUBO for PointLight {
    type UBOData = PointLightUBO;
    type Context = Transform;
    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData {
        PointLightUBO {
            position: context.position,
            color: self.color,
            brightness: self.brightness,
            linear: self.linear,
            quadratic: self.quadratic,
            _padding: 0.0,
        }
    }
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocation: Vec<Allocation>) {
        self.allocations = Some(allocation);
        self.u_buffers = Some(buffer);
    }
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>> {
        self.u_buffers.as_ref()
    }
}

impl DescriptorPerObject for PointLight {
    fn set_descriptor_sets(&mut self, sets: Vec<DescriptorSet>) {
        self.descriptor_set = Some(sets);
    }
}

impl PointLight {
    pub fn create(
        renderer: &mut Renderer,
        color: Vec3,
        brightness: f32,
        linear: Option<f32>,
        quadratic: Option<f32>,
        position: &mut Transform,
    ) -> Self {
        //TODO: this is even more bullshit, ill get to it eventually
        let mut light = PointLight {
            color,
            brightness,
            linear: linear.unwrap_or(3.00),
            quadratic: quadratic.unwrap_or(0.00),
            dirty: true,
            u_buffers: None,
            allocations: None,
            descriptor_set: None,
        };
        light.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            position,
            FRAMES_IN_FLIGHT,
        );
        light.allocate_descriptor_sets(
            &renderer.device,
            renderer.lighting_per_light_pool_2,
            renderer.descriptor_layouts.lighting_per_light_layout_2,
            FRAMES_IN_FLIGHT as u32,
        );
        light.write_descriptor_sets(
            &renderer.device,
            light.descriptor_set.as_ref().unwrap(),
            BINDINGS.point,
        );
        light
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
impl PointLightUBO {
    pub(crate) fn new() -> Self {
        PointLightUBO {
            position: Vec3::zero(),
            _padding: 0.0,
            color: Vec3::zero(),
            brightness: 0.0,
            linear: 0.0,
            quadratic: 0.0,
        }
    }
}

pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],

    pub(crate) u_buffers: Option<Vec<vk::Buffer>>,
    pub(crate) allocations: Option<Vec<Allocation>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct DirectionalLightUBO {
    pub(crate) position: [f32; 4],
    pub(crate) color: [f32; 3],
}

impl DirectionalLight {
    pub fn create(renderer: &mut Renderer, position: [f32; 4], color: [f32; 3]) -> Self {
        let mut light = Self {
            position,
            color,
            u_buffers: None,
            allocations: None,
        };
        light.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            &mut (),
            FRAMES_IN_FLIGHT,
        );
        light.write_descriptor_sets(
            &renderer.device,
            &mut renderer.lighting_per_frame_sets_1,
            layouts::BINDINGS.directional,
        );
        light
    }
}

impl HasUBO for DirectionalLight {
    type UBOData = DirectionalLightUBO;
    type Context = ();
    #[allow(unused)]
    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData {
        Self::UBOData {
            color: self.color,
            position: self.position,
        }
    }
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocation: Vec<Allocation>) {
        self.u_buffers = Some(buffer);
        self.allocations = Some(allocation);
    }
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>> {
        self.u_buffers.as_ref()
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
