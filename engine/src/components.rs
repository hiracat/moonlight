#![allow(dead_code)]
#![allow(unreachable_patterns)]
use gpu_allocator::vulkan::*;
use image::{DynamicImage, EncodableLayout, ImageBuffer, ImageReader, RgbaImage};
use std::{io::Write, path::Path, ptr};
use ultraviolet as uv;

use ash::vk::{self, ComponentMapping, DescriptorSet};

use ultraviolet::{Mat4, Rotor3, Vec3};

use crate::{
    layouts::{self, BINDINGS},
    renderer::{self, alloc_buffer, HasUBO, Renderer, Vertex, FRAMES_IN_FLIGHT},
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
    ///Panics: panics if any scale has negative components
    pub fn penetration_vector(
        from: &Collider,
        to: &Collider,
        from_tr: &Transform,
        to_tr: &Transform,
    ) -> Option<Vec3> {
        debug_assert!(from_tr.scale.x > 0.0 && from_tr.scale.y > 0.0 && from_tr.scale.z > 0.0);
        debug_assert!(to_tr.scale.x > 0.0 && to_tr.scale.y > 0.0 && to_tr.scale.z > 0.0);
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
    pub(crate) descriptor_set_2: Option<Vec<vk::DescriptorSet>>,
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
        self.descriptor_set_2 = Some(sets);
    }
}

impl PointLight {
    pub fn create(
        renderer: &mut Renderer,
        color: Vec3,
        brightness: f32,
        linear: Option<f32>,
        quadratic: Option<f32>,
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
            descriptor_set_2: None,
        };
        light.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            &mut Transform::new(),
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
            light.descriptor_set_2.as_ref().unwrap(),
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
#[derive(Debug)]
pub struct Texture {
    image: vk::Image,
    memory: Allocation,

    pub(crate) descriptor_set_2: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl Texture {
    pub fn default_image(renderer: &mut Renderer) -> Texture {
        let mut image = RgbaImage::new(1, 1);
        image.put_pixel(0, 0, image::Rgba::<u8>([u8::MAX, u8::MAX, u8::MAX, 0]));
        let dynamic_image = DynamicImage::ImageRgba8(image);
        Texture::create_image(&dynamic_image, renderer)
    }
    pub fn create_image(image: &DynamicImage, renderer: &mut Renderer) -> Texture {
        let image = image.to_rgba8();
        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &renderer.queue_family_index,
            queue_family_index_count: 1,
            size: (image.width() * image.height() * 4) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { renderer.device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe {
            renderer
                .device
                .get_buffer_memory_requirements(staging_buffer)
        };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = renderer
            .allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            renderer
                .device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        let image_format = vk::Format::R8G8B8A8_SRGB;

        let create_info = vk::ImageCreateInfo {
            p_queue_family_indices: &renderer.queue_family_index,
            queue_family_index_count: 1,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            extent: vk::Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1,
            },
            flags: vk::ImageCreateFlags::empty(),
            format: image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let final_image = unsafe { renderer.device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { renderer.device.get_image_memory_requirements(final_image) };

        let final_image_mem_desc = AllocationCreateDesc {
            name: "image memory",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let image_mem = renderer
            .allocator
            .lock()
            .unwrap()
            .allocate(&final_image_mem_desc)
            .unwrap();

        unsafe {
            renderer
                .device
                .bind_image_memory(final_image, image_mem.memory(), image_mem.offset())
                .unwrap()
        }

        staging_mem
            .mapped_slice_mut()
            .unwrap()
            .write(image.as_bytes())
            .unwrap();

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        renderer::instant_submit_command_buffer(
            &renderer.device,
            renderer.one_time_submit,
            renderer.one_time_submit_pool,
            renderer.queue,
            |command_buffer| {
                let to_writable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: final_image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };

                unsafe {
                    renderer.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_writable],
                    )
                };

                let regions = [vk::BufferImageCopy {
                    image_offset: vk::Offset3D::default(),
                    image_subresource: vk::ImageSubresourceLayers {
                        mip_level: 0,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        base_array_layer: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: image.width(),
                        height: image.height(),
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                }];

                unsafe {
                    renderer.device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer,
                        final_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    )
                };
                let to_readable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    image: final_image,
                    subresource_range,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                };
                //barrier the image into the shader readable layout
                unsafe {
                    renderer.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_readable],
                    )
                };
            },
        );

        unsafe { renderer.device.destroy_buffer(staging_buffer, None) };
        renderer
            .allocator
            .lock()
            .unwrap()
            .free(staging_mem)
            .unwrap();

        let allocate_info = vk::DescriptorSetAllocateInfo {
            p_set_layouts: &renderer.descriptor_layouts.geometry_static_texture_layout_2,
            descriptor_pool: renderer.image_descriptor_pool_0,
            descriptor_set_count: 1,
            ..Default::default()
        };

        let descriptor_set = unsafe {
            renderer
                .device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()[0]
        };

        let sampler_create_info = vk::SamplerCreateInfo {
            flags: vk::SamplerCreateFlags::empty(),

            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,

            mipmap_mode: vk::SamplerMipmapMode::LINEAR,

            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,

            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,

            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,

            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE, // or the max mip level of your texture

            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,

            ..Default::default()
        };

        let sampler = unsafe {
            renderer
                .device
                .create_sampler(&sampler_create_info, None)
                .unwrap()
        };
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: final_image,
            format: image_format,
            view_type: vk::ImageViewType::TYPE_2D,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range,
            ..Default::default()
        };
        let image_view = unsafe {
            renderer
                .device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };

        let descriptor_image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let descriptor_writes = [vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 0,
            p_image_info: &descriptor_image_info,
            p_buffer_info: ptr::null(),
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            dst_array_element: 0,
            p_texel_buffer_view: ptr::null(),
            ..Default::default()
        }];

        unsafe {
            renderer
                .device
                .update_descriptor_sets(&descriptor_writes, &[])
        };

        Self {
            image: final_image,
            memory: image_mem,
            sampler,
            descriptor_set_2: descriptor_set,
        }
    }
}
