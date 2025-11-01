use std::{io::Write, path::Path};

use ash::vk;
use bytemuck::{bytes_of, Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use image::{DynamicImage, EncodableLayout, GenericImage, ImageReader, Rgba};
use ultraviolet as uv;

use crate::renderer::draw::{
    alloc_buffers, instant_submit_command_buffer, QueueFamilyIndex, Renderer, SharedAllocator,
};

pub struct ResourceManager {
    device: ash::Device,
    allocator: SharedAllocator,

    meshes: Vec<Option<GpuMesh>>,
    textures: Vec<Option<GpuTexture>>,

    default_texture: GpuTexture,

    pub(crate) ring_buffer: UniformRingBuffer,

    //NOTE: unowned resources, just handles for reference/creating without requiring renderer to be
    //passed in
    queue_family_index: QueueFamilyIndex,
    queue: vk::Queue,
    one_time_submit_buffer: vk::CommandBuffer,
    one_time_submit_pool: vk::CommandPool,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub albedo: Texture,
}

#[derive(Debug, Clone, Copy)]
pub struct Texture {
    id: usize,
    path: Option<&'static str>,
}
#[derive(Debug, Clone, Copy)]
pub struct Skybox {
    pub(crate) material: Material,
}
impl Skybox {
    pub fn new(cubemap: Texture) -> Self {
        Self {
            material: Material { albedo: cubemap },
        }
    }
}
impl Material {
    pub fn create(albedo: Texture) -> Self {
        Self { albedo }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mesh {
    id: usize,
    path: Option<&'static str>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

pub(crate) struct GpuTexture {
    pub(crate) image: vk::Image,
    pub(crate) image_view: vk::ImageView,
    pub(crate) sampler: vk::Sampler,
    pub(crate) memory: Allocation,
}
pub(crate) struct GpuMesh {
    pub(crate) vertex_buffer: vk::Buffer,
    pub(crate) index_buffer: vk::Buffer,
    pub(crate) index_alloc: Allocation,
    pub(crate) vertex_alloc: Allocation,
    pub(crate) index_count: u32,
}

pub struct UniformRingBuffer {
    buffer: vk::Buffer,
    memory: Allocation,
    size: u64,
    current: u64,
}
impl UniformRingBuffer {
    fn create(allocator: SharedAllocator, device: &ash::Device, size: u64) -> Self {
        let sharing_mode = vk::SharingMode::EXCLUSIVE;
        let mut buffers = alloc_buffers(
            allocator,
            1,
            size,
            device,
            sharing_mode,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            &vec![0u8; size as usize],
            "ring buffer",
        );
        let buffer = buffers.0.remove(0);
        let allocation = buffers.1.remove(0);

        Self {
            buffer: buffer,
            memory: allocation,
            size: size,
            current: 0,
        }
    }
    pub fn allocate(&mut self, size: usize) -> (vk::Buffer, u64) {
        if self.current + size as u64 > self.size {
            eprintln!(
                "Ring buffer wrapping! current={}, size={}, buffer_size={}",
                self.current, size, self.size
            );
            self.current = 0;
        }

        let offset = self.current;
        self.current += size as u64;

        (self.buffer, offset)
    }
    pub fn write<T: Pod + Zeroable>(&mut self, data: &T, offset: u64) {
        let offset = offset as usize;
        let data = bytes_of(data);
        let size = size_of::<T>();

        let memory = self
            .memory
            .mapped_slice_mut()
            .expect("memory should be host visable");
        let data_segment = &mut memory[offset..offset + size];
        data_segment.copy_from_slice(data);
    }
}

impl Vertex {
    pub fn new(postion: uv::Vec3, normal: uv::Vec3, uv: uv::Vec2) -> Self {
        Self {
            position: *postion.as_array(),
            normal: *normal.as_array(),
            uv: *uv.as_array(),
        }
    }
    pub fn get_vertex_attributes() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            // position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0x0,
            },
            // normal
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0xC,
            },
            // uv
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0x18,
            },
        ]
    }
}

impl ResourceManager {
    pub(crate) fn init(
        device: &ash::Device,
        allocator: SharedAllocator,
        one_time_submit_pool: vk::CommandPool,
        one_time_submit_buffer: vk::CommandBuffer,
        queue: vk::Queue,
        queue_family_index: QueueFamilyIndex,
    ) -> Self {
        let ring_buffer = UniformRingBuffer::create(allocator.clone(), device, 0xFFFFF);
        let pixel = Rgba::from([255, 255, 255, 0]);
        let mut image = DynamicImage::new_rgb8(1, 1);
        image.put_pixel(0, 0, pixel);
        let default = GpuTexture::create_2d(
            device,
            queue_family_index,
            allocator.clone(),
            &image,
            queue,
            one_time_submit_buffer,
            one_time_submit_pool,
        );

        ResourceManager {
            ring_buffer,
            default_texture: default,
            device: device.clone(),
            allocator,
            meshes: Vec::new(),
            textures: Vec::new(),
            one_time_submit_buffer,
            queue,
            queue_family_index,
            one_time_submit_pool,
        }
    }
    pub(crate) fn default_texture(&mut self) -> &GpuTexture {
        &self.default_texture
    }
    pub fn create_texture(&mut self, path: &'static str) -> Texture {
        let image = ImageReader::open(Path::new(path))
            .unwrap()
            .decode()
            .unwrap();
        self.textures.push(Some(GpuTexture::create_2d(
            &self.device,
            self.queue_family_index,
            self.allocator.clone(),
            &image,
            self.queue,
            self.one_time_submit_buffer,
            self.one_time_submit_pool,
        )));
        Texture {
            id: self.textures.len() - 1,
            path: Some(path),
        }
    }
    pub fn create_cubemap(&mut self, paths: [&'static str; 6]) -> Texture {
        let images: [DynamicImage; 6] = paths.map(|path| {
            ImageReader::open(Path::new(path))
                .unwrap()
                .decode()
                .unwrap()
        });
        self.textures.push(Some(GpuTexture::create_cubemap(
            &self.device,
            self.queue_family_index,
            self.allocator.clone(),
            images.into(),
            self.queue,
            self.one_time_submit_buffer,
            self.one_time_submit_pool,
        )));
        Texture {
            id: self.textures.len() - 1,
            //HACK: need to either seperate different texture types out(probably smartest and
            //easiest, or figure out a way to do multiple path types per texture(maybe enum)
            path: None,
        }
    }
    pub fn create_mesh(&mut self, path: &'static str) -> Mesh {
        let (document, buffers, _images) = gltf::import(path).unwrap();
        assert_eq!(document.meshes().len(), 1);
        let mesh = &document.meshes().next().unwrap();
        let primative: &gltf::Primitive = &mesh.primitives().next().unwrap();

        let reader = primative.reader(|buffer| Some(&buffers[buffer.index()]));

        let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();
        let positions = reader.read_positions().unwrap();
        let normals = reader.read_normals().unwrap();
        let tex_coords = reader.read_tex_coords(0).unwrap().into_f32();

        let vertices: Vec<_> = positions
            .zip(normals)
            .zip(tex_coords)
            .map(|((position, normal), uv)| Vertex {
                position,
                normal,
                uv,
            })
            .collect();

        let (vertex_buffer, mut vertex_alloc) = alloc_buffers(
            self.allocator.clone(),
            1,
            vertices.len() as u64 * size_of::<Vertex>() as u64,
            &self.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(vertices.as_ref()),
            "vertex buffer",
        );

        let (index_buffer, mut index_alloc) = alloc_buffers(
            self.allocator.clone(),
            1,
            indices.len() as u64 * 4,
            &self.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(indices.as_ref()),
            "index buffer",
        );

        let mesh = GpuMesh {
            index_alloc: index_alloc.remove(0),
            vertex_alloc: vertex_alloc.remove(0),
            index_count: indices.len() as u32,
            index_buffer: index_buffer[0],
            vertex_buffer: vertex_buffer[0],
        };
        self.meshes.push(Some(mesh));
        Mesh {
            id: self.meshes.len() - 1,
            path: Some(path),
        }
    }
    pub(crate) fn get_mesh(&mut self, mesh: Mesh) -> Option<&GpuMesh> {
        self.meshes.get(mesh.id).and_then(|m| m.as_ref())
    }
    pub(crate) fn get_texture(&mut self, tex: Texture) -> Option<&GpuTexture> {
        self.textures.get(tex.id).and_then(|m| m.as_ref())
    }
    pub(crate) fn allocate_temp_descriptor_set(
        &mut self,
        set_layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
    ) -> vk::DescriptorSet {
        let allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            p_set_layouts: &set_layout,
            descriptor_set_count: 1,
            ..Default::default()
        };
        unsafe {
            self.device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()[0]
        }
    }
}

impl GpuTexture {
    pub fn create_cubemap(
        device: &ash::Device,
        queue_family_index: QueueFamilyIndex,
        allocator: SharedAllocator,
        mut images: [DynamicImage; 6],
        queue: vk::Queue,
        one_time_submit_buffer: vk::CommandBuffer,
        one_time_submit_pool: vk::CommandPool,
    ) -> Self {
        let image_bytes: Vec<_> = images
            .iter_mut()
            .map(|x| x.to_rgba8().into_raw())
            .flatten()
            .collect();

        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            // 4 bits per pixel * 6 images
            size: (images[0].width() * images[0].height() * 4 * 6) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        let image_format = vk::Format::R8G8B8A8_SRGB;

        let create_info = vk::ImageCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            extent: vk::Extent3D {
                width: images[0].width(),
                height: images[0].height(),
                depth: 1,
            },
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format: image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 6,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let final_image = unsafe { device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_image_memory_requirements(final_image) };

        let final_image_mem_desc = AllocationCreateDesc {
            name: "image memory",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let image_mem = allocator
            .lock()
            .unwrap()
            .allocate(&final_image_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_image_memory(final_image, image_mem.memory(), image_mem.offset())
                .unwrap()
        }

        staging_mem
            .mapped_slice_mut()
            .unwrap()
            .write(&image_bytes)
            .unwrap();

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        };
        instant_submit_command_buffer(
            &device,
            one_time_submit_buffer,
            one_time_submit_pool,
            queue,
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
                    device.cmd_pipeline_barrier(
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
                        layer_count: 6,
                        base_array_layer: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: images[0].width(),
                        height: images[0].height(),
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                }];

                unsafe {
                    device.cmd_copy_buffer_to_image(
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
                    device.cmd_pipeline_barrier(
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

        unsafe { device.destroy_buffer(staging_buffer, None) };
        allocator.lock().unwrap().free(staging_mem).unwrap();

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

        let sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: final_image,
            format: image_format,
            view_type: vk::ImageViewType::CUBE,
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
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        Self {
            image_view,
            image: final_image,
            memory: image_mem,
            sampler,
        }
    }
    pub fn create_2d(
        device: &ash::Device,
        queue_family_index: QueueFamilyIndex,
        allocator: SharedAllocator,
        image: &DynamicImage,
        queue: vk::Queue,
        one_time_submit_buffer: vk::CommandBuffer,
        one_time_submit_pool: vk::CommandPool,
    ) -> Self {
        let image = image.to_rgba8();
        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            size: (image.width() * image.height() * 4) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        let image_format = vk::Format::R8G8B8A8_SRGB;

        let create_info = vk::ImageCreateInfo {
            p_queue_family_indices: &queue_family_index,
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
        let final_image = unsafe { device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_image_memory_requirements(final_image) };

        let final_image_mem_desc = AllocationCreateDesc {
            name: "image memory",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let image_mem = allocator
            .lock()
            .unwrap()
            .allocate(&final_image_mem_desc)
            .unwrap();

        unsafe {
            device
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
        instant_submit_command_buffer(
            &device,
            one_time_submit_buffer,
            one_time_submit_pool,
            queue,
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
                    device.cmd_pipeline_barrier(
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
                    device.cmd_copy_buffer_to_image(
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
                    device.cmd_pipeline_barrier(
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

        unsafe { device.destroy_buffer(staging_buffer, None) };
        allocator.lock().unwrap().free(staging_mem).unwrap();

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

        let sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
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
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        Self {
            image_view,
            image: final_image,
            memory: image_mem,
            sampler,
        }
    }
}

impl GpuMesh {
    pub fn create(renderer: &mut Renderer, vertices: &[Vertex], indices: &[u32]) -> Self {
        let (mut vertex_buffers, mut vertex_allocs) = alloc_buffers(
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

        let (mut index_buffers, mut index_allocs) = alloc_buffers(
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

        Self {
            index_count: indices.len() as u32,
            index_alloc: index_allocs.pop().unwrap(),
            index_buffer: index_buffers.pop().unwrap(),
            vertex_alloc: vertex_allocs.pop().unwrap(),
            vertex_buffer: vertex_buffers.pop().unwrap(),
        }
    }
}
