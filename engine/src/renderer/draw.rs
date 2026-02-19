#![allow(clippy::cast_possible_truncation)]

use crate::ecs::World;
use crate::renderer::pipelines::{
    create_builtin_graphics_pipelines, create_graphics_pipeline,
    create_pipeline_layout_from_vert_frag, ColorBlendState, DepthStencilState,
    GraphicsPipelineDesc, InputAssemblyState, MultisampleState, PipelineBundle, PipelineKey,
    RasterState, VertexInputState,
};
use crate::renderer::{
    resources::{GpuTexture, Mesh, ResourceManager},
    swapchain::{framebuffers::GBufferResources, renderpass::create_renderpass},
};
use crate::vulkan::{QueueFamilyIndex, SharedAllocator, VulkanContext};
use crate::{components::Camera, renderer::swapchain::SwapchainResources};
use ash::vk::{
    self, DynamicState, PipelineColorBlendAttachmentState, VertexInputAttributeDescription,
    VertexInputBindingDescription,
};
use bytemuck::cast_slice;
use egui::{epaint, ClippedPrimitive, TextureId};
use gpu_allocator::vulkan::*;
use image::{DynamicImage, ImageBuffer, Rgba};
use std::collections::HashMap;
use std::path::Path;
use std::{default::Default, ptr};
use std::{
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
    usize,
};
use winit::dpi::PhysicalSize;
//INFO: idk how to fix this, one wants raw window handles and the other says no
#[allow(deprecated)]
use winit::{
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

pub const GEOMETRY_SUBPASS: u32 = 0;
pub const LIGHTING_SUBPASS: u32 = 1;
pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct UIRenderer {
    pub ui_ctx: egui::Context,
    pub winit_egui_state: egui_winit::State,
    pub full_output: Option<egui::FullOutput>,

    pub renderpass: vk::RenderPass,
    pub pipeline: vk::Pipeline,

    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    vertex_memory: Allocation,
    index_memory: Allocation,
    pub framebuffers: Vec<vk::Framebuffer>,

    descriptor_set_layout: vk::DescriptorSetLayout,

    descriptor_sets: HashMap<TextureId, vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    textures: HashMap<egui::TextureId, GpuTexture>,
    // not owned/handles
    queue_family_index: QueueFamilyIndex,
    allocator: SharedAllocator,
    queue: vk::Queue,
    one_time_submit: vk::CommandBuffer,
    one_time_submit_pool: vk::CommandPool,
}

pub struct WorldRenderer {
    // indexed by pipelineHandle
    pub render_pass: vk::RenderPass,
    pub pipelines: Vec<PipelineBundle>,

    pub per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,
    pub per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub gbuffers: GBufferResources,
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: Option<Mesh>,
    // shaders set indices are required to be contiguous
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for WorldRenderer {
    fn drop(&mut self) {
        todo!();
    }
}

impl WorldRenderer {
    #[allow(clippy::too_many_lines)]
    pub fn record_commands(
        &mut self,
        world: &mut World,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        window_size: vk::Extent2D,
        resource_manager: &mut ResourceManager,
        swapchain_image_index: usize,
    ) {
        let jobs = Self::setup_gpu_build_draw_jobs(self, world);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: window_size,
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size.width as f32,
            height: window_size.height as f32,
            max_depth: 1.0,
            min_depth: 0.0,
        };
        let clear_values = vec![
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        ];
        unsafe {
            device.cmd_begin_render_pass2(
                command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[swapchain_image_index],
                    p_clear_values: clear_values.as_ptr(),
                    clear_value_count: clear_values.len() as u32,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: window_size,
                    },
                    ..Default::default()
                },
                &vk::SubpassBeginInfo {
                    contents: vk::SubpassContents::INLINE,
                    ..Default::default()
                },
            );
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        };

        for (pipeline_index, job_list) in jobs.iter().enumerate() {
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[pipeline_index].pipeline,
                );
            }
            for job in job_list {
                assert_eq!(
                    job.descriptor_sets.len(),
                    self.pipelines[pipeline_index as usize]
                        .descriptor_set_layouts
                        .len(),
                    "must be an equal amount of descriptor sets as descriptor set layouts"
                );

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines[pipeline_index].layout,
                        0,
                        &job.descriptor_sets,
                        &[],
                    );
                }
                if let Some(mesh) = job.mesh {
                    let mesh = resource_manager.get_mesh(mesh).unwrap();

                    unsafe {
                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[mesh.vertex_buffer],
                            &[0],
                        );
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            mesh.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                    }
                } else {
                    unsafe {
                        device.cmd_draw(command_buffer, 3, 1, 0, 0);
                    }
                }
            }
            if pipeline_index == PipelineKey::AnimatedGeometry as usize {
                unsafe {
                    device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
                }
            }
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[pipeline_index as usize].pipeline,
                );
                device.cmd_end_render_pass(command_buffer);
            }
        }
    }

    // this is sorta like a frame graph builder kinda thing
    fn setup_gpu_build_draw_jobs(&mut self, world: &mut World) -> Vec<Vec<DrawJob>> {
        // indexed by pipelinekey, then just everything that belongs in that pipeline
        let mut jobs: Vec<Vec<DrawJob>> = Vec::with_capacity(self.pipelines.len());

        for pipeline in &self.pipelines {
            let job_set = (pipeline.write_data_and_build_draw_jobs)(
                &self.device,
                &mut self.resource_manager,
                self.per_frame[self.current_frame].transient_pool,
                world,
                &pipeline.descriptor_set_layouts,
                &self.swapchain.per_swapchain_image_descriptor_sets
                    [self.swapchain_image_index as usize],
            );
            jobs.push(job_set);
        }
        return jobs;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(context: &VulkanContext) -> Self {
        let framebuffers = create_framebuffers(
            device,
            &swapchain_images,
            render_pass,
            allocator.clone(),
            window_size,
            image_format.format,
        );

        // one set per swapchain image, * 4 for color normal position and final_color * 3 for
        // safety
        let required_sets: u32 = (swapchain_images.len() * 4 * 3) as u32;

        let pool_sizes = [vk::DescriptorPoolSize {
            // per uniform buffer, so six descriptors per
            // set(2x what is needed for safety
            descriptor_count: 6 * required_sets,
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
        }];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: required_sets,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        pool_size_count: pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        let descriptor_sets = create_attachment_descriptor_sets(
            device,
            descriptor_pool,
            per_swapchain_image_set_layout,
            &[
                framebuffers.color_images.as_slice(),
                framebuffers.normal_images.as_slice(),
                framebuffers.position_images.as_slice(),
            ],
            &[0, 1, 2],
        );

        let render_finished = create_semaphores(device, image_count as usize);

        let render_pass = create_renderpass(&device, swapchain_image_format.format);
        let shared_allocator = Arc::new(Mutex::new(memory_allocator));

        let pipelines = create_builtin_graphics_pipelines(&device, render_pass);
        let pool_create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
            ..Default::default()
        };
        let one_time_command_pool =
            unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: one_time_command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };
        let one_time_submit = unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] };
        let resource_manager = ResourceManager::init(
            &device,
            shared_allocator.clone(),
            one_time_command_pool,
            one_time_submit,
            queue,
            queue_family_index,
        );

        let swapchain = SwapchainResources::create(
            &surface_loader,
            &swapchain_loader,
            &device,
            physical_device,
            render_pass,
            None,
            swapchain_image_format,
            window_size,
            shared_allocator.clone(),
            surface,
            //HACK: i need to find some other way than just having this a explicit number
            pipelines[2].descriptor_set_layouts[0],
            &[queue_family_index],
        );

        let mut per_frame = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            per_frame.push(PerFrame::create(&device, queue_family_index));
        }

        let ui = UIRenderer::init(
            &device,
            window.clone(),
            one_time_command_pool,
            one_time_submit,
            queue,
            shared_allocator.clone(),
            queue_family_index,
            &swapchain.framebuffers.swapchain_image_views,
            swapchain_image_format.format,
            vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            },
        );

        WorldRenderer {
            ui,
            pipelines,
            one_time_submit,
            one_time_submit_pool: one_time_command_pool,
            _entry: entry,
            debug_messenger,
            debug_utils_loader,
            instance,
            framebuffer_resized: false,
            current_frame: 0,
            swapchain,
            render_pass,
            allocator: shared_allocator,
            surface,
            surface_loader,
            swapchain_loader: swapchain_loader.clone(),
            device,
            queue,
            start_time,
            window: window.clone(),
            queue_family_index,
            physical_device,
            per_frame,
            swapchain_image_index: 0,
            resource_manager,
        }
    }
}
struct UIDrawJob {
    vertex_offset: usize,
    index_offset: usize,
    index_count: usize,
    descriptor_set: vk::DescriptorSet,
    scissor: vk::Rect2D,
}

impl UIRenderer {
    pub fn draw_meshes(
        &mut self,
        geometry: &[ClippedPrimitive],
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        pixels_per_point: f32,
        window_size: PhysicalSize<u32>,
    ) {
        let screen_size = [window_size.width as f32, window_size.height as f32];
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,                                  // offset
                bytemuck::cast_slice(&screen_size), // converts [f32; 2] to &[u8]
            );
        }

        let mut draw_jobs = Vec::new();
        let mut current_vertex_offset = 0;
        let mut current_vertex_byte_offset = 0;
        let mut current_index_offset = 0;
        let mut current_index_byte_offset = 0;
        for item in geometry {
            match &item.primitive {
                egui::epaint::Primitive::Mesh(x) => {
                    let vertex_memory = self.vertex_memory.mapped_slice_mut().unwrap();
                    let vertex_writeable_slice = &mut vertex_memory[current_vertex_byte_offset
                        ..current_vertex_byte_offset
                            + x.vertices.len() * size_of::<epaint::Vertex>()];

                    let vertex_bytes: &[u8] = cast_slice(&x.vertices);
                    vertex_writeable_slice.copy_from_slice(vertex_bytes);

                    let index_memory = self.index_memory.mapped_slice_mut().unwrap();
                    let index_writeable_slice = &mut index_memory[current_index_byte_offset
                        ..current_index_byte_offset + x.indices.len() * size_of::<u32>()];

                    let index_bytes: &[u8] = cast_slice(&x.indices);
                    index_writeable_slice.copy_from_slice(index_bytes);
                    // Convert egui's clip_rect (in points) to Vulkan scissor rect (in pixels)
                    // egui uses points (logical pixels), Vulkan uses physical pixels
                    let clip_min_x = (item.clip_rect.min.x * pixels_per_point).round() as i32;
                    let clip_min_y = (item.clip_rect.min.y * pixels_per_point).round() as i32;
                    let clip_max_x = (item.clip_rect.max.x * pixels_per_point).round() as i32;
                    let clip_max_y = (item.clip_rect.max.y * pixels_per_point).round() as i32;

                    // Clamp to window bounds
                    let clip_min_x = clip_min_x.max(0);
                    let clip_min_y = clip_min_y.max(0);
                    let clip_max_x = clip_max_x.min(window_size.width as i32);
                    let clip_max_y = clip_max_y.min(window_size.height as i32);

                    // Calculate width and height
                    let clip_width = (clip_max_x - clip_min_x).max(0) as u32;
                    let clip_height = (clip_max_y - clip_min_y).max(0) as u32;

                    // Create Vulkan scissor rect
                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D {
                            x: clip_min_x,
                            y: clip_min_y,
                        },
                        extent: vk::Extent2D {
                            width: clip_width,
                            height: clip_height,
                        },
                    };
                    draw_jobs.push(UIDrawJob {
                        descriptor_set: *self.descriptor_sets.get(&x.texture_id).unwrap(),
                        index_count: x.indices.len(),
                        index_offset: current_index_offset,
                        vertex_offset: current_vertex_offset,
                        scissor: scissor,
                    });

                    current_vertex_offset += x.vertices.len();
                    current_vertex_byte_offset += vertex_bytes.len();
                    current_index_offset += x.indices.len();
                    current_index_byte_offset += index_bytes.len();
                }
                _ => {}
            }
        }

        for job in draw_jobs {
            unsafe {
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    self.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[job.descriptor_set],
                    &[],
                );
                device.cmd_set_scissor(command_buffer, 0, &[job.scissor]);
                device.cmd_draw_indexed(
                    command_buffer,
                    job.index_count as u32,
                    1,
                    job.index_offset as u32,
                    job.vertex_offset as i32,
                    1,
                );
            }
        }
    }
    pub fn cleanup_old_textures(&mut self, textures_delta: egui::TexturesDelta) {
        for image in textures_delta.free {
            self.textures.remove(&image);
        }
    }

    pub fn handle_new_textures(
        &mut self,
        device: &ash::Device,
        textures_delta: &egui::TexturesDelta,
    ) {
        for texture in textures_delta.set.as_slice() {
            let (texture_id, image_delta) = texture;
            if image_delta.pos.is_some() {
                let texture = self.textures.get_mut(texture_id).unwrap();
                let pos = image_delta.pos.unwrap(); // [x, y]
                let size = match &image_delta.image {
                    egui::ImageData::Color(image) => image.size,
                };
                let width = size[0] as u32;
                let height = size[1] as u32;

                let pixels: Vec<u8> = match &image_delta.image {
                    egui::ImageData::Color(image) => {
                        image.pixels.iter().flat_map(|c| c.to_array()).collect()
                    }
                };
                let image_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(width, height, pixels)
                        .expect("Invalid image buffer size");
                let image = DynamicImage::ImageRgba8(image_buffer);
                texture.update_subimage(
                    image,
                    pos[0],
                    pos[1],
                    device,
                    self.allocator.clone(),
                    self.queue_family_index,
                    self.queue,
                    self.one_time_submit,
                    self.one_time_submit_pool,
                );
            } else {
                let (width, height, pixels): (u32, u32, Vec<u8>) = match &image_delta.image {
                    egui::ImageData::Color(image) => {
                        let size = image.size;
                        let pixels: Vec<u8> = image
                            .pixels
                            .iter()
                            .flat_map(|color| color.to_array()) // Color32 → [u8; 4]
                            .collect();
                        (size[0] as u32, size[1] as u32, pixels)
                    }
                };

                let image_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(width, height, pixels)
                        .expect("Invalid image buffer size");

                let dynamic_image = image::DynamicImage::ImageRgba8(image_buffer);

                let gpu_texture = GpuTexture::create_2d(
                    device,
                    self.queue_family_index,
                    self.allocator.clone(),
                    &dynamic_image,
                    self.queue,
                    self.one_time_submit,
                    self.one_time_submit_pool,
                );
                let allocate_info = vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.descriptor_pool,
                    p_set_layouts: &self.descriptor_set_layout,
                    descriptor_set_count: 1,
                    ..Default::default()
                };
                let set = unsafe { device.allocate_descriptor_sets(&allocate_info).unwrap()[0] };
                let descriptor_write = vk::WriteDescriptorSet {
                    descriptor_count: 1,
                    dst_set: set,
                    dst_binding: 0,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: gpu_texture.sampler,
                        image_view: gpu_texture.image_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                    p_buffer_info: ptr::null(),
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    dst_array_element: 0,
                    p_texel_buffer_view: ptr::null(),
                    ..Default::default()
                };
                unsafe {
                    device.update_descriptor_sets(&[descriptor_write], &[]);
                }
                self.descriptor_sets.insert(*texture_id, set);
                self.textures.insert(*texture_id, gpu_texture);
            }
        }
    }

    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        swapchain_image_views: &[vk::ImageView],
        swapchain_image_extent: vk::Extent2D,
    ) {
        self.framebuffers = create_framebuffers(
            &device,
            swapchain_image_views,
            self.renderpass,
            swapchain_image_extent,
        );
    }

    pub fn init(context: &VulkanContext) -> UIRenderer {
        let ui_ctx = egui::Context::default();
        let winit_egui_state = egui_winit::State::new(
            ui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            Some(4096),
        );
        let renderpass = create_ui_renderpass(device, swapchain_format);
        let (render_pipeline, pipeline_layout, descriptor_set_layouts) =
            create_ui_render_pipeline(device, renderpass);

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: 0xFFFFF,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            ..Default::default()
        };
        let index_buffer = unsafe {
            device
                .create_buffer(&index_buffer_create_info, None)
                .unwrap()
        };
        let vertex_buffer_create_info = vk::BufferCreateInfo {
            size: 0xFFFFF,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            ..Default::default()
        };
        let vertex_buffer = unsafe {
            device
                .create_buffer(&vertex_buffer_create_info, None)
                .unwrap()
        };
        let vertex_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

        let vertex_alloc_desc = AllocationCreateDesc {
            name: "gui vertex buffer",
            linear: true,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            requirements: vertex_memory_requirements,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let vertex_memory = allocator
            .lock()
            .unwrap()
            .allocate(&vertex_alloc_desc)
            .unwrap();

        let index_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(index_buffer) };

        let index_alloc_desc = AllocationCreateDesc {
            name: "gui index buffer",
            linear: true,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            requirements: index_memory_requirements,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let index_memory = allocator
            .lock()
            .unwrap()
            .allocate(&index_alloc_desc)
            .unwrap();
        let pool_sizes = vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 64,
        }];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 64,
            p_pool_sizes: pool_sizes.as_ptr(),
            pool_size_count: pool_sizes.len() as u32,
            ..Default::default()
        };
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        };
        unsafe {
            device
                .bind_buffer_memory(
                    vertex_buffer,
                    vertex_memory.memory(),
                    vertex_memory.offset(),
                )
                .unwrap();
            device
                .bind_buffer_memory(index_buffer, index_memory.memory(), index_memory.offset())
                .unwrap();
        }
        let framebuffers = create_framebuffers(
            device,
            swapchain_image_views,
            renderpass,
            swapchain_image_extent,
        );

        UIRenderer {
            ui_ctx,
            full_output: None,
            winit_egui_state,
            textures: HashMap::new(),
            one_time_submit_pool: one_time_pool,
            one_time_submit: one_time_command_buffer,
            queue: queue,
            allocator: allocator,
            queue_family_index: queue_family_index,
            descriptor_sets: HashMap::new(),
            pipeline: render_pipeline,
            pipeline_layout: pipeline_layout,
            renderpass: renderpass,

            index_buffer: index_buffer,
            index_memory: index_memory,
            framebuffers: framebuffers,
            vertex_buffer: vertex_buffer,
            vertex_memory: vertex_memory,
            descriptor_pool: descriptor_pool,
            // there should only be one for the ui
            descriptor_set_layout: descriptor_set_layouts[0],
        }
    }
}

fn create_framebuffers(
    device: &ash::Device,
    swapchain_image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
    swapchain_image_extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
    let mut framebuffers = Vec::new();

    for i in 0..swapchain_image_views.len() {
        let attachments = vec![swapchain_image_views[i]];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo {
                        render_pass,
                        p_attachments: attachments.as_ptr(),
                        attachment_count: attachments.len() as u32,
                        width: swapchain_image_extent.width,
                        height: swapchain_image_extent.height,
                        layers: 1,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        framebuffers.push(framebuffer);
    }
    framebuffers
}

fn create_ui_renderpass(device: &ash::Device, swapchain_format: vk::Format) -> vk::RenderPass {
    let subpass_dependancies = vec![vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::BY_REGION,
    }];
    let attachments = vec![vk::AttachmentDescription {
        format: swapchain_format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::LOAD,
        store_op: vk::AttachmentStoreOp::STORE,
        initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        stencil_load_op: vk::AttachmentLoadOp::LOAD,
        stencil_store_op: vk::AttachmentStoreOp::STORE,
        flags: vk::AttachmentDescriptionFlags::default(),
    }];
    let color_attachment_reference = vec![vk::AttachmentReference {
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        attachment: 0,
    }];

    let subpasses = vec![vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        p_color_attachments: color_attachment_reference.as_ptr(),
        color_attachment_count: color_attachment_reference.len() as u32,

        p_input_attachments: ptr::null(),
        input_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
        preserve_attachment_count: 0,
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        ..Default::default()
    }];
    let create_info = vk::RenderPassCreateInfo {
        dependency_count: subpass_dependancies.len() as u32,
        p_dependencies: subpass_dependancies.as_ptr(),
        p_attachments: attachments.as_ptr(),
        attachment_count: attachments.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        subpass_count: subpasses.len() as u32,
        ..Default::default()
    };
    unsafe { device.create_render_pass(&create_info, None).unwrap() }
}

fn create_ui_render_pipeline(
    device: &ash::Device,
    renderpass: vk::RenderPass,
) -> (
    vk::Pipeline,
    vk::PipelineLayout,
    Vec<vk::DescriptorSetLayout>,
) {
    let shaders = create_pipeline_layout_from_vert_frag(
        device,
        Path::new("shaders/egui_vert.spv"),
        Path::new("shaders/egui_frag.spv"),
    );
    let desc = &GraphicsPipelineDesc {
        shaders: &shaders.0,
        vertex_input_state: VertexInputState {
            vertex_binding_descriptions: vec![VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<egui::epaint::Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vertex_attribute_descriptions: vec![
                VertexInputAttributeDescription {
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                    location: 0,
                },
                VertexInputAttributeDescription {
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 8,
                    location: 1,
                },
                VertexInputAttributeDescription {
                    binding: 0,
                    format: vk::Format::R8G8B8A8_UNORM,
                    offset: 16,
                    location: 2,
                },
            ],
        },
        input_assembly: InputAssemblyState {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
        },
        viewport_state: None,
        raster_state: RasterState {
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            depth_bias_enable: false,
            depth_bias_clamp: 0.0,
            depth_clamp_enable: false,
            depth_bias_slope_factor: 0.0,
            depth_bias_constant_factor: 0.0,
            rasterizer_discard_enable: false,
        },
        multisample_state: MultisampleState {
            sample_mask: None,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: false,
            min_sample_shading: 0.0,
            alpha_to_one_enable: false,
            alpha_to_coverage_enable: false,
        },
        depth_stencil_state: DepthStencilState {
            stencil_test_enable: false,
            depth_test_enable: false,
            depth_compare_op: vk::CompareOp::NEVER,
            back: vk::StencilOpState::default(),
            front: vk::StencilOpState::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            depth_write_enable: false,
            depth_bounds_test_enable: false,
        },
        color_blend_state: ColorBlendState {
            logic_op: None,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            attachments: vec![PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                color_blend_op: vk::BlendOp::ADD,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,

                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            }],
        },
        dynamic_state: vec![DynamicState::VIEWPORT, DynamicState::SCISSOR],
        tesselation_state: None,
        pipeline_layout: shaders.1,
        renderpass: renderpass,
        subpass_index: 0,
    };
    (
        create_graphics_pipeline(device, desc).unwrap(),
        shaders.1,
        shaders.2,
    )
}

pub(crate) fn instant_submit_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_record: impl FnOnce(vk::CommandBuffer),
) {
    let begin_info = vk::CommandBufferBeginInfo {
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap()
    };

    command_record(command_buffer);

    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    let submits = vk::SubmitInfo {
        command_buffer_count: 1,
        p_wait_semaphores: ptr::null(),
        p_command_buffers: &command_buffer,
        p_signal_semaphores: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_dst_stage_mask: ptr::null(),
        signal_semaphore_count: 0,
        ..Default::default()
    };
    let create_info = vk::FenceCreateInfo {
        ..Default::default()
    };
    let fence = unsafe { device.create_fence(&create_info, None).unwrap() };

    unsafe { device.queue_submit(queue, &[submits], fence).unwrap() }

    let wait = [fence];
    unsafe { device.wait_for_fences(&wait, true, u64::MAX).unwrap() };
    unsafe {
        device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap()
    };
    unsafe { device.destroy_fence(fence, None) };
}

pub fn alloc_buffers(
    memory_allocator: SharedAllocator,
    buffer_count: usize,
    size: u64,
    device: &ash::Device,
    sharing: vk::SharingMode,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
    linear: bool,
    data: &[u8],
    name: &str,
) -> (Vec<vk::Buffer>, Vec<Allocation>) {
    let mut buffers = vec![];
    let mut allocations = vec![];
    for _ in 0..buffer_count {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    sharing_mode: sharing,
                    usage,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut alloc = memory_allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        alloc.mapped_slice_mut().unwrap().write_all(data).unwrap();
        unsafe {
            device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                .unwrap()
        }

        allocations.push(alloc);
        buffers.push(buffer);
    }
    (buffers, allocations)
}
