use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;
use std::ptr;

use ash::vk;
use ash::vk::{
    DynamicState, PipelineColorBlendAttachmentState, VertexInputAttributeDescription,
    VertexInputBindingDescription,
};
use bytemuck::cast_slice;
use egui::{ClippedPrimitive, TextureId, epaint};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use image::{DynamicImage, ImageBuffer, Rgba};
use winit::dpi::PhysicalSize;

use crate::renderers::world::pipelines::{
    ColorBlendState, DepthStencilState, GraphicsPipelineDesc, InputAssemblyState, MultisampleState,
    RasterState, VertexInputState, create_graphics_pipeline, create_pipeline_layout_from_vert_frag,
};
use crate::renderers::world::swapchain::SwapchainResources;
use crate::resources::GpuTexture;
use crate::vulkan::{QueueFamilyIndex, SharedAllocator, VulkanContext};

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
        context: &VulkanContext,
        textures_delta: &egui::TexturesDelta,
    ) {
        let device = &context.device;
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

                let gpu_texture = GpuTexture::create_2d(context, &dynamic_image);
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

    pub fn update_swapchain_resources(
        &mut self,
        context: &VulkanContext,
        swapchain_resources: &SwapchainResources,
    ) {
        self.framebuffers = Self::create_ui_framebuffers(
            &context.device,
            &swapchain_resources.swapchain_image_views,
            self.renderpass,
            swapchain_resources.image_size,
        );
    }

    pub fn create_ui_framebuffers(
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
    pub fn init(context: &VulkanContext, swapchain_resources: &SwapchainResources) -> UIRenderer {
        let ui_ctx = egui::Context::default();
        let winit_egui_state = egui_winit::State::new(
            ui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*context.window,
            None,
            None,
            Some(4096),
        );
        let renderpass = Self::create_ui_renderpass(
            &context.device,
            swapchain_resources.swapchain_image_format.format,
        );
        let (render_pipeline, pipeline_layout, descriptor_set_layouts) =
            Self::create_ui_render_pipeline(&context.device, renderpass);

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: 0xFFFFF,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: &context.queue_family_index,
            queue_family_index_count: 1,
            ..Default::default()
        };
        let index_buffer = unsafe {
            context
                .device
                .create_buffer(&index_buffer_create_info, None)
                .unwrap()
        };
        let vertex_buffer_create_info = vk::BufferCreateInfo {
            size: 0xFFFFF,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: &context.queue_family_index,
            queue_family_index_count: 1,
            ..Default::default()
        };
        let vertex_buffer = unsafe {
            context
                .device
                .create_buffer(&vertex_buffer_create_info, None)
                .unwrap()
        };
        let vertex_memory_requirements =
            unsafe { context.device.get_buffer_memory_requirements(vertex_buffer) };

        let vertex_alloc_desc = AllocationCreateDesc {
            name: "gui vertex buffer",
            linear: true,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            requirements: vertex_memory_requirements,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let vertex_memory = context
            .allocator
            .lock()
            .unwrap()
            .allocate(&vertex_alloc_desc)
            .unwrap();

        let index_memory_requirements =
            unsafe { context.device.get_buffer_memory_requirements(index_buffer) };

        let index_alloc_desc = AllocationCreateDesc {
            name: "gui index buffer",
            linear: true,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            requirements: index_memory_requirements,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let index_memory = context
            .allocator
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
            context
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        };
        unsafe {
            context
                .device
                .bind_buffer_memory(
                    vertex_buffer,
                    vertex_memory.memory(),
                    vertex_memory.offset(),
                )
                .unwrap();
            context
                .device
                .bind_buffer_memory(index_buffer, index_memory.memory(), index_memory.offset())
                .unwrap();
        }
        let framebuffers = Self::create_ui_framebuffers(
            &context.device,
            &swapchain_resources.swapchain_image_views,
            renderpass,
            swapchain_resources.image_size,
        );

        UIRenderer {
            ui_ctx,
            full_output: None,
            winit_egui_state,
            textures: HashMap::new(),
            one_time_submit_pool: context.one_time_submit_pool,
            one_time_submit: context.one_time_submit_buffer,
            queue: context.queue,
            allocator: context.allocator.clone(),
            queue_family_index: context.queue_family_index,
            descriptor_sets: HashMap::new(),
            pipeline: render_pipeline,
            pipeline_layout,
            renderpass,
            index_buffer,
            index_memory,
            framebuffers,
            vertex_buffer,
            vertex_memory,
            descriptor_pool,
            descriptor_set_layout: descriptor_set_layouts[0],
        }
    }
}
