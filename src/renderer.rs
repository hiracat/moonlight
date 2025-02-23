use std::{f32::consts::FRAC_PI_4, sync::Arc, time::Instant};

use ultraviolet::{projection, Mat4, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            subpass::PipelineSubpassType,
            vertex_input::{self, Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateFlags},
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDependency, SubpassDescription,
    },
    shader::ShaderStages,
    swapchain::{
        acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, AccessFlags, DependencyFlags, GpuFuture, HostAccessError, PipelineStages},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{event_loop::ActiveEventLoop, window::Window};
pub const FRAMES_IN_FLIGHT: usize = 2;

#[derive(vulkano::buffer::BufferContents, vertex_input::Vertex)]
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

impl DummyVertex {
    pub fn screen_quad() -> [DummyVertex; 6] {
        [
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [-1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, -1.0],
            },
        ]
    }
}

#[derive(vulkano::buffer::BufferContents, vertex_input::Vertex)]
#[repr(C)]
pub struct DummyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

#[derive(Default, Debug)]
pub struct Scene {
    pub camera: Camera,
    pub ambient: AmbientLight,

    pub lights: Vec<PointLight>,
    pub models: Vec<Model>,
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct Camera {
    pub view: ultraviolet::Mat4,
    pub proj: ultraviolet::Mat4,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct PointLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Default, Debug)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub model: Mat4,
    pub normals: Mat4,

    pub ubo: Option<Vec<Subbuffer<ModelUBO>>>,
    pub vertex_buffer: Option<Subbuffer<[Vertex]>>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
pub struct ModelUBO {
    pub model: Mat4,
    pub normal: Mat4,
}
impl Model {
    pub fn update_normals(&mut self) {
        self.normals = self.model.inversed().transposed();
    }
}

pub struct Renderer {
    pub window: Arc<Window>,
    pub recreate_swapchain: bool,
    pub memory_allocator: Arc<dyn MemoryAllocator>,
    pub start_time: Instant,

    surface: Arc<Surface>,
    aspect_ratio: f32,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    dummy_verts: Subbuffer<[DummyVertex]>,
    current_frame: usize,
    frame_counter: u128,

    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,

    viewport: Viewport,

    framebuffers: Vec<Arc<Framebuffer>>,

    view_proj_buffers: Vec<Subbuffer<Camera>>,
    ambient_buffers: Vec<Subbuffer<AmbientLight>>,
    directional_buffers: Vec<Vec<Subbuffer<PointLight>>>,
    color_buffers: Vec<Arc<ImageView>>,
    normal_buffers: Vec<Arc<ImageView>>,

    view_proj_sets: Vec<Arc<PersistentDescriptorSet>>,
    model_sets: Vec<Arc<PersistentDescriptorSet>>,
    directional_sets: Vec<Vec<Arc<PersistentDescriptorSet>>>,
    ambient_sets: Vec<Arc<PersistentDescriptorSet>>,

    frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}

impl Renderer {
    pub fn draw(&mut self, scene: &mut Scene) {
        self.frames_resources_free[self.current_frame]
            .as_mut()
            .take()
            .unwrap()
            .cleanup_finished();

        self.window.request_redraw();
        let window_size = self.window.inner_size();
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        self.frame_counter += 1;
        dbg!(self.frame_counter);

        if self.recreate_swapchain {
            // Use the new dimensions of the window.
            dbg!(self.recreate_swapchain);
            let new_images;
            (self.swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to create swapchain");
            // Because framebuffers contains a reference to the old swapchain, we need to
            // recreate framebuffers as well.
            (self.framebuffers, self.color_buffers, self.normal_buffers) = create_framebuffers(
                &new_images,
                &self.render_pass,
                self.memory_allocator.clone(),
            );
            self.viewport.extent = self.window.inner_size().into();
            self.aspect_ratio = self.viewport.extent[0] as f32 / self.viewport.extent[1] as f32;

            let mut view_proj_buffers: Vec<Subbuffer<Camera>> = vec![];
            for _ in 0..self.swapchain.image_count() {
                view_proj_buffers.push(
                    Buffer::from_data(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        Camera {
                            view: ultraviolet::Mat4::look_at(
                                Vec3::new(0.0, 0.0, 5.0), // Move camera back slightly
                                Vec3::new(0.0, 0.0, 0.0), // Look at origin
                                Vec3::new(0.0, 1.0, 0.0), // Up vector
                            ),
                            proj: projection::perspective_vk(
                                FRAC_PI_4,
                                self.aspect_ratio,
                                0.1,
                                10.0,
                            ),
                        },
                    )
                    .unwrap(),
                );
            }

            (
                self.view_proj_sets,
                self.model_sets,
                self.directional_sets,
                self.ambient_sets,
            ) = create_descriptor_sets(
                &self.device,
                &self.deferred_pipeline,
                &self.directional_pipeline,
                &self.ambient_pipeline,
                &view_proj_buffers,
                &scene.models,
                &self.ambient_buffers,
                &self.directional_buffers,
                &self.color_buffers,
                &self.normal_buffers,
                new_images.len(),
            );

            self.recreate_swapchain = false;
        }

        // NOTE: RENDERING START

        let (image_index, is_suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if is_suboptimal {
            self.recreate_swapchain = true;
        }
        let image_index = image_index as usize;

        dbg!(self.current_frame);

        for model in &mut scene.models {
            model.update_normals()
        }

        for model in &mut scene.models {
            let data = ModelUBO {
                model: model.model,
                normal: model.normals,
            };
            let buffer = &mut model.ubo.as_mut().unwrap();
            loop {
                match buffer[self.current_frame].write() {
                    Ok(mut write) => {
                        *write = data;
                        break;
                    }
                    Err(error) => match error {
                        HostAccessError::AccessConflict(_) => {
                            println!("failed loop");
                            self.frames_resources_free[self.current_frame]
                                .as_mut()
                                .take()
                                .unwrap()
                                .cleanup_finished();
                        }
                        _ => {
                            panic!("failed to write to model buffer");
                        }
                    },
                }
            }
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Before we can draw, we have to *enter a render pass*.
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    // Only attachments that have `AttachmentLoadOp::Clear` are provided
                    // others should use none as their value
                    clear_values: vec![
                        Some([0.1, 0.1, 0.1, 1.0].into()),
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some(1.0.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index].clone())
                },
                SubpassBeginInfo {
                    // The contents of the first (and only) subpass. This can be either
                    // `Inline` or `SecondaryCommandBuffers`
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.deferred_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.deferred_pipeline.layout().clone(),
                0,
                (
                    self.view_proj_sets[image_index].clone(),
                    self.model_sets[self.current_frame].clone(),
                ),
            )
            .unwrap();

        for model in &scene.models {
            builder
                .bind_vertex_buffers(0, model.vertex_buffer.as_ref().unwrap().clone())
                .unwrap()
                .bind_index_buffer(model.index_buffer.as_ref().unwrap().clone())
                .unwrap()
                .draw_indexed(
                    model.index_buffer.as_ref().unwrap().len() as u32,
                    1,
                    0,
                    0,
                    0,
                )
                .unwrap();
        }
        builder
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        dbg!(self.dummy_verts.len());
        builder
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .unwrap();
        for i in 0..self.directional_sets[image_index].len() {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.directional_pipeline.layout().clone(),
                    0,
                    self.directional_sets[image_index][i].clone(),
                )
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }

        builder
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                self.ambient_sets[image_index].clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();

        builder.end_render_pass(SubpassEndInfo::default()).unwrap();

        let command_buffer = builder.build().unwrap();
        let future = self.frames_resources_free[self.current_frame]
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            // dosent present imediately but submits a present command to
            // the queue, so the triangle will finish rendering
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index as u32,
                ),
            )
            .then_signal_fence_and_flush(); // signal will tell gpu to finish and use fence

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.frames_resources_free[self.current_frame] = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.frames_resources_free[self.current_frame] =
                    Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                self.frames_resources_free[self.current_frame] =
                    Some(Box::new(sync::now(self.device.clone())).boxed());
                println!("Failed to flush future: {:?}", e);
            }
        }
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    pub fn init(event_loop: &ActiveEventLoop, scene: &mut Scene) -> Self {
        let start_time = Instant::now();
        let instance = create_instance(event_loop);
        let window = create_window(event_loop);
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, &surface, &device_extensions);
        let (device, queue) =
            create_device(physical_device, queue_family_index, &device_extensions);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: FRAMES_IN_FLIGHT,
                ..Default::default()
            },
        ));
        let window_size = window.inner_size();
        let aspect_ratio = {
            let windowsize: [f32; 2] = window_size.into();
            windowsize[0] / windowsize[1]
        };

        for model in &mut scene.models {
            let vertex_buffer: Subbuffer<[Vertex]>;
            let index_buffer: Subbuffer<[u32]>;
            vertex_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                model.vertices.clone(),
            )
            .unwrap();
            model.vertex_buffer = Some(vertex_buffer);
            index_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                model.indices.clone(),
            )
            .unwrap();
            model.index_buffer = Some(index_buffer);
        }

        let dummy_verts = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DummyVertex::screen_quad(),
        )
        .unwrap();

        let (swapchain, swapchain_images) = create_swapchain(&device, &window, &surface);

        let render_pass = create_renderpass(&device, swapchain.image_format());
        let deferred_pass = Arc::new(Subpass::from(render_pass.clone(), 0).unwrap());
        let lighting_pass = Arc::new(Subpass::from(render_pass.clone(), 1).unwrap());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let (framebuffers, color_buffers, normal_buffers) =
            create_framebuffers(&swapchain_images, &render_pass, memory_allocator.clone());
        let mut view_proj_buffers: Vec<Subbuffer<Camera>> = vec![];
        let mut ambient_buffers: Vec<Subbuffer<AmbientLight>> = vec![];
        let mut directional_buffers: Vec<Vec<Subbuffer<PointLight>>> = vec![];

        let window_size = window.inner_size();
        let ratio = window_size.width as f32 / window_size.height as f32;
        for _ in 0..swapchain_images.len() {
            view_proj_buffers.push(
                Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    Camera {
                        view: ultraviolet::Mat4::look_at(
                            Vec3::new(0.0, 0.0, 3.0), // Move camera back slightly
                            Vec3::new(0.0, 0.0, 0.0), // Look at origin
                            Vec3::new(0.0, 1.0, 0.0), // Up vector
                        ),
                        proj: projection::perspective_vk(FRAC_PI_4, ratio, 0.1, 5.0),
                    },
                )
                .unwrap(),
            );
        }

        for model in &mut scene.models {
            let mut model_buffers: Vec<Subbuffer<ModelUBO>> = vec![];
            for _ in 0..FRAMES_IN_FLIGHT {
                model_buffers.push(
                    Buffer::new_sized(
                        memory_allocator.clone(),
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
            model.ubo = Some(model_buffers);
        }

        for _ in 0..swapchain_images.len() {
            ambient_buffers.push(
                Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    AmbientLight {
                        color: [1.0, 1.0, 1.0],
                        intensity: 0.1,
                    },
                )
                .unwrap(),
            );
        }

        for _ in 0..swapchain_images.len() {
            directional_buffers.push(vec![
                Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    PointLight {
                        position: [-4.0, 0.0, 4.0, 1.0],
                        color: [1.0, 0.0, 0.0],
                    },
                )
                .unwrap(),
                Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    PointLight {
                        position: [0.0, -4.0, 1.0, 1.0],
                        color: [0.0, 1.0, 0.0],
                    },
                )
                .unwrap(),
                Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    PointLight {
                        position: [4.0, -2.0, 1.0, 1.0],
                        color: [0.0, 0.0, 1.0],
                    },
                )
                .unwrap(),
            ])
        }

        let (deferred_pipeline, directional_pipeline, ambient_pipeline) =
            create_graphics_pipelines(&device, &deferred_pass, &lighting_pass);
        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };
        let mut frames_resources_free = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            frames_resources_free.push(Some(vulkano::sync::now(device.clone()).boxed()));
        }

        let (view_proj_sets, model_sets, directional_sets, ambient_sets) = create_descriptor_sets(
            &device,
            &deferred_pipeline,
            &directional_pipeline,
            &ambient_pipeline,
            &view_proj_buffers,
            &scene.models,
            &ambient_buffers,
            &directional_buffers,
            &color_buffers,
            &normal_buffers,
            swapchain_images.len(),
        );
        Renderer {
            recreate_swapchain: false,
            current_frame: 0,
            frame_counter: 0,
            swapchain,
            framebuffers,
            frames_resources_free,
            render_pass,
            memory_allocator,
            viewport,
            surface,
            command_buffer_allocator,
            dummy_verts,
            device,
            queue,
            start_time,
            window: window.clone(),
            aspect_ratio,

            color_buffers,
            normal_buffers,

            view_proj_buffers,
            ambient_buffers,
            directional_buffers,

            deferred_pipeline,
            directional_pipeline,
            ambient_pipeline,

            view_proj_sets,
            model_sets,
            directional_sets,
            ambient_sets,
        }
    }
}

fn create_swapchain(
    device: &Arc<Device>,
    window: &Arc<Window>,
    surface: &Arc<Surface>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let window_size = window.inner_size();
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(surface, Default::default())
        .unwrap();
    let (image_format, _) = device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()[0]; // take the first available format
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            // always keep a min image count of 2 bc required for fullscreen, and have 1-2 more
            // than ur frames in flight
            min_image_count: surface_capabilities
                .min_image_count
                .max(1 + FRAMES_IN_FLIGHT as u32),
            image_format,
            // always use the window_size bc some drivers dont report in
            // swapchain.currentextent
            image_extent: window_size.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            // change how window and alpha relate, if window is transparent or opaque
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            present_mode: PresentMode::Mailbox,
            ..Default::default()
        },
    )
    .unwrap()
}
pub fn create_descriptor_sets(
    device: &Arc<Device>,
    deferred_pipeline: &Arc<GraphicsPipeline>,
    directional_pipeline: &Arc<GraphicsPipeline>,
    ambient_pipeline: &Arc<GraphicsPipeline>,

    view_proj_buffers: &[Subbuffer<Camera>],
    models: &[Model],
    ambient_buffers: &[Subbuffer<AmbientLight>],
    directional_buffers: &[Vec<Subbuffer<PointLight>>],
    color_buffer: &[Arc<ImageView>],
    normal_buffer: &[Arc<ImageView>],
    swapchain_image_count: usize,
) -> (
    Vec<Arc<PersistentDescriptorSet>>,      // view projection
    Vec<Arc<PersistentDescriptorSet>>,      // model
    Vec<Vec<Arc<PersistentDescriptorSet>>>, // directional
    Vec<Arc<PersistentDescriptorSet>>,      // ambient
) {
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo {
            set_count: 2,
            ..Default::default()
        },
    );

    let view_proj_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
    let model_layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();
    let directional_layout = directional_pipeline.layout().set_layouts().get(0).unwrap();
    let ambient_layout = ambient_pipeline.layout().set_layouts().get(0).unwrap();

    let mut view_proj_sets = vec![];
    let mut model_sets = vec![];
    let mut directional_sets = vec![];
    let mut ambient_sets = vec![];

    for model in models {
        for i in 0..FRAMES_IN_FLIGHT {
            let model_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                model_layout.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    model.ubo.as_ref().unwrap()[i].clone(),
                )],
                [],
            )
            .unwrap();
            model_sets.push(model_set);
        }
    }

    for i in 0..swapchain_image_count {
        let view_proj_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            view_proj_layout.clone(),
            [WriteDescriptorSet::buffer(0, view_proj_buffers[i].clone())],
            [],
        )
        .unwrap();
        view_proj_sets.push(view_proj_set);

        let mut directional_subset = vec![];
        for j in 0..directional_buffers[0].len() {
            let directional_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                directional_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer[i].clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer[i].clone()),
                    WriteDescriptorSet::buffer(2, directional_buffers[i][j].clone()),
                ],
                [],
            )
            .unwrap();
            directional_subset.push(directional_set);
        }
        directional_sets.push(directional_subset);

        let ambient_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            ambient_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_buffer[i].clone()),
                WriteDescriptorSet::image_view(1, normal_buffer[i].clone()),
                WriteDescriptorSet::buffer(2, ambient_buffers[i].clone()),
            ],
            [],
        )
        .unwrap();
        ambient_sets.push(ambient_set);
    }
    (view_proj_sets, model_sets, directional_sets, ambient_sets)
}
pub fn create_framebuffers(
    swapchain_images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    allocator: Arc<dyn MemoryAllocator>,
) -> (
    Vec<Arc<Framebuffer>>,
    Vec<Arc<ImageView>>, // color
    Vec<Arc<ImageView>>, // normal
) {
    let mut framebuffers = vec![];
    let mut color_buffers = vec![];
    let mut normal_buffers = vec![];
    let extent = swapchain_images[0].extent();
    for swapchain_image in swapchain_images {
        let depth_image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                extent,
                format: Format::D32_SFLOAT,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let color_image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                extent,
                format: Format::A2B10G10R10_UNORM_PACK32,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let normal_image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                extent,
                format: Format::R16G16B16A16_SFLOAT,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let final_color_view = ImageView::new_default(swapchain_image.clone()).unwrap(); // 0: Final color
        let color_view = ImageView::new_default(color_image.clone()).unwrap(); // 1: G-Buffer color
        let normal_view = ImageView::new_default(normal_image.clone()).unwrap(); // 2: G-Buffer normal
        let depth_view = ImageView::new_default(depth_image.clone()).unwrap(); // 3: Depth

        color_buffers.push(color_view.clone());
        normal_buffers.push(normal_view.clone());

        framebuffers.push(
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![final_color_view, color_view, normal_view, depth_view],
                    ..Default::default()
                },
            )
            .unwrap(),
        );
    }

    (framebuffers, color_buffers, normal_buffers)
}

fn create_renderpass(device: &Arc<Device>, swapchain_image_format: Format) -> Arc<RenderPass> {
    RenderPass::new(
        device.clone(),
        RenderPassCreateInfo {
            attachments: vec![
                // final color attachment
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    format: swapchain_image_format,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::PresentSrc,
                    ..Default::default()
                },
                // color attachment (gbuffer)
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // normal attachment (gbuffer)
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // depth attachment(gbuffer)
                AttachmentDescription {
                    format: Format::D32_SFLOAT,
                    samples: SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    initial_layout: ImageLayout::DepthAttachmentOptimal,
                    final_layout: ImageLayout::DepthAttachmentOptimal,
                    ..Default::default()
                },
            ],
            subpasses: vec![
                // geometry pass
                SubpassDescription {
                    color_attachments: vec![
                        // gcolor
                        Some(AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        }),
                        // gnormal
                        Some(AttachmentReference {
                            attachment: 2,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        }),
                    ],
                    depth_stencil_attachment: Some(AttachmentReference {
                        attachment: 3,
                        layout: ImageLayout::DepthAttachmentOptimal,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // lighting/final pass
                SubpassDescription {
                    input_attachments: vec![
                        // gcolor attachment
                        Some(AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::ShaderReadOnlyOptimal,
                            ..Default::default()
                        }),
                        // gnormal
                        Some(AttachmentReference {
                            attachment: 2,
                            layout: ImageLayout::ShaderReadOnlyOptimal,
                            ..Default::default()
                        }),
                    ],
                    color_attachments: vec![Some(AttachmentReference {
                        attachment: 0, // Color attachment
                        layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    })],
                    ..Default::default()
                },
            ],
            dependencies: vec![
                // Transition from geometry pass (subpass 0) to lighting pass (subpass 1)
                SubpassDependency {
                    src_subpass: Some(0),
                    dst_subpass: Some(1),
                    src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT, // Wait for writes to finish
                    dst_stages: PipelineStages::FRAGMENT_SHADER, // Transition before fragment shader reads src_access: AccessFlags::COLOR_ATTACHMENT_WRITE, // Geometry pass writes
                    dst_access: AccessFlags::INPUT_ATTACHMENT_READ, // Lighting pass reads
                    dependency_flags: DependencyFlags::BY_REGION,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    )
    .unwrap()
}
fn create_graphics_pipelines(
    device: &Arc<Device>,
    deferred_subpass: &Subpass,
    lighting_subpass: &Subpass,
) -> (
    Arc<GraphicsPipeline>,
    Arc<GraphicsPipeline>,
    Arc<GraphicsPipeline>,
) {
    mod defered_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/defered_vert.glsl",
        }
    }
    mod defered_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/defered_frag.glsl"
        }
    }
    mod lighting_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/directionalvert.glsl"
        }
    }
    mod lighting_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/directionalfrag.glsl"
        }
    }
    mod ambient_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/ambient_vert.glsl",
        }
    }
    mod ambient_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/ambient_frag.glsl",
        }
    }

    let deferred_vert = defered_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let deferred_frag = defered_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let directional_vert = lighting_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let directional_frag = lighting_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let ambient_vert = ambient_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let ambient_frag = ambient_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state = Vertex::per_vertex()
        .definition(&deferred_vert.info().input_interface)
        .unwrap();

    let dummy_vertex_input_state = DummyVertex::per_vertex()
        .definition(&directional_vert.info().input_interface)
        .unwrap();

    let deferred_stagse = [
        PipelineShaderStageCreateInfo::new(deferred_vert),
        PipelineShaderStageCreateInfo::new(deferred_frag),
    ];
    let directional_stages = [
        PipelineShaderStageCreateInfo::new(directional_vert),
        PipelineShaderStageCreateInfo::new(directional_frag),
    ];
    let ambient_stages = [
        PipelineShaderStageCreateInfo::new(ambient_vert),
        PipelineShaderStageCreateInfo::new(ambient_frag),
    ];

    // We must now create a **pipeline layout** object, which describes the locations and
    // types of descriptor sets and push constants used by the shaders in the pipeline.
    //
    // Multiple pipelines can share a common layout object, which is more efficient. The
    // shaders in a pipeline must use a subset of the resources described in its pipeline
    // layout, but the pipeline layout is allowed to contain resources that are not present
    // in the shaders; they can be used by shaders in other pipelines that share the same
    // layout. Thus, it is a good idea to design shaders so that many pipelines have common
    // resource locations, which allows them to share pipeline layouts.

    let deferred_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::VERTEX,
                            descriptor_type: DescriptorType::UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    )]
                    .into(),
                    ..Default::default()
                },
                DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::VERTEX,
                            descriptor_type: DescriptorType::UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    )]
                    .into(),
                    ..Default::default()
                },
            ],
            push_constant_ranges: vec![],
            flags: PipelineLayoutCreateFlags::empty(),
        }
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let directional_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo {
            set_layouts: vec![DescriptorSetLayoutCreateInfo {
                bindings: [
                    (
                        0, // Binding 0 for color input attachment
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::InputAttachment,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::InputAttachment,
                            )
                        },
                    ),
                    (
                        1, // Binding 2 for normals input attachment
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::InputAttachment,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::InputAttachment,
                            )
                        },
                    ),
                    (
                        2, // directional light data
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                ]
                .into(),
                ..Default::default()
            }],
            push_constant_ranges: vec![],
            flags: PipelineLayoutCreateFlags::empty(),
        }
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let ambient_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo {
            set_layouts: vec![DescriptorSetLayoutCreateInfo {
                bindings: [
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::InputAttachment,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                    (
                        1,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::InputAttachment,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                    (
                        2,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            descriptor_type: DescriptorType::UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                ]
                .into(),
                ..Default::default()
            }],
            push_constant_ranges: vec![],
            flags: PipelineLayoutCreateFlags::empty(),
        }
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let cull_mode = CullMode::None;
    let front_face = FrontFace::Clockwise;
    // We have to indicate which subpass of which render pass this pipeline is going to be
    // used in. The pipeline will only be usable from this particular subpass.
    let deferred_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: deferred_stagse.into_iter().collect(),
            // How vertex data is read from the vertex buffers into the vertex shader.
            vertex_input_state: Some(vertex_input_state.clone()),
            // How vertices are arranged into primitive shapes. The default primitive shape
            // is a triangle.
            input_assembly_state: Some(InputAssemblyState::default()),
            // How primitives are transformed and clipped to fit the framebuffer. We use a
            // resizable viewport, set to draw over the entire window.
            viewport_state: Some(ViewportState::default()),
            // How polygons are culled and converted into a raster of pixels. The default
            // value does not perform any culling.
            rasterization_state: Some(RasterizationState {
                cull_mode,
                front_face,
                ..Default::default()
            }),
            // How multiple fragment shader samples are converted to a single pixel value.
            // The default value does not perform any multisampling.
            multisample_state: Some(MultisampleState::default()),
            // How pixel values are combined with the values already present in the
            // framebuffer. The default value overwrites the old value with the new one,
            // without any blending.
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                deferred_subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                stencil: None,
                ..Default::default()
            }),

            // Dynamic states allows us to specify parts of the pipeline settings when
            // recording the command buffer, before we perform drawing. Here, we specify
            // that the viewport should be dynamic.
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),

            subpass: Some(PipelineSubpassType::BeginRenderPass(
                deferred_subpass.clone(),
            )),
            ..GraphicsPipelineCreateInfo::layout(deferred_layout.clone())
        },
    )
    .expect("failed to create defered graphics pipeline");

    let directional_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: directional_stages.into_iter().collect(),
            vertex_input_state: Some(dummy_vertex_input_state.clone()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode,
                front_face,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                lighting_subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_blend_op: BlendOp::Add,
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Max,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                    }),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),

            subpass: Some(PipelineSubpassType::BeginRenderPass(
                lighting_subpass.clone(),
            )),
            ..GraphicsPipelineCreateInfo::layout(directional_layout)
        },
    )
    .expect("failed to create directional graphics pipeline");

    let ambient_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: ambient_stages.into_iter().collect(),
            vertex_input_state: Some(dummy_vertex_input_state.clone()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode,
                front_face,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                lighting_subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_blend_op: BlendOp::Add,
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Max,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                    }),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                lighting_subpass.clone(),
            )),
            ..GraphicsPipelineCreateInfo::layout(ambient_layout)
        },
    )
    .expect("failed to create lighting graphics pipeline");

    (deferred_pipeline, directional_pipeline, ambient_pipeline)
}
fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
}

fn create_instance(event_loop: &ActiveEventLoop) -> Arc<Instance> {
    let library =
        VulkanLibrary::new().expect("failed to load library, please install vulkan drivers");
    let required_extensions = Surface::required_extensions(event_loop);

    let validation_layer = "VK_LAYER_KHRONOS_validation";

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enabled_layers: vec![validation_layer.to_string()],
            ..Default::default()
        },
    )
    .expect("failed to create instance")
}
fn create_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    required_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|physical_device| {
            physical_device
                .supported_extensions()
                .contains(required_extensions)
        })
        .filter_map(|physical_device| {
            physical_device
                .queue_family_properties()
                .iter() // iterate over all available queues for each device
                .enumerate() // returns an iterator of type (physicaldevice, u32)
                .position(|(index, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS)
                        && physical_device
                            .surface_support(index as u32, surface)
                            .unwrap()
                })
                .map(|index| (physical_device, index as u32))
        })
        .min_by_key(
            |(physical_device, _)| match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            },
        )
        .expect("no qualified gpu")
}
fn create_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    required_extensions: &DeviceExtensions,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_features: Features {
                separate_depth_stencil_layouts: true, // MUST ENABLE THIS
                ..Default::default()
            },
            enabled_extensions: *required_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    (device, queue)
}
