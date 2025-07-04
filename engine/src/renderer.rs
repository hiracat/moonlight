#![allow(clippy::cast_possible_truncation)]
#![allow(dead_code)]
use std::{sync::Arc, time::Instant};

use ultraviolet::{projection, Rotor3, Vec3};
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
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwnedVulkanObject, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount},
    instance::{debug::ValidationFeatureEnable, Instance, InstanceCreateInfo},
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
        acquire_next_image, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, AccessFlags, DependencyFlags, GpuFuture, HostAccessError, PipelineStages},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

#[derive(Default)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Rotor3,
    pub fov_rads: f32,
    pub near: f32,
    pub far: f32,
    pub pitch: f32,
    pub yaw: f32,

    aspect_ratio: f32,

    u_buffer: Option<Vec<Subbuffer<CameraUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}
impl Camera {
    pub fn new(position: Vec3, fov: f32, near: f32, far: f32, window: &Arc<Window>) -> Self {
        let size: [f32; 2] = window.inner_size().into();
        let aspect_ratio = size[0] / size[1];
        let fov_rads = fov * (std::f32::consts::PI / 180.0);
        let rotation = Rotor3::identity();

        Self {
            pitch: 0.0,
            yaw: 0.0,
            position,
            rotation,
            fov_rads,
            near,
            far,
            aspect_ratio,

            u_buffer: None,
            descriptor_set: None,
        }
    }
    fn get_ubo_data(&self) -> CameraUBO {
        let forward = (-Vec3::unit_z()).rotated_by(self.rotation);
        let look_at = self.position + forward;
        CameraUBO {
            view: ultraviolet::Mat4::look_at(
                self.position,
                look_at,
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            proj: projection::perspective_vk(self.fov_rads, self.aspect_ratio, self.near, self.far),
        }
    }
    fn populate_u_buffer(
        &mut self,
        swapchain_image_count: u32,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) {
        let mut camera_buffers = vec![];
        for _ in 0..swapchain_image_count {
            camera_buffers.push(
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
                    self.get_ubo_data(),
                )
                .unwrap(),
            );
        }
        self.u_buffer = Some(camera_buffers);
    }
}
#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
struct CameraUBO {
    view: ultraviolet::Mat4,
    proj: ultraviolet::Mat4,
}

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

pub const FRAMES_IN_FLIGHT: usize = 1;

#[derive(vulkano::buffer::BufferContents, vertex_input::Vertex)]
#[repr(C)]
pub struct DummyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}
impl DummyVertex {
    pub const fn screen_quad() -> [DummyVertex; 6] {
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

pub struct Renderer {
    pub window: Arc<Window>,
    pub recreate_swapchain: bool,
    pub memory_allocator: Arc<dyn MemoryAllocator>,
    pub start_time: Instant,
    pub pipelines: Vec<Arc<GraphicsPipeline>>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pub(crate) queue: Arc<Queue>,

    surface: Arc<Surface>,
    aspect_ratio: f32,
    pub(crate) device: Arc<Device>,
    dummy_verts: Subbuffer<[DummyVertex]>,
    current_frame: usize,
    frame_counter: u128,

    pub(crate) swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,

    viewport: Viewport,

    framebuffers: Vec<Arc<Framebuffer>>,

    pub(crate) color_buffers: Vec<Arc<ImageView>>,
    pub(crate) normal_buffers: Vec<Arc<ImageView>>,
    pub(crate) position_buffers: Vec<Arc<ImageView>>,

    frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}

impl Renderer {
    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, world: &mut World) {
        self.frames_resources_free[self.current_frame]
            .as_mut()
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
            (
                self.framebuffers,
                self.color_buffers,
                self.normal_buffers,
                self.position_buffers,
            ) = create_framebuffers(
                &new_images,
                &self.render_pass,
                self.memory_allocator.clone(),
            );
            self.viewport.extent = self.window.inner_size().into();
            self.aspect_ratio = self.viewport.extent[0] / self.viewport.extent[1];

            let camera = world
                .resource_get_mut::<Camera>()
                .expect("camera should definately exist or something is very wrong");
            camera.aspect_ratio = self.aspect_ratio;
            camera.populate_u_buffer(self.swapchain.image_count(), self.memory_allocator.clone());

            write_descriptor_sets(
                world,
                self.descriptor_set_allocator.clone(),
                &self.pipelines,
                &self.color_buffers,
                &self.normal_buffers,
                &self.position_buffers,
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

        {
            let camera = world
                .resource_get_mut::<Camera>()
                .expect("should have a camera resource");
            let data = camera.get_ubo_data();
            let buffer = camera.u_buffer.as_mut().unwrap();
            for _ in 0..5 {
                match buffer[image_index].write() {
                    Ok(mut write) => {
                        *write = data;
                        break;
                    }
                    Err(error) => match error {
                        HostAccessError::AccessConflict(_) => {
                            println!("camera failed loop");
                            dbg!(error);
                            self.frames_resources_free[image_index]
                                .as_mut()
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

        {
            let entities = world.query_entities::<PointLight>();
            for entity in entities {
                let transform;
                if !world.has_component::<Transform>(entity) {
                    eprintln!(
                        "WARNING: light {:?} does not have transform component, giving identity matrix", entity
                    );
                    transform = Transform::new();
                } else {
                    transform = *world
                        .component_get_mut::<Transform>(entity)
                        .expect("what the fuck, impossible");
                }
                let point_light = world
                    .component_get_mut::<PointLight>(entity)
                    .expect("what the fuck, impossible");

                let data = point_light.as_point_ubo(transform);

                for _ in 0..5 {
                    match point_light.u_buffers[image_index].write() {
                        Ok(mut write) => {
                            *write = data;
                            break;
                        }
                        Err(error) => match error {
                            HostAccessError::AccessConflict(_) => {
                                println!("light failed loop");
                                self.frames_resources_free[self.current_frame]
                                    .as_mut()
                                    .unwrap()
                                    .cleanup_finished();
                            }
                            _ => {
                                panic!("failed to write to light buffer");
                            }
                        },
                    }
                }
            }
        }

        {
            let entities = world.query_entities::<Model>();
            for entity in entities {
                let data;
                if !world.has_component::<Transform>(entity) {
                    eprintln!(
                        "WARNING: model {:?} does not have transform component, giving identity matrix", entity
                    );
                    data = ModelUBO::new(Vec3::zero(), Rotor3::identity());
                } else {
                    let transform = world
                        .component_get_mut::<Transform>(entity)
                        .expect("what the fuck, impossible");
                    data = transform.as_model_ubo();
                }
                let model = world
                    .component_get_mut::<Model>(entity)
                    .expect("what the fuck, impossible");
                for _ in 0..5 {
                    match model.u_buffer[self.current_frame].write() {
                        Ok(mut write) => {
                            *write = data;
                            break;
                        }
                        Err(error) => match error {
                            HostAccessError::AccessConflict(_) => {
                                println!("model failed loop");
                                self.frames_resources_free[self.current_frame]
                                    .as_mut()
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
                        Some([0.0, 0.0, 0.0, 1.0].into()),
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
            .bind_pipeline_graphics(self.pipelines[0].clone())
            .unwrap();

        let camera_descripor_set = world
            .resource_get::<Camera>()
            .unwrap()
            .descriptor_set
            .clone();
        let entities = world.query_mut::<Model>();
        for (_entity, model) in entities {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipelines[0].layout().clone(),
                    0,
                    (
                        camera_descripor_set.as_ref().unwrap()[image_index].clone(),
                        model.descriptor_set[self.current_frame].clone(),
                    ),
                )
                .unwrap()
                .bind_vertex_buffers(0, model.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(model.index_buffer.clone())
                .unwrap()
                .draw_indexed(model.index_buffer.len() as u32, 1, 0, 0, 0)
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

        builder
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap();

        builder
            .bind_pipeline_graphics(self.pipelines[1].clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipelines[1].layout().clone(),
                0,
                world
                    .resource_get::<DirectionalLight>()
                    .unwrap()
                    .descriptor_set
                    .as_ref()
                    .unwrap()[image_index]
                    .clone(),
            )
            .unwrap()
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();

        builder
            .bind_pipeline_graphics(self.pipelines[2].clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipelines[2].layout().clone(),
                0,
                world
                    .resource_get::<AmbientLight>()
                    .unwrap()
                    .descriptor_set
                    .as_ref()
                    .unwrap()[image_index]
                    .clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();

        builder
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_pipeline_graphics(self.pipelines[3].clone())
            .unwrap();

        let entities = world.query_mut::<PointLight>();
        for (_entity, light) in entities {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipelines[3].layout().clone(),
                    0,
                    light.descriptor_set[image_index].clone(),
                )
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }

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
                println!("Failed to flush future: {e}");
            }
        }
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(event_loop: &ActiveEventLoop, world: &mut World, window: &Arc<Window>) -> Self {
        let start_time = Instant::now();
        let instance = create_instance(event_loop);
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
        let window_size: [f32; 2] = window.inner_size().into();
        let aspect_ratio = window_size[0] / window_size[1];

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

        let (swapchain, swapchain_images) = create_swapchain(&device, window, &surface);
        for i in 0..swapchain_images.len() {
            let _ = swapchain_images[i]
                .set_debug_utils_object_name(Some(&format!("swapchain image{}", i)));
        }

        let render_pass = create_renderpass(&device, swapchain.image_format());
        let deferred_pass = Arc::new(Subpass::from(render_pass.clone(), 0).unwrap());
        let lighting_pass = Arc::new(Subpass::from(render_pass.clone(), 1).unwrap());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let (framebuffers, color_buffers, normal_buffers, position_buffers) =
            create_framebuffers(&swapchain_images, &render_pass, memory_allocator.clone());

        let window_size = window.inner_size();

        let mut directional_buffers = vec![];
        if let Some(directional) = world.resource_get_mut::<DirectionalLight>() {
            for _ in 0..swapchain_images.len() {
                directional_buffers.push(
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
                        DirectionalLightUBO {
                            color: directional.color,
                            position: directional.position,
                        },
                    )
                    .unwrap(),
                );
            }
            dbg!(&directional_buffers);
            directional.u_buffer = Some(directional_buffers);
        } else {
            eprintln!("no  directional resource, skipping");
        }

        let mut ambient_buffers = vec![];
        if let Some(ambient) = world.resource_get_mut::<AmbientLight>() {
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
                        AmbientLightUBO {
                            color: ambient.color,
                            intensity: ambient.intensity,
                        },
                    )
                    .unwrap(),
                );
            }
            dbg!(&ambient_buffers);
            ambient.u_buffer = Some(ambient_buffers);
        }

        let camera = world
            .resource_get_mut::<Camera>()
            .expect("should create camera resource");
        camera.populate_u_buffer(swapchain_images.len() as u32, memory_allocator.clone());

        let pipelines = create_graphics_pipelines(&device, &deferred_pass, &lighting_pass);

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

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                set_count: 2,
                ..Default::default()
            },
        ));
        write_descriptor_sets(
            world,
            descriptor_set_allocator.clone(),
            &pipelines,
            &color_buffers,
            &normal_buffers,
            &position_buffers,
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
            descriptor_set_allocator: descriptor_set_allocator.into(),

            color_buffers,
            normal_buffers,
            position_buffers,

            pipelines,
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
        .surface_capabilities(surface, SurfaceInfo::default())
        .unwrap();

    let (image_format, _) = device
        .physical_device()
        .surface_formats(surface, SurfaceInfo::default())
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
            present_mode: PresentMode::Fifo,
            ..Default::default()
        },
    )
    .unwrap()
}
#[allow(clippy::too_many_lines)]
pub fn write_descriptor_sets(
    world: &mut World,

    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    graphics_pipelines: &[Arc<GraphicsPipeline>],

    color_buffer: &[Arc<ImageView>],
    normal_buffer: &[Arc<ImageView>],
    position_buffer: &[Arc<ImageView>],

    swapchain_image_count: usize,
) {
    let camera_layout = graphics_pipelines[0]
        .layout()
        .set_layouts()
        .first()
        .unwrap();
    let model_layout = graphics_pipelines[0].layout().set_layouts().get(1).unwrap();
    let directional_layout = graphics_pipelines[1]
        .layout()
        .set_layouts()
        .first()
        .unwrap();
    let ambient_layout = graphics_pipelines[2]
        .layout()
        .set_layouts()
        .first()
        .unwrap();
    let point_layout = graphics_pipelines[3]
        .layout()
        .set_layouts()
        .first()
        .unwrap();

    let models = world.query_entities::<Model>();
    for entity in models {
        let mut model_sets = vec![];
        let model = world.component_get_mut::<Model>(entity).unwrap();
        for i in 0..FRAMES_IN_FLIGHT {
            let model_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                model_layout.clone(),
                [WriteDescriptorSet::buffer(0, model.u_buffer[i].clone())],
                [],
            )
            .unwrap();
            model_sets.push(model_set);
        }
        model.descriptor_set = model_sets;
    }

    let point_lights = world.query_mut::<PointLight>();
    for (_entity, point) in point_lights {
        let mut pt_sets = vec![];
        for i in 0..swapchain_image_count {
            let dir_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                point_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer[i].clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer[i].clone()),
                    WriteDescriptorSet::image_view(2, position_buffer[i].clone()),
                    WriteDescriptorSet::buffer(3, point.u_buffers[i].clone()),
                ],
                [],
            )
            .unwrap();
            pt_sets.push(dir_set);
        }
        point.descriptor_set = pt_sets;
    }
    if let Some(ambient) = world.resource_get_mut::<AmbientLight>() {
        let mut ambient_sets = Vec::new();
        for i in 0..swapchain_image_count {
            let ambient_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                ambient_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer[i].clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer[i].clone()),
                    WriteDescriptorSet::buffer(2, ambient.u_buffer.as_ref().unwrap()[i].clone()),
                ],
                [],
            )
            .unwrap();
            ambient_sets.push(ambient_set);
        }
        ambient.descriptor_set = Some(ambient_sets);
    }

    let directional_light = world.resource_get_mut::<DirectionalLight>();
    if let Some(dir_light) = directional_light {
        let mut dir_sets = Vec::new();
        for i in 0..swapchain_image_count {
            let directional_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                directional_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer[i].clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer[i].clone()),
                    WriteDescriptorSet::buffer(2, dir_light.u_buffer.as_ref().unwrap()[i].clone()),
                ],
                [],
            )
            .unwrap();
            dir_sets.push(directional_set);
        }
        dir_light.descriptor_set = Some(dir_sets);
    }

    if let Some(camera) = world.resource_get_mut::<Camera>() {
        let mut camera_sets = vec![];
        for i in 0..swapchain_image_count {
            let camera_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                camera_layout.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    camera.u_buffer.as_ref().unwrap()[i].clone(),
                )],
                [],
            )
            .unwrap();
            camera_sets.push(camera_set);
        }
        camera.descriptor_set = Some(camera_sets);
    }
}
type Color = Vec<Arc<ImageView>>;
type Normal = Vec<Arc<ImageView>>;
type Position = Vec<Arc<ImageView>>;
pub fn create_framebuffers(
    swapchain_images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    allocator: Arc<dyn MemoryAllocator>,
) -> (Vec<Arc<Framebuffer>>, Color, Normal, Position) {
    let mut framebuffers = vec![];
    let mut color_buffers = vec![];
    let mut normal_buffers = vec![];
    let mut position_buffers = vec![];
    let extent = swapchain_images[0].extent();
    dbg!(swapchain_images);
    for i in 0..swapchain_images.len() {
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
        let _ = depth_image.set_debug_utils_object_name(Some(&format!("depth image{}", i)));
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
        let _ = color_image.set_debug_utils_object_name(Some(&format!("color image{}", i)));

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
        let _ = normal_image.set_debug_utils_object_name(Some(&format!("normal image{}", i)));

        let position_image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                extent,
                format: Format::R32G32B32A32_SFLOAT,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let _ = position_image.set_debug_utils_object_name(Some(&format!("position image{}", i)));

        let final_color_view = ImageView::new_default(swapchain_images[i].clone()).unwrap(); // 0: Final color
        let color_view = ImageView::new_default(color_image.clone()).unwrap(); // 1: G-Buffer color
        let normal_view = ImageView::new_default(normal_image.clone()).unwrap(); // 2: G-Buffer normal
        let position_view = ImageView::new_default(position_image.clone()).unwrap(); // 2: G-Buffer normal
        let depth_view = ImageView::new_default(depth_image.clone()).unwrap(); // 3: Depth

        color_buffers.push(color_view.clone());
        normal_buffers.push(normal_view.clone());
        position_buffers.push(position_view.clone());

        framebuffers.push(
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        final_color_view,
                        color_view,
                        normal_view,
                        depth_view,
                        position_view,
                    ],
                    ..Default::default()
                },
            )
            .unwrap(),
        );
    }

    (
        framebuffers,
        color_buffers,
        normal_buffers,
        position_buffers,
    )
}

#[allow(clippy::too_many_lines)]
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
                    store_op: AttachmentStoreOp::DontCare,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // normal attachment (gbuffer)
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // depth attachment(gbuffer)
                AttachmentDescription {
                    format: Format::D32_SFLOAT,
                    samples: SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthAttachmentOptimal,
                    ..Default::default()
                },
                // position attachment (gbuffer)
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    format: Format::R32G32B32A32_SFLOAT,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
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
                        // gposition
                        Some(AttachmentReference {
                            attachment: 4,
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
                        // gposition
                        Some(AttachmentReference {
                            attachment: 4,
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
            dependencies: vec![SubpassDependency {
                src_subpass: Some(0),
                dst_subpass: Some(1),

                // Wait for geometry/base color/normal&posotion writes to finish
                src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                // before running the fragment shader &
                // to finish writing
                dst_stages: PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT,

                // wait for color write data
                src_access: AccessFlags::COLOR_ATTACHMENT_WRITE, // Geometry pass writes
                // before reading input attachemnts and color attachments
                dst_access: AccessFlags::INPUT_ATTACHMENT_READ
                    | AccessFlags::COLOR_ATTACHMENT_WRITE,

                dependency_flags: DependencyFlags::BY_REGION,
                ..Default::default()
            }],

            ..Default::default()
        },
    )
    .unwrap()
}
use crate::{
    components::{DirectionalLight, DirectionalLightUBO},
    shaders as s,
};

use crate::{
    components::{AmbientLight, AmbientLightUBO, Transform},
    components::{Model, ModelUBO, PointLight},
    ecs::World,
};
#[allow(clippy::too_many_lines)]
fn create_graphics_pipelines(
    device: &Arc<Device>,
    deferred_subpass: &Subpass,
    lighting_subpass: &Subpass,
) -> Vec<Arc<GraphicsPipeline>> {
    let deferred_vert = s::defered_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let deferred_frag = s::defered_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let directional_vert = s::lighting_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let directional_frag = s::lighting_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let ambient_vert = s::ambient_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let ambient_frag = s::ambient_frag::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let point_vert = s::point_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let point_frag = s::point_frag::load(device.clone())
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
    let point_stages = [
        PipelineShaderStageCreateInfo::new(point_vert),
        PipelineShaderStageCreateInfo::new(point_frag),
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

    let point_layout = PipelineLayout::new(
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
                        2, // frag position data
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
                        3, // point light uniform
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

    let cull_mode = CullMode::Back;
    let front_face = FrontFace::CounterClockwise;
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

    let point_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: point_stages.into_iter().collect(),
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
            ..GraphicsPipelineCreateInfo::layout(point_layout)
        },
    )
    .expect("failed to create point graphics pipeline");
    vec![
        deferred_pipeline,
        directional_pipeline,
        ambient_pipeline,
        point_pipeline,
    ]
}
pub fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
}

fn create_instance(event_loop: &ActiveEventLoop) -> Arc<Instance> {
    let library =
        VulkanLibrary::new().expect("failed to load library, please install vulkan drivers");
    let mut required_extensions = Surface::required_extensions(event_loop);
    required_extensions.ext_validation_features = true;
    required_extensions.ext_debug_utils = true;

    let validation_layer = "VK_LAYER_KHRONOS_validation";

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enabled_layers: vec![validation_layer.to_string()],
            enabled_validation_features: vec![
                ValidationFeatureEnable::BestPractices,
                ValidationFeatureEnable::SynchronizationValidation,
            ],
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
