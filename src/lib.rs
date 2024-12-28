use bytemuck::{Pod, Zeroable};
use std::{f32::consts::FRAC_PI_4, sync::Arc, time::Instant};
use ultraviolet::{projection, Mat4, Vec3};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType::{self, UniformBuffer},
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::{self, Format},
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
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
        SubpassDescription,
    },
    shader::ShaderStages,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};
const VERTICES: [Vert; 36] = [
    // front face
    Vert {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.75, 0.837],
    },
    Vert {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    // back face
    Vert {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    // top face
    Vert {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // bottom face
    Vert {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // left face
    Vert {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // right face
    Vert {
        position: [1.000000, -1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, 1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    Vert {
        position: [1.000000, -1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
];

const FRAMES_IN_FLIGHT: usize = 3;
#[derive(Default)]
pub struct App {
    render_context: Option<RenderContext>,
    app_context: Option<AppContext>,
}
struct AppContext {
    window: Arc<Window>,
    aspect_ratio: f32,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[Vert]>,
    start_time: Instant,
    memory_allocator: Arc<dyn MemoryAllocator>,
}
struct RenderContext {
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,

    recreate_swapchain: bool,
    current_frame: usize,
    frame_counter: u128,

    framebuffers: Vec<Arc<Framebuffer>>,
    uniform_buffers: Vec<Subbuffer<TransformationUBO>>,
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
    frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vert {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
struct TransformationUBO {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_context.is_some() {
            eprintln!("resumed called while app is already some");
            return;
        }
        let start_time = Instant::now();
        let instance = App::create_instance(event_loop);
        let window = App::create_window(event_loop);
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            App::create_physical_device(&instance, &surface, &device_extensions);
        let (device, queue) =
            App::create_device(physical_device, queue_family_index, &device_extensions);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: FRAMES_IN_FLIGHT,
                ..Default::default()
            },
        ));
        let vertex_buffer = Buffer::from_iter(
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
            VERTICES,
        )
        .unwrap();

        let window_size = window.inner_size();
        let (swapchain, images) = App::create_swapchain(&window, &device, &surface);

        let render_pass = App::create_renderpass(&device, swapchain.image_format());

        let framebuffers =
            App::recreate_framebuffers(&images, &render_pass, memory_allocator.clone());
        let pipeline = App::create_graphics_pipeline(&device, &render_pass);
        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };
        let recreate_swapchain = false;
        let frames_resources_free: Vec<Option<Box<dyn GpuFuture>>> = {
            let mut vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let previous_frame_end = Some(sync::now(device.clone()).boxed());
                vec.push(previous_frame_end);
            }
            vec
        };
        let aspect_ratio = {
            let windowsize: [f32; 2] = window_size.into();
            windowsize[0] / windowsize[1]
        };
        let (uniform_buffers, descriptor_sets) =
            App::create_descriptor_sets(device.clone(), memory_allocator.clone());
        self.render_context = Some(RenderContext {
            framebuffers,
            swapchain,
            render_pass,
            recreate_swapchain,
            pipeline,
            viewport,
            current_frame: 0,
            frame_counter: 0,
            descriptor_sets,
            frames_resources_free,
            uniform_buffers,
        });
        self.app_context = Some(AppContext {
            command_buffer_allocator,
            device,
            vertex_buffer,
            queue,
            start_time,
            window: window.clone(),
            aspect_ratio,
            memory_allocator: memory_allocator.clone(),
        });
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // INFO: RCX = render_context, its just a pain to have so many really long lines, APX is the app context
        let rcx = self.render_context.as_mut().unwrap();
        let acx = self.app_context.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close ~uhhh~ button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => rcx.recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                println!("frame start UwU");
                acx.window.request_redraw();
                let window_size = acx.window.inner_size();
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }
                rcx.frame_counter += 1;
                rcx.frames_resources_free[rcx.current_frame]
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                {
                    let mut write = rcx.uniform_buffers[rcx.current_frame].write().unwrap();
                    *write = App::calculate_current_transform(acx.start_time, acx.aspect_ratio);
                    // write needs to be dropped to free the lock on the uniform buffer
                }

                if rcx.recreate_swapchain {
                    // Use the new dimensions of the window.
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect(
                            "failed to create the stinkin swapchain wooo baby that was difficult",
                        );
                    rcx.swapchain = new_swapchain;
                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    rcx.framebuffers = App::recreate_framebuffers(
                        &new_images,
                        &rcx.render_pass,
                        acx.memory_allocator.clone(),
                    );
                    rcx.viewport.extent = window_size.into();
                    acx.aspect_ratio = window_size.width as f32 / window_size.height as f32;

                    rcx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };
                if suboptimal {
                    rcx.recreate_swapchain = true;
                }
                let mut builder = AutoCommandBufferBuilder::primary(
                    &acx.command_buffer_allocator,
                    acx.queue.queue_family_index(),
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
                                Some([0.0, 0.68, 1.0, 1.0].into()),
                                Some(1.0.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass. This can be either
                            // `Inline` or `SecondaryCommandBuffers`
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.pipeline.layout().clone(),
                        0,
                        rcx.descriptor_sets[rcx.current_frame].clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, acx.vertex_buffer.clone())
                    .unwrap();
                builder // not necessary to seperate but good for clear thinking about it
                    .draw(acx.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap();
                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();
                let future = rcx.frames_resources_free[rcx.current_frame]
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(acx.queue.clone(), command_buffer)
                    .unwrap()
                    // dosent present imediately but submits a present command to
                    // the queue, so the triangle will finish rendering
                    .then_swapchain_present(
                        acx.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush(); // signal will tell gpu to finish and use fence

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.frames_resources_free[rcx.current_frame] = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.frames_resources_free[rcx.current_frame] =
                            Some(sync::now(acx.device.clone()).boxed());
                    }
                    Err(e) => {
                        rcx.frames_resources_free[rcx.current_frame] =
                            Some(Box::new(sync::now(acx.device.clone())).boxed());
                        println!("Failed to flush future: {:?}", e);
                    }
                }
                rcx.current_frame = (rcx.current_frame + 1) % FRAMES_IN_FLIGHT;
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let app_context = self.app_context.as_mut().unwrap();
        app_context.window.request_redraw();
    }
}
impl App {
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

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
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
            .expect("poor ass, u dont have a good gpu")
    }
    fn create_device(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
        required_extensions: &DeviceExtensions,
    ) -> (Arc<Device>, Arc<Queue>) {
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
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
    fn create_depth_buffer(
        memory_allocator: Arc<dyn MemoryAllocator>,
        dimensions: [u32; 3],
    ) -> Arc<Image> {
        // First, create the image for the depth buffer
        let depth_image = Image::new(
            memory_allocator,
            ImageCreateInfo {
                // Makes the image the same size as our window
                extent: dimensions,
                // Tell Vulkan this is for depth information
                format: Format::D16_UNORM,
                // We want to use this as a depth attachment
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        depth_image
    }
    fn create_swapchain(
        window: &Arc<Window>,
        device: &Arc<Device>,
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
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn create_renderpass(device: &Arc<Device>, image_format: Format) -> Arc<RenderPass> {
        RenderPass::new(
            device.clone(),
            RenderPassCreateInfo {
                attachments: vec![
                    // Your existing color attachment
                    AttachmentDescription {
                        format: image_format,
                        samples: SampleCount::Sample1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::PresentSrc,
                        ..Default::default()
                    },
                    // New depth attachment
                    AttachmentDescription {
                        format: Format::D16_UNORM,
                        samples: SampleCount::Sample1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::DontCare, // We don't need to keep depth data
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                        ..Default::default()
                    },
                ],
                subpasses: vec![SubpassDescription {
                    color_attachments: vec![Some(AttachmentReference {
                        attachment: 0, // Color attachment
                        layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    })],
                    depth_stencil_attachment: Some(AttachmentReference {
                        attachment: 1, // Depth attachment
                        layout: ImageLayout::DepthStencilAttachmentOptimal,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn create_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "shaders/vertex.glsl",
            }
        }
        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "shaders/fragment.glsl",
            }
        }
        let vertex_shader = vertex_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fragment_shader = fragment_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vert::per_vertex()
            .definition(&vertex_shader.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vertex_shader),
            PipelineShaderStageCreateInfo::new(fragment_shader),
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
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo {
                set_layouts: vec![DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::VERTEX,
                            descriptor_type: UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(UniformBuffer)
                        },
                    )]
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
        // We have to indicate which subpass of which render pass this pipeline is going to be
        // used in. The pipeline will only be usable from this particular subpass.
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                // How vertex data is read from the vertex buffers into the vertex shader.
                vertex_input_state: Some(vertex_input_state),
                // How vertices are arranged into primitive shapes. The default primitive shape
                // is a triangle.
                input_assembly_state: Some(InputAssemblyState::default()),
                // How primitives are transformed and clipped to fit the framebuffer. We use a
                // resizable viewport, set to draw over the entire window.
                viewport_state: Some(ViewportState::default()),
                // How polygons are culled and converted into a raster of pixels. The default
                // value does not perform any culling.
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face: FrontFace::Clockwise,
                    ..Default::default()
                }),
                // How multiple fragment shader samples are converted to a single pixel value.
                // The default value does not perform any multisampling.
                multisample_state: Some(MultisampleState::default()),
                // How pixel values are combined with the values already present in the
                // framebuffer. The default value overwrites the old value with the new one,
                // without any blending.
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                // Dynamic states allows us to specify parts of the pipeline settings when
                // recording the command buffer, before we perform drawing. Here, we specify
                // that the viewport should be dynamic.
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),

                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .expect("failed to create graphics pipeline")
    }
    fn recreate_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
        allocator: Arc<dyn MemoryAllocator>,
    ) -> Vec<Arc<Framebuffer>> {
        let mut framebuffers = vec![];
        for i in 0..images.len() {
            let depth_buffer = App::create_depth_buffer(allocator.clone(), images[0].extent());
            let image_view = ImageView::new_default(images[i].clone()).unwrap();
            let depth_view = ImageView::new_default(depth_buffer).unwrap();
            framebuffers.push(
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view, depth_view],
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
        framebuffers
    }
    fn calculate_current_transform(start_time: Instant, aspect_ratio: f32) -> TransformationUBO {
        let move_back = Mat4::from_translation(Vec3 {
            x: 0.0,
            y: 0.0,
            z: -5.0,
        });
        let rotation = Mat4::from_euler_angles(
            0.0,
            (start_time.elapsed().as_secs_f32() * 0.5) % 360.0,
            (start_time.elapsed().as_secs_f32() * 0.3) % 360.0,
        );

        println!("elapsed time{}", start_time.elapsed().as_millis());

        TransformationUBO {
            model: move_back * rotation,
            view: ultraviolet::Mat4::look_at(
                Vec3::new(0.0, 0.0, 3.0), // Move camera back slightly
                Vec3::new(0.0, 0.0, 0.0), // Look at origin
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            proj: projection::perspective_vk(FRAC_PI_4, aspect_ratio, 0.001, 100.0),
        }
    }
    fn create_descriptor_sets(
        device: Arc<Device>,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> (
        Vec<Subbuffer<TransformationUBO>>,
        Vec<Arc<PersistentDescriptorSet>>,
    ) {
        let layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0, // binding
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::VERTEX,
                        descriptor_count: 1,
                        descriptor_type: DescriptorType::UniformBuffer,

                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                set_count: FRAMES_IN_FLIGHT as usize,
                ..Default::default()
            },
        );
        let mut uniform_buffers = vec![];
        let mut descriptor_sets = vec![];
        for i in 0..FRAMES_IN_FLIGHT {
            uniform_buffers.push(
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
            let descriptor_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffers[i].clone())],
                [],
            )
            .unwrap();
            descriptor_sets.push(descriptor_set);
        }
        (uniform_buffers, descriptor_sets)
    }
}
