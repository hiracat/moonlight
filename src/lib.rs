use bytemuck::{Pod, Zeroable};
use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Instant;
use ultraviolet::{projection, Mat4, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::DescriptorType::UniformBuffer;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateFlags},
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
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
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,

    uniform_buffer: Arc<Subbuffer<TransformationUBO>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
}

impl App {
    // TODO: all init is done in the resumed function because window needs an
    // activeeventloop to be created, there is a way around this, but idk what it is

    // INFO: if you still want an empty app struct it derives default
    pub fn new() -> Self {
        App {
            app_context: None,
            render_context: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_context.is_some() {
            eprintln!("resumed called while app is already some, why?");
            return;
        }

        let start_time = Instant::now();

        // create temorary variables, then assign them all at the end to avoid forgetting things

        let instance = App::create_instance(event_loop);
        let window = App::create_window(event_loop);

        let surface = App::create_surface(&instance, &window);
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            App::create_physical_device(&instance, &surface, &device_extensions);

        let (device, queue) =
            App::create_device(physical_device, queue_family_index, &device_extensions);

        let vertices = [
            Vert {
                in_position: [-0.5, -0.25, 0.0],
                in_color: [1.0, 0.0, 0.0, 1.0],
            },
            Vert {
                in_position: [0.0, 0.5, 0.0],
                in_color: [0.0, 1.0, 0.0, 1.0],
            },
            Vert {
                in_position: [0.25, -0.1, 0.0],
                in_color: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        //TODO: change this to be more generic and not rely on whatever this bs is
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
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
            vertices,
        )
        .unwrap();

        let window_size = window.inner_size();
        let (swapchain, images) = App::create_swapchain(&window, &device, &surface);

        let render_pass = App::create_renderpass(&device, swapchain.image_format());

        let framebuffers = App::recreate_framebuffers(&images, &render_pass);
        let pipeline = App::create_graphics_pipeline(&device, &render_pass);

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };
        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        let window_ratio = {
            let windowsize: [f32; 2] = window_size.into();
            windowsize[0] / windowsize[1]
        };

        let uniform_buffer: Arc<Subbuffer<TransformationUBO>> = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap()
        .into();
        let initial_transform = App::calculate_current_transform(start_time, window_ratio);
        let mut buffer_write = uniform_buffer.write().unwrap();
        *buffer_write = initial_transform;
        drop(buffer_write);

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                set_count: 1,
                ..Default::default()
            },
        );

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(
                0,
                uniform_buffer.as_ref().clone(),
            )],
            [],
        )
        .unwrap();

        self.render_context = Some(RenderContext {
            descriptor_set,
            window: window.clone(),
            swapchain,
            render_pass,
            recreate_swapchain,
            framebuffers,
            pipeline,
            previous_frame_end,
            viewport,
            uniform_buffer,
        });
        self.app_context = Some(AppContext {
            command_buffer_allocator,
            device,
            vertex_buffer,
            queue,
            start_time,
            window: window.clone(),
            aspect_ratio: window_ratio,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let render_context = self.render_context.as_mut().unwrap();
        let app_context = self.app_context.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => render_context.recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                let window_size = render_context.window.inner_size();
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }
                render_context
                    .previous_frame_end
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                app_context.window.request_redraw();

                if render_context.recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) = render_context
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..render_context.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    render_context.swapchain = new_swapchain;

                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    render_context.framebuffers =
                        App::recreate_framebuffers(&new_images, &render_context.render_pass);

                    render_context.viewport.extent = window_size.into();

                    render_context.recreate_swapchain = false;
                }
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(render_context.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            render_context.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if suboptimal {
                    render_context.recreate_swapchain = true;
                }
                let mut builder = AutoCommandBufferBuilder::primary(
                    &app_context.command_buffer_allocator,
                    app_context.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // Before we can draw, we have to *enter a render pass*.
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // with clear values, any others should use `None` as the clear value.
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                render_context.framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass. This can be either
                            // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                            // advanced and is not covered here.
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [render_context.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(render_context.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        render_context.pipeline.layout().clone(),
                        0,
                        render_context.descriptor_set.clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, app_context.vertex_buffer.clone())
                    .unwrap();

                builder
                    .draw(app_context.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap();

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();

                let future = render_context
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(app_context.queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        app_context.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            render_context.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();
                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        render_context.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        render_context.recreate_swapchain = true;
                        render_context.previous_frame_end =
                            Some(sync::now(app_context.device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let render_context = self.render_context.as_mut().unwrap();
        render_context.window.request_redraw();
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
        //DEBUG: println!("{:?}", library.supported_extensions());

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance")
    }
    fn create_surface(instance: &Arc<Instance>, window: &Arc<Window>) -> Arc<Surface> {
        Surface::from_window(instance.clone(), window.clone()).unwrap()
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
            // takes an iterator of physical device, and pairs it
            // to a u32
            .filter_map(|physical_device| {
                physical_device
                    .queue_family_properties()
                    .iter() // iterate over all available queues for each device
                    .enumerate() // returns an iterator of type (physicaldevice, u32)
                    // returns first elemnt to return true, or none
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
            .expect("no suitable physical device found")
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
    fn create_swapchain(
        window: &Arc<Window>,
        device: &Arc<Device>,
        surface: &Arc<Surface>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let window_size = window.inner_size();
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let (image_format, _) = device
            .physical_device()
            .surface_formats(surface, Default::default())
            .unwrap()[0]; // take the first available format

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                // Some drivers report an `min_image_count` of 1, but fullscreen mode requires
                // at least 2. Therefore we must ensure the count is at least 2, otherwise the
                // program would crash when entering fullscreen mode on those drivers.
                min_image_count: surface_capabilities.min_image_count.max(2),

                image_format,

                // The size of the window, only used to initially setup the swapchain.
                //
                // NOTE:
                // On some drivers the swapchain extent is specified by
                // `surface_capabilities.current_extent` and the swapchain size must use this
                // extent. This extent is always the same as the window size.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window size.
                //
                // Both of these cases need the swapchain to use the window size, so we just
                // use that.
                image_extent: window_size.into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,

                // The alpha mode indicates how the alpha value of the final image will behave.
                // For example, you can choose whether the window will be
                // opaque or transparent.
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
        // this is a macro, the proper way to do this is probably very different but we dont talk
        // about that
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `format: <ty>` indicates the type of the format of the image. This has to be
                    // one of the types of the `vulkano::format` module (or alternatively one of
                    // your structs that implements the `FormatDesc` trait). Here we use the same
                    // format as the swapchain.
                    format: image_format,
                    // `samples: 1` means that we ask the GPU to use one sample to determine the
                    // value of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    // msaa-renderpass.rs.
                    samples: 1,
                    // `load_op: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load_op: Clear,
                    // `store_op: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store_op: Store,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
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
            // here we create a uniform
            // buffer descriptor set
            // layout
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

        // Finally, create the pipeline.
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
                rasterization_state: Some(RasterizationState::default()),
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
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn calculate_current_transform(start_time: Instant, aspect_ratio: f32) -> TransformationUBO {
        let elapsed_time = start_time - Instant::now();
        let rotation = Mat4::from_euler_angles(0.0, 0.0, elapsed_time.as_secs() as f32 * 100.0);
        TransformationUBO {
            model: rotation,
            view: ultraviolet::Mat4::look_at(
                Vec3::new(0.0, 0.0, 2.0), // Move camera back slightly
                Vec3::new(0.0, 0.0, 0.0), // Look at origin
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            proj: projection::perspective_vk(FRAC_PI_4, aspect_ratio, 0.001, 100.0),
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vert {
    #[format(R32G32B32_SFLOAT)]
    in_position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    in_color: [f32; 4],
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
struct TransformationUBO {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}
