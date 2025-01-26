use bytemuck::{Pod, Zeroable};
use models::VERTICES;
use std::{f32::consts::FRAC_PI_4, sync::Arc, time::Instant};
use ultraviolet::{projection, Mat4, Vec3};
use vulkan::{create_device, create_physical_device};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{
            DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType::{self, UniformBuffer},
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceExtensions, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount},
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
            subpass::PipelineSubpassType,
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
        SubpassDependency, SubpassDescription,
    },
    shader::ShaderStages,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, AccessFlags, DependencyFlags, GpuFuture, PipelineStages},
    Validated, VulkanError,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};
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
    defered_pipeline: Arc<GraphicsPipeline>,
    lighting_pipeline: Arc<GraphicsPipeline>,

    viewport: Viewport,

    recreate_swapchain: bool,
    current_frame: usize,
    frame_counter: u128,

    framebuffers: Vec<Arc<Framebuffer>>,
    uniform_buffers: Vec<Subbuffer<TransformationUBO>>,
    defered_sets: Vec<Arc<PersistentDescriptorSet>>,
    lighting_sets: Vec<Arc<PersistentDescriptorSet>>,
    frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}
#[derive(vulkano::buffer::BufferContents, Vertex)]
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
//#[derive(Pod, Zeroable, Copy, Debug, Clone)]
//#[repr(C)]
//struct AmbientLightUBO {
//    color: [f32; 3],
//    intensity: f32,
//}
//#[derive(Pod, Zeroable, Copy, Debug, Clone)]
//#[repr(C)]
//struct DirectionalLightUBO {
//    position: [f32; 3],
//    color: [f32; 3],
//}
mod models;
mod vulkan;
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_context.is_some() {
            eprintln!("resumed called while app is already some");
            return;
        }
        let start_time = Instant::now();
        let instance = vulkan::create_instance(event_loop);
        let window = App::create_window(event_loop);
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
            crate::models::VERTICES,
        )
        .unwrap();

        let window_size = window.inner_size();
        let (swapchain, images) = App::create_swapchain(&window, &device, &surface);

        let render_pass = App::create_renderpass(&device, swapchain.image_format());
        let deferred_pass = Arc::new(Subpass::from(render_pass.clone(), 0).unwrap());
        let lighting_pass = Arc::new(Subpass::from(render_pass.clone(), 1).unwrap());

        let (framebuffers, color_buffers, normal_buffers) =
            App::recreate_framebuffers(&images, &render_pass, memory_allocator.clone());

        let (deffered_pipeline, lighting_pipeline) =
            App::create_graphics_pipeline(&device, &deferred_pass, &lighting_pass);
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
        let (uniform_buffers, defered_sets, lighting_sets) = App::create_descriptor_sets(
            device.clone(),
            deffered_pipeline.clone(),
            lighting_pipeline.clone(),
            memory_allocator.clone(),
            color_buffers[0].clone(),
            normal_buffers[0].clone(),
        );

        self.render_context = Some(RenderContext {
            framebuffers,
            swapchain,
            render_pass,
            recreate_swapchain,
            defered_pipeline: deffered_pipeline,
            lighting_pipeline,
            viewport,
            current_frame: 0,
            frame_counter: 0,
            defered_sets,
            lighting_sets,
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

                let color_buffers: Vec<Arc<ImageView>>;
                let normal_buffers: Vec<Arc<ImageView>>;
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
                    (rcx.framebuffers, normal_buffers, color_buffers) = App::recreate_framebuffers(
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
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
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
                    .bind_pipeline_graphics(rcx.defered_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.defered_pipeline.layout().clone(),
                        0,
                        rcx.defered_sets[rcx.current_frame].clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, acx.vertex_buffer.clone())
                    .unwrap()
                    .draw(acx.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .next_subpass(
                        SubpassEndInfo::default(),
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(rcx.lighting_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.lighting_pipeline.layout().clone(),
                        0,
                        rcx.lighting_sets[rcx.current_frame].clone(),
                    )
                    .unwrap()
                    .draw(VERTICES.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();

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
                    // final color attachment
                    AttachmentDescription {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        format: image_format,
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
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    },
                    // normal attachment (gbuffer)
                    AttachmentDescription {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        format: Format::R16G16B16A16_SFLOAT,
                        samples: SampleCount::Sample1,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    },
                    // depth attachment(gbuffer)
                    AttachmentDescription {
                        format: Format::D16_UNORM,
                        samples: SampleCount::Sample1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::DontCare, // We don't need to keep depth data
                        initial_layout: ImageLayout::Undefined,
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
                dependencies: vec![SubpassDependency {
                    src_subpass: Some(0),
                    dst_subpass: Some(1),
                    src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT, // Geometry pass writes
                    dst_stages: PipelineStages::FRAGMENT_SHADER,         // Lighting pass reads
                    src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,     // Geometry writes to color
                    dst_access: AccessFlags::INPUT_ATTACHMENT_READ,      // Lighting reads as input
                    dependency_flags: DependencyFlags::BY_REGION,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn create_graphics_pipeline(
        device: &Arc<Device>,
        deffered_subpass: &Subpass,
        lighting_subpass: &Subpass,
    ) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>) {
        mod deferred_vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/shaders/defered_vert.glsl",
            }
        }
        mod deferred_frag {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/shaders/defered_frag.glsl"
            }
        }
        mod lighting_vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/shaders/lighting_vert.glsl"
            }
        }
        mod lighting_frag {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/shaders/lighting_frag.glsl"
            }
        }

        let deferred_vert = deferred_vert::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let deferred_frag = deferred_frag::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let lighting_vert = lighting_vert::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let lighting_frag = lighting_frag::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = Vert::per_vertex()
            .definition(&deferred_vert.info().input_interface)
            .unwrap();

        let defered_stagse = [
            PipelineShaderStageCreateInfo::new(deferred_vert),
            PipelineShaderStageCreateInfo::new(deferred_frag),
        ];
        let lighting_stages = [
            PipelineShaderStageCreateInfo::new(lighting_vert),
            PipelineShaderStageCreateInfo::new(lighting_frag),
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

        let defered_layout = PipelineLayout::new(
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

        let lighting_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo {
                set_layouts: vec![DescriptorSetLayoutCreateInfo {
                    bindings: [
                        (
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::VERTEX,
                                descriptor_type: UniformBuffer,
                                descriptor_count: 1,
                                ..DescriptorSetLayoutBinding::descriptor_type(UniformBuffer)
                            },
                        ),
                        (
                            1, // Binding 1 for color input attachment
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
                            2, // Binding 2 for normals input attachment
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::FRAGMENT,
                                descriptor_type: DescriptorType::InputAttachment,
                                descriptor_count: 1,
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::InputAttachment,
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
        // We have to indicate which subpass of which render pass this pipeline is going to be
        // used in. The pipeline will only be usable from this particular subpass.
        let defered_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: defered_stagse.into_iter().collect(),
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
                    deffered_subpass.num_color_attachments(),
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

                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    deffered_subpass.clone(),
                )),
                ..GraphicsPipelineCreateInfo::layout(defered_layout.clone())
            },
        )
        .expect("failed to create defered graphics pipeline");

        let lighting_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: lighting_stages.into_iter().collect(),
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
                    lighting_subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                // Dynamic states allows us to specify parts of the pipeline settings when
                // recording the command buffer, before we perform drawing. Here, we specify
                // that the viewport should be dynamic.
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),

                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    lighting_subpass.clone(),
                )),
                ..GraphicsPipelineCreateInfo::layout(lighting_layout)
            },
        )
        .expect("failed to create lighting graphics pipeline");

        (defered_pipeline, lighting_pipeline)
    }
    fn recreate_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
        allocator: Arc<dyn MemoryAllocator>,
    ) -> (
        Vec<Arc<Framebuffer>>,
        Vec<Arc<ImageView>>,
        Vec<Arc<ImageView>>,
    ) {
        let mut framebuffers = vec![];
        let mut color_buffers = vec![];
        let mut normal_buffers = vec![];
        for image in images {
            let depth_buffer = Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    // Makes the image the same size as our window
                    extent: image.extent(),
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
            let gbuffer_color = Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    extent: image.extent(),
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
            let gbuffer_normal = Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: image.extent(),
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            let final_color_view = ImageView::new_default(image.clone()).unwrap(); // 0: Final color
            let deffered_color_view = ImageView::new_default(gbuffer_color.clone()).unwrap(); // 1: G-Buffer color
            let defered_normal_view = ImageView::new_default(gbuffer_normal.clone()).unwrap(); // 2: G-Buffer normal
            let depth_view = ImageView::new_default(depth_buffer.clone()).unwrap(); // 3: Depth

            color_buffers.push(deffered_color_view.clone());
            normal_buffers.push(defered_normal_view.clone());

            let attachments = vec![
                // Must match render pass attachment order:
                final_color_view,
                deffered_color_view,
                defered_normal_view,
                depth_view,
            ];
            framebuffers.push(
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
        (framebuffers, color_buffers, normal_buffers)
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
        deffered_pipeline: Arc<GraphicsPipeline>,
        lighting_pipeline: Arc<GraphicsPipeline>,
        memory_allocator: Arc<dyn MemoryAllocator>,

        color_buffer: Arc<ImageView>,
        normal_buffer: Arc<ImageView>,
    ) -> (
        Vec<Subbuffer<TransformationUBO>>,
        Vec<Arc<PersistentDescriptorSet>>,
        Vec<Arc<PersistentDescriptorSet>>,
    ) {
        let defered_layout = deffered_pipeline.layout().set_layouts().get(0).unwrap();
        let lighting_layout = lighting_pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                set_count: FRAMES_IN_FLIGHT as usize,
                ..Default::default()
            },
        );
        let mut uniform_buffers = vec![];
        let mut deffered_sets = vec![];
        let mut lighting_sets = vec![];

        for i in 0..FRAMES_IN_FLIGHT {
            let uniform_buffer = Buffer::new_sized(
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
            .unwrap();
            uniform_buffers.push(uniform_buffer.clone());

            let deffered_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                defered_layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffers[i].clone())],
                [],
            )
            .unwrap();
            deffered_sets.push(deffered_set);

            let lighting_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                lighting_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_buffer),
                    WriteDescriptorSet::image_view(1, color_buffer.clone()),
                    WriteDescriptorSet::image_view(2, normal_buffer.clone()),
                ],
                [],
            )
            .unwrap();
            lighting_sets.push(lighting_set);
        }
        (uniform_buffers, deffered_sets, lighting_sets)
    }
}
