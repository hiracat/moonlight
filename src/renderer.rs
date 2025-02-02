use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{
        view::ImageView, Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageUsage,
        SampleCount,
    },
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
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDependency, SubpassDescription,
    },
    shader::ShaderStages,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{AccessFlags, DependencyFlags, GpuFuture, PipelineStages},
};
use winit::window::Window;

use crate::{
    resources::{self, TransformationUBO},
    FRAMES_IN_FLIGHT,
};

//marker

pub struct Context {
    pub recreate_swapchain: bool,
    pub current_frame: usize,
    pub frame_counter: u128,

    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub deferred_pipeline: Arc<GraphicsPipeline>,
    pub lighting_pipeline: Arc<GraphicsPipeline>,

    pub viewport: Viewport,

    pub framebuffers: Vec<Arc<Framebuffer>>,

    pub color_buffers: Vec<Arc<ImageView>>,
    pub normal_buffers: Vec<Arc<ImageView>>,

    pub uniform_buffers: Vec<Subbuffer<TransformationUBO>>,
    pub deferred_sets: Vec<Arc<PersistentDescriptorSet>>,
    pub lighting_sets: Vec<Arc<PersistentDescriptorSet>>,
    pub memory_allocator: Arc<dyn MemoryAllocator>,

    pub frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}

impl Context {
    pub fn init(device: &Arc<Device>, window: &Arc<Window>, surface: &Arc<Surface>) -> Self {
        let window_size = window.inner_size();
        let (swapchain, images) = create_swapchain(&device, &window, &surface);

        let render_pass = create_renderpass(&device, swapchain.image_format());
        let deferred_pass = Arc::new(Subpass::from(render_pass.clone(), 0).unwrap());
        let lighting_pass = Arc::new(Subpass::from(render_pass.clone(), 1).unwrap());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let (framebuffers, color_buffers, normal_buffers) =
            create_framebuffers(&images, &render_pass, memory_allocator.clone());
        let mut uniform_buffers: Vec<Subbuffer<TransformationUBO>> = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
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
        }

        let (defered_pipeline, lighting_pipeline) =
            create_graphics_pipelines(&device, &deferred_pass, &lighting_pass);
        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };
        let frames_resources_free: Vec<Option<Box<dyn GpuFuture>>> = {
            let mut vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                vec.push(previous_frame_end);
            }
            vec
        };
        let (defered_sets, lighting_sets) = create_descriptor_sets(
            &device,
            &defered_pipeline,
            &lighting_pipeline,
            &uniform_buffers,
            &color_buffers,
            &normal_buffers,
        );
        Context {
            recreate_swapchain: false,
            current_frame: 0,
            frame_counter: 0,
            deferred_pipeline: defered_pipeline,
            deferred_sets: defered_sets,
            swapchain,
            framebuffers,
            frames_resources_free,
            color_buffers,
            normal_buffers,
            lighting_pipeline,
            render_pass,
            memory_allocator,
            viewport,
            uniform_buffers,
            lighting_sets,
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
            ..Default::default()
        },
    )
    .unwrap()
}
pub fn create_descriptor_sets(
    device: &Arc<Device>,
    deferred_pipeline: &Arc<GraphicsPipeline>,
    lighting_pipeline: &Arc<GraphicsPipeline>,

    uniform_buffers: &[Subbuffer<TransformationUBO>],
    color_buffer: &[Arc<ImageView>],
    normal_buffer: &[Arc<ImageView>],
) -> (
    Vec<Arc<PersistentDescriptorSet>>, // deferred
    Vec<Arc<PersistentDescriptorSet>>, // lighting
) {
    let deferred_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
    let lighting_layout = lighting_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo {
            set_count: 2 * FRAMES_IN_FLIGHT as usize,
            ..Default::default()
        },
    );
    let mut deferred_sets = vec![];
    let mut lighting_sets = vec![];

    for i in 0..FRAMES_IN_FLIGHT {
        let deferred_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            deferred_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffers[i].clone())],
            [],
        )
        .unwrap();
        deferred_sets.push(deferred_set);

        let lighting_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            lighting_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffers[i].clone()),
                WriteDescriptorSet::image_view(1, color_buffer[i].clone()),
                WriteDescriptorSet::image_view(2, normal_buffer[i].clone()),
            ],
            [],
        )
        .unwrap();
        lighting_sets.push(lighting_set);
    }
    (deferred_sets, lighting_sets)
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
        let depth_buffer = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                // Makes the image the same size as our window
                extent,
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
        let depth_view = ImageView::new_default(depth_buffer.clone()).unwrap(); // 3: Depth

        color_buffers.push(color_view.clone());
        normal_buffers.push(normal_view.clone());

        let attachments = vec![
            // Must match render pass attachment order:
            final_color_view,
            color_view,
            normal_view,
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
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // normal attachment (gbuffer)
                AttachmentDescription {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: SampleCount::Sample1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ShaderReadOnlyOptimal,
                    ..Default::default()
                },
                // depth attachment(gbuffer)
                AttachmentDescription {
                    format: Format::D16_UNORM,
                    samples: SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store, // We don't need to keep depth data
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilAttachmentOptimal,
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
                        layout: ImageLayout::DepthStencilAttachmentOptimal,
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
                    dst_stages: PipelineStages::FRAGMENT_SHADER, // Transition before fragment shader reads
                    src_access: AccessFlags::COLOR_ATTACHMENT_WRITE, // Geometry pass writes
                    dst_access: AccessFlags::INPUT_ATTACHMENT_READ, // Lighting pass reads

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
) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>) {
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
            path: "src/shaders/lighting_vert.glsl"
        }
    }
    mod lighting_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/lighting_frag.glsl"
        }
    }

    let defered_vert = defered_vert::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let defered_frag = defered_frag::load(device.clone())
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

    let vertex_input_state = resources::Vertex::per_vertex()
        .definition(&defered_vert.info().input_interface)
        .unwrap();

    let deferred_stagse = [
        PipelineShaderStageCreateInfo::new(defered_vert),
        PipelineShaderStageCreateInfo::new(defered_frag),
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
                        descriptor_type: DescriptorType::UniformBuffer,
                        descriptor_count: 1,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
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
                            descriptor_type: DescriptorType::UniformBuffer,
                            descriptor_count: 1,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
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
                deferred_subpass.num_color_attachments(),
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
                deferred_subpass.clone(),
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

    (deferred_pipeline, lighting_pipeline)
}
