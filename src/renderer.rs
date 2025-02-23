use std::{f32::consts::FRAC_PI_4, sync::Arc};

use ultraviolet::{projection, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount},
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
    resources::{self, AmbientLightUBO, DirectionalLightUBO, ModelData, ViewProjUBO},
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
    pub directional_pipeline: Arc<GraphicsPipeline>,
    pub ambient_pipeline: Arc<GraphicsPipeline>,

    pub viewport: Viewport,

    pub framebuffers: Vec<Arc<Framebuffer>>,

    pub view_proj_buffers: Vec<Subbuffer<ViewProjUBO>>,
    pub model_buffers: Vec<Subbuffer<ModelData>>,
    pub ambient_buffers: Vec<Subbuffer<AmbientLightUBO>>,
    pub directional_buffers: Vec<Vec<Subbuffer<DirectionalLightUBO>>>,
    pub color_buffers: Vec<Arc<ImageView>>,
    pub normal_buffers: Vec<Arc<ImageView>>,

    pub view_proj_sets: Vec<Arc<PersistentDescriptorSet>>,
    pub model_sets: Vec<Arc<PersistentDescriptorSet>>,
    pub directional_sets: Vec<Vec<Arc<PersistentDescriptorSet>>>,
    pub ambient_sets: Vec<Arc<PersistentDescriptorSet>>,

    pub memory_allocator: Arc<dyn MemoryAllocator>,

    pub frames_resources_free: Vec<Option<Box<dyn GpuFuture>>>,
}

impl Context {
    pub fn init(device: &Arc<Device>, window: &Arc<Window>, surface: &Arc<Surface>) -> Self {
        let (swapchain, swapchain_images) = create_swapchain(&device, &window, &surface);

        let render_pass = create_renderpass(&device, swapchain.image_format());
        let deferred_pass = Arc::new(Subpass::from(render_pass.clone(), 0).unwrap());
        let lighting_pass = Arc::new(Subpass::from(render_pass.clone(), 1).unwrap());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let (framebuffers, color_buffers, normal_buffers) =
            create_framebuffers(&swapchain_images, &render_pass, memory_allocator.clone());
        let mut view_proj_buffers: Vec<Subbuffer<ViewProjUBO>> = vec![];
        let mut model_buffers: Vec<Subbuffer<ModelData>> = vec![];
        let mut ambient_buffers: Vec<Subbuffer<AmbientLightUBO>> = vec![];
        let mut directional_buffers: Vec<Vec<Subbuffer<DirectionalLightUBO>>> = vec![];

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
                    ViewProjUBO {
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

        for _ in 0..swapchain_images.len() {
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
                    DirectionalLightUBO {
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
                    DirectionalLightUBO {
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
                    DirectionalLightUBO {
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
        let frames_resources_free: Vec<Option<Box<dyn GpuFuture>>> = {
            let mut vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                vec.push(previous_frame_end);
            }
            vec
        };
        let (view_proj_sets, model_sets, directional_sets, ambient_sets) = create_descriptor_sets(
            &device,
            &deferred_pipeline,
            &directional_pipeline,
            &ambient_pipeline,
            &view_proj_buffers,
            &model_buffers,
            &ambient_buffers,
            &directional_buffers,
            &color_buffers,
            &normal_buffers,
            swapchain_images.len(),
        );
        println!(
            "##########init directional sets len:{}\ninner length:{}",
            directional_sets.len(),
            directional_sets[0].len()
        );

        Context {
            recreate_swapchain: false,
            current_frame: 0,
            frame_counter: 0,
            swapchain,
            framebuffers,
            frames_resources_free,
            render_pass,
            memory_allocator,
            viewport,

            color_buffers,
            normal_buffers,

            model_buffers,
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

    view_proj_buffers: &[Subbuffer<ViewProjUBO>],
    model_buffers: &[Subbuffer<ModelData>],
    ambient_buffers: &[Subbuffer<AmbientLightUBO>],
    directional_buffers: &[Vec<Subbuffer<DirectionalLightUBO>>],
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
            set_count: 4 * swapchain_image_count,
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

    for i in 0..swapchain_image_count {
        let view_proj_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            view_proj_layout.clone(),
            [WriteDescriptorSet::buffer(0, view_proj_buffers[i].clone())],
            [],
        )
        .unwrap();
        view_proj_sets.push(view_proj_set);

        let model_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            model_layout.clone(),
            [WriteDescriptorSet::buffer(0, model_buffers[i].clone())],
            [],
        )
        .unwrap();
        model_sets.push(model_set);

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
            println!("directional_subsetslength:{}", directional_subset.len());
        }
        directional_sets.push(directional_subset);
        println!("directional_sets length:{}", directional_sets.len());

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
    println!(
        "directional_sets in create descriptor sets length:{}",
        directional_sets.len()
    );
    println!("directional_subsetslength:{}", directional_sets[0].len());
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

    let vertex_input_state = resources::Vertex::per_vertex()
        .definition(&deferred_vert.info().input_interface)
        .unwrap();

    let dummy_vertex_input_state = resources::DummyVertex::per_vertex()
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
