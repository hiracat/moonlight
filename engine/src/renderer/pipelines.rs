use std::ffi::{self, c_void};
use std::path::PathBuf;
use std::{fs, marker};
use std::{
    path::{self, Path},
    ptr,
};

use ash::vk;
use rspirv_reflect::{self as rr, rspirv::binary::Assemble, Reflection};

use crate::ecs::{Opt, World};
use crate::renderer::resources::{Material, Mesh, ResourceManager};
use crate::renderer::resources::{Texture, Vertex};
use crate::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    ecs::Req,
    renderer::draw::{DrawJob, GEOMETRY_SUBPASS, LIGHTING_SUBPASS},
};
pub struct PipelineBundle {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,

    //PERF: this eventually should be replaced with a system that knows what components have a gpu
    //side and queries them all and writes, but too much effort for now(would probably need a
    //proc-marco)
    pub write_data_and_build_draw_jobs: Box<
        dyn Fn(
            &ash::Device,
            &mut ResourceManager,
            vk::DescriptorPool,
            &mut World,
            &Vec<vk::DescriptorSetLayout>,
            // for swapchain image descriptor set
            &vk::DescriptorSet,
        ) -> Vec<DrawJob>,
    >,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[repr(u32)]
#[derive(Debug)]
pub enum PipelineKey {
    Geometry = 0,
    Directional,
    Ambient,
    Point,
    // anything this and after is fairgame to be created from a number
    CustomStart,
    End = u32::MAX,
}

// CREATE ALL THE ENGINES BUILTIN GRAPHICS PIPELINES(I want to extend to being able to add your
// own)
pub fn create_builtin_graphics_pipelines(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> Vec<PipelineBundle> {
    let geometry = create_pipeline_layout_from_vert_frag(
        device,
        Path::new("shaders/geometry_vert.spv"),
        Path::new("shaders/geometry_frag.spv"),
    );

    let directional_layouts = create_pipeline_layout_from_vert_frag(
        device,
        Path::new("shaders/directional_vert.spv"),
        Path::new("shaders/directional_frag.spv"),
    );

    let ambient = create_pipeline_layout_from_vert_frag(
        device,
        Path::new("shaders/ambient_vert.spv"),
        Path::new("shaders/ambient_frag.spv"),
    );

    let point = create_pipeline_layout_from_vert_frag(
        device,
        Path::new("shaders/point_vert.spv"),
        Path::new("shaders/point_frag.spv"),
    );

    let geometry_desc = GraphicsPipelineDesc {
        renderpass: render_pass,
        pipeline_layout: geometry.1,
        shaders: &geometry.0,
        subpass_index: GEOMETRY_SUBPASS,
        tesselation_state: None,
        dynamic_state: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        multisample_state: MultisampleState {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: false,
            min_sample_shading: 0.0,
            sample_mask: None,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: true,
        },
        raster_state: RasterState {
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
        },
        input_assembly: InputAssemblyState {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
        },
        vertex_input_state: VertexInputState {
            vertex_attribute_descriptions: Vertex::get_vertex_attributes(),
            vertex_binding_descriptions: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
        },
        color_blend_state: ColorBlendState {
            logic_op: None,
            attachments: vec![
                vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::FALSE,
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                };
                3
            ],
            blend_constants: [0.0; 4],
        },
        depth_stencil_state: DepthStencilState {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
        },
        viewport_state: None,
    };

    let geometry_pipeline = create_graphics_pipeline(device, &geometry_desc).unwrap();

    let mut ambient_desc = geometry_desc;
    ambient_desc.subpass_index = LIGHTING_SUBPASS;
    ambient_desc.color_blend_state = ColorBlendState {
        logic_op: None,
        blend_constants: [0.0; 4],
        attachments: vec![vk::PipelineColorBlendAttachmentState {
            color_blend_op: vk::BlendOp::ADD,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE,
            alpha_blend_op: vk::BlendOp::MAX,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::TRUE,
            ..Default::default()
        }],
    };
    ambient_desc.pipeline_layout = ambient.1;
    ambient_desc.shaders = &ambient.0;
    let ambient_pipeline = create_graphics_pipeline(device, &ambient_desc).unwrap();

    let mut directional_desc = ambient_desc;

    directional_desc.pipeline_layout = directional_layouts.1;
    directional_desc.shaders = &directional_layouts.0;
    let directional_pipeline = create_graphics_pipeline(device, &directional_desc).unwrap();

    let mut point_desc = directional_desc;

    point_desc.pipeline_layout = point.1;
    point_desc.shaders = &point.0;
    let point_pipeline = create_graphics_pipeline(device, &point_desc).unwrap();

    let mut pipelines = Vec::new();
    pipelines.push(PipelineBundle {
        pipeline: geometry_pipeline,
        layout: geometry.1,
        descriptor_set_layouts: geometry.2,
        write_data_and_build_draw_jobs: Box::new(
            |device,
             resource_manager,
             descriptor_pool,
             world,
             descriptor_set_layouts,
             _swapchain_descriptor_set| {
                let mut jobs = Vec::new();
                let camera = world.get_resource::<Camera>().unwrap();
                let mut builder = DescriptorWriteBuilder::new();

                let camera_set = builder.add_uniform_buffer(
                    resource_manager,
                    descriptor_pool,
                    descriptor_set_layouts[1],
                    0,
                    &camera.as_ubo(),
                );

                //NOTE: defined in shader, 1 is the index of the camera set
                for entity in world.query::<(Req<Mesh>, Req<Transform>, Opt<Material>)>() {
                    let (_entityid, (mesh, transform, material)) = entity;
                    let model_set = builder.add_uniform_buffer(
                        resource_manager,
                        descriptor_pool,
                        descriptor_set_layouts[0],
                        0,
                        &transform.as_model_ubo(),
                    );
                    let image_set = builder.add_texture(
                        resource_manager,
                        descriptor_pool,
                        descriptor_set_layouts[2],
                        0,
                        material.copied(),
                    );

                    // Allocate/update descriptor set for this draw call
                    // Set 0: Per-frame data (camera matrices)
                    // Set 1: Per-object data (model matrix)
                    // Set 2: Material data (textures)
                    let descriptor_sets = vec![model_set, camera_set, image_set];
                    jobs.push(DrawJob {
                        mesh: Some(*mesh),
                        descriptor_sets,
                    });
                }

                builder.submit(device);
                jobs
            },
        ),
    });

    pipelines.push(PipelineBundle {
        pipeline: ambient_pipeline,
        layout: ambient.1,
        descriptor_set_layouts: ambient.2,
        write_data_and_build_draw_jobs: Box::new(
            |device,
             resource_manager,
             descriptor_pool,
             world,
             descriptor_set_layouts,
             swapchain_descriptor_set| {
                let mut builder = DescriptorWriteBuilder::new();
                let ambient = world.get_resource::<AmbientLight>().unwrap();
                let ambient_set = builder.add_uniform_buffer(
                    resource_manager,
                    descriptor_pool,
                    descriptor_set_layouts[1],
                    0,
                    &ambient.as_ubo(),
                );
                builder.submit(device);

                vec![DrawJob {
                    mesh: None,
                    descriptor_sets: vec![*swapchain_descriptor_set, ambient_set],
                }]
            },
        ),
    });
    pipelines.push(PipelineBundle {
        pipeline: directional_pipeline,
        layout: directional_layouts.1,
        descriptor_set_layouts: directional_layouts.2,
        write_data_and_build_draw_jobs: Box::new(
            |device,
             resource_manager,
             descriptor_pool,
             world,
             descriptor_set_layouts,
             swapchain_descriptor_set| {
                let mut builder = DescriptorWriteBuilder::new();
                let directional = world.get_resource::<DirectionalLight>().unwrap();
                let directional_set = builder.add_uniform_buffer(
                    resource_manager,
                    descriptor_pool,
                    descriptor_set_layouts[1],
                    0,
                    &directional.as_ubo(),
                );
                builder.submit(device);

                vec![DrawJob {
                    mesh: None,
                    descriptor_sets: vec![*swapchain_descriptor_set, directional_set],
                }]
            },
        ),
    });
    pipelines.push(PipelineBundle {
        pipeline: point_pipeline,
        layout: point.1,
        descriptor_set_layouts: point.2,
        write_data_and_build_draw_jobs: Box::new(
            |device,
             resource_manager,
             descriptor_pool,
             world,
             descriptor_set_layouts,
             swapchain_descriptor_set| {
                let mut jobs = Vec::new();
                let mut builder = DescriptorWriteBuilder::new();

                for entity in world.query::<(Req<PointLight>, Req<Transform>)>() {
                    let (_entityid, (point, transform)) = entity;
                    let point_set = builder.add_uniform_buffer(
                        resource_manager,
                        descriptor_pool,
                        descriptor_set_layouts[1],
                        0,
                        &point.as_ubo(transform),
                    );

                    let descriptor_sets = vec![*swapchain_descriptor_set, point_set];
                    jobs.push(DrawJob {
                        mesh: None,
                        descriptor_sets,
                    });
                }

                builder.submit(device);
                jobs
            },
        ),
    });
    pipelines
}

/*--------------PIPELINE CREATION HELPERS-------------
-----------------------------------------------------*/
pub struct VertexInputState {
    pub vertex_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
}
pub struct InputAssemblyState {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool,
}
pub struct SpecalizationInfo {
    pub map_entries: Vec<vk::SpecializationMapEntry>,
    pub data: Vec<u8>,
}
pub struct Viewport {
    pub viewport: Vec<vk::Viewport>,
    pub scissor: Vec<vk::Rect2D>,
}
pub struct ShaderStage {
    pub shader: vk::ShaderModule,
    pub kind: vk::ShaderStageFlags,
    //NOTE: specalization info is like compiler injected #defines, set constants to avoid
    //recompiling shader
    pub entry_point: ffi::CString,
    pub specalization_info: Option<SpecalizationInfo>,
}

pub struct RasterState {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

pub struct MultisampleState {
    pub rasterization_samples: vk::SampleCountFlags,
    pub sample_shading_enable: bool,
    pub min_sample_shading: f32,
    pub sample_mask: Option<u32>,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

pub struct ColorBlendState {
    pub logic_op: Option<vk::LogicOp>,
    pub attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    pub blend_constants: [f32; 4],
}
pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: vk::StencilOpState,
    pub back: vk::StencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

pub struct TesselationState {
    pub patch_control_points: u32,
}

pub struct GraphicsPipelineDesc<'a> {
    pub shaders: &'a [ShaderStage],
    pub vertex_input_state: VertexInputState,
    pub input_assembly: InputAssemblyState,
    pub viewport_state: Option<Viewport>,
    pub raster_state: RasterState,
    pub multisample_state: MultisampleState,
    pub depth_stencil_state: DepthStencilState,
    pub color_blend_state: ColorBlendState,
    pub dynamic_state: Vec<vk::DynamicState>,
    pub tesselation_state: Option<TesselationState>,
    pub pipeline_layout: vk::PipelineLayout,
    pub renderpass: vk::RenderPass,
    pub subpass_index: u32,
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    desc: &GraphicsPipelineDesc,
) -> Result<vk::Pipeline, (vk::Pipeline, vk::Result)> {
    let mut stages = Vec::new();
    for shader in desc.shaders {
        let specalization_info;
        match &shader.specalization_info {
            Some(info) => {
                specalization_info = Some(vk::SpecializationInfo {
                    map_entry_count: info.map_entries.len() as u32,
                    p_map_entries: info.map_entries.as_ptr(),
                    p_data: info.data.as_ptr() as *const c_void,
                    data_size: info.data.len(),
                    _marker: marker::PhantomData,
                });
            }
            None => {
                specalization_info = None;
            }
        }

        stages.push(vk::PipelineShaderStageCreateInfo {
            p_specialization_info: specalization_info.as_ref().map_or(ptr::null(), |info| info),
            stage: shader.kind,
            module: shader.shader,
            p_name: shader.entry_point.as_ptr(),
            ..Default::default()
        });
    }

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        p_vertex_binding_descriptions: desc.vertex_input_state.vertex_binding_descriptions.as_ptr(),
        p_vertex_attribute_descriptions: desc
            .vertex_input_state
            .vertex_attribute_descriptions
            .as_ptr(),
        vertex_binding_description_count: desc.vertex_input_state.vertex_binding_descriptions.len()
            as u32,
        vertex_attribute_description_count: desc
            .vertex_input_state
            .vertex_attribute_descriptions
            .len() as u32,
        ..Default::default()
    };

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
        topology: desc.input_assembly.topology,
        primitive_restart_enable: desc.input_assembly.primitive_restart_enable.into(),
        ..Default::default()
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        p_scissors: desc
            .viewport_state
            .as_ref()
            .map_or(ptr::null(), |x| x.scissor.as_ptr()),
        p_viewports: desc
            .viewport_state
            .as_ref()
            .map_or(ptr::null(), |x| x.viewport.as_ptr()),
        scissor_count: desc
            .viewport_state
            .as_ref()
            .map_or(1, |x| x.scissor.len() as u32),
        viewport_count: desc
            .viewport_state
            .as_ref()
            .map_or(1, |x| x.viewport.len() as u32),
        ..Default::default()
    };

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
        depth_clamp_enable: desc.raster_state.depth_clamp_enable.into(),
        rasterizer_discard_enable: desc.raster_state.rasterizer_discard_enable.into(),
        polygon_mode: desc.raster_state.polygon_mode,
        cull_mode: desc.raster_state.cull_mode,
        front_face: desc.raster_state.front_face,
        depth_bias_enable: desc.raster_state.depth_bias_enable.into(),
        depth_bias_constant_factor: desc.raster_state.depth_bias_constant_factor,
        depth_bias_clamp: desc.raster_state.depth_bias_clamp,
        depth_bias_slope_factor: desc.raster_state.depth_bias_slope_factor,
        line_width: desc.raster_state.line_width,

        ..Default::default()
    };

    let multisample_state = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: desc.multisample_state.rasterization_samples,
        sample_shading_enable: desc.multisample_state.sample_shading_enable.into(),
        min_sample_shading: desc.multisample_state.min_sample_shading,
        p_sample_mask: desc
            .multisample_state
            .sample_mask
            .as_ref()
            .map_or(ptr::null(), |x| x),
        alpha_to_coverage_enable: desc.multisample_state.alpha_to_coverage_enable.into(),
        alpha_to_one_enable: desc.multisample_state.alpha_to_one_enable.into(),

        ..Default::default()
    };

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        logic_op_enable: desc.color_blend_state.logic_op.is_some().into(),
        logic_op: desc.color_blend_state.logic_op.unwrap_or(vk::LogicOp::AND),
        attachment_count: desc.color_blend_state.attachments.len() as u32,
        p_attachments: desc.color_blend_state.attachments.as_ptr(),
        blend_constants: desc.color_blend_state.blend_constants,

        ..Default::default()
    };
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable: desc.depth_stencil_state.depth_test_enable.into(),
        depth_write_enable: desc.depth_stencil_state.depth_write_enable.into(),

        depth_compare_op: desc.depth_stencil_state.depth_compare_op,
        depth_bounds_test_enable: desc.depth_stencil_state.depth_bounds_test_enable.into(),

        stencil_test_enable: desc.depth_stencil_state.stencil_test_enable.into(),
        front: desc.depth_stencil_state.front,
        back: desc.depth_stencil_state.back,

        min_depth_bounds: desc.depth_stencil_state.min_depth_bounds,
        max_depth_bounds: desc.depth_stencil_state.max_depth_bounds,

        ..Default::default()
    };
    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: desc.dynamic_state.len() as u32,
        p_dynamic_states: desc.dynamic_state.as_ptr(),

        ..Default::default()
    };

    let tesselation_state = match &desc.tesselation_state {
        Some(x) => Some(vk::PipelineTessellationStateCreateInfo {
            patch_control_points: x.patch_control_points,

            ..Default::default()
        }),
        None => None,
    };

    let create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: stages.as_ptr(),
        stage_count: stages.len() as u32,
        // How vertex data is read from the vertex buffers into the vertex shader.
        p_vertex_input_state: &vertex_input_state,
        // How vertices are arranged into primitive shapes. The default primitive shape
        // is a triangle.
        p_input_assembly_state: &input_assembly_state,
        // How primitives are transformed and clipped to fit the framebuffer. We use a
        // resizable viewport, set to draw over the entire window.
        p_viewport_state: &viewport_state,
        // How polygons are culled and converted into a raster of pixels. The default
        // value does not perform any culling.
        p_rasterization_state: &rasterization_state,
        // How multiple fragment shader samples are converted to a single pixel value.
        // The default value does not perform any multisampling.
        p_multisample_state: &multisample_state,
        // How pixel values are combined with the values already present in the
        // framebuffer. The default value overwrites the old value with the new one,
        // without any blending.
        p_color_blend_state: &color_blend_state,

        p_depth_stencil_state: &depth_stencil_state,

        // Dynamic states allows us to specify parts of the pipeline settings when
        // recording the command buffer, before we perform drawing. Here, we specify
        // that the viewport should be dynamic.
        p_dynamic_state: &dynamic_state,
        p_tessellation_state: tesselation_state.map_or(ptr::null(), |x| &x),

        layout: desc.pipeline_layout,
        // the renderpass this graphics pipeline is one
        render_pass: desc.renderpass,
        // the subpass index
        subpass: desc.subpass_index,

        // if deriving from a graphics pipeline, the index
        base_pipeline_index: 0,
        // and the handle to that pipeline
        base_pipeline_handle: vk::Pipeline::null(),
        ..Default::default()
    };

    unsafe {
        Ok(
            match device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            {
                Ok(it) => it[0],
                Err(err) => {
                    let pipeline = err.0[0];
                    let err = err.1;
                    return Err((pipeline, err));
                }
            },
        )
    }
}

/*-------------------SHADER REFLECTION----------------
-----------------------------------------------------*/
const ASM_ENTRY_POINT_EXECUTION_MODEL_IDX: usize = 0;
const ASM_ENTRY_POINT_NAME_IDX: usize = 2;
pub fn create_pipeline_layout_from_vert_frag(
    device: &ash::Device,
    vertex_path: &path::Path,
    fragment_path: &path::Path,
) -> (
    Vec<ShaderStage>,
    vk::PipelineLayout,
    Vec<vk::DescriptorSetLayout>,
) {
    let vertex_reflection = load_path_data(vertex_path);
    let fragment_reflection = load_path_data(fragment_path);

    assert_eq!(
        vertex_reflection.0.entry_points.len(),
        1,
        "only single entry point supported"
    );
    assert_eq!(
        fragment_reflection.0.entry_points.len(),
        1,
        "only single entry point supported"
    );

    let code = vertex_reflection.0.assemble();
    let vertex_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: code.as_ptr(),
        code_size: code.len() * 4, // convert from u32 to bytes
        ..Default::default()
    };
    let vertex_module = unsafe {
        device
            .create_shader_module(&vertex_module_create_info, None)
            .unwrap()
    };
    let vertex_stage = ShaderStage {
        shader: vertex_module,
        kind: get_shader_kind(&vertex_reflection),
        entry_point: get_entry_name(&vertex_reflection),
        specalization_info: None,
    };

    let code = fragment_reflection.0.assemble();
    let fragment_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: code.as_ptr(),
        code_size: code.len() * 4, // convert from u32 to bytes
        ..Default::default()
    };
    let fragment_module = unsafe {
        device
            .create_shader_module(&fragment_module_create_info, None)
            .unwrap()
    };
    let fragment_stage = ShaderStage {
        shader: fragment_module,
        kind: get_shader_kind(&fragment_reflection),
        entry_point: get_entry_name(&fragment_reflection),
        specalization_info: None,
    };

    let vertex_descriptor_sets = vertex_reflection.get_descriptor_sets().unwrap();
    let fragment_descriptor_sets = fragment_reflection.get_descriptor_sets().unwrap();
    let max_set = *vertex_descriptor_sets
        .keys()
        .max()
        .unwrap_or(&0)
        .max(fragment_descriptor_sets.keys().max().unwrap_or(&0));
    //NOTE: CREATE SET LAYOUT PER SET IN THE FILE
    let mut set_layouts = Vec::new();
    for set_index in 0..=max_set {
        let mut bindings = Vec::new();
        let vertex_bindings = vertex_descriptor_sets.get(&set_index);
        let fragment_bindings = fragment_descriptor_sets.get(&set_index);

        let max_binding = {
            let max_vertex = vertex_bindings.map_or(None, |x| x.keys().max());
            let max_fragment = fragment_bindings.map_or(None, |x| x.keys().max());
            assert!(max_vertex.is_some() || max_fragment.is_some());
            *max_vertex.unwrap_or(&0).max(max_fragment.unwrap_or(&0))
        };

        for binding_index in 0..=max_binding {
            let vertex_binding_info = vertex_bindings.and_then(|x| x.get(&binding_index));
            let fragment_binding_info = fragment_bindings.and_then(|x| x.get(&binding_index));

            if vertex_binding_info.is_none() && fragment_binding_info.is_none() {
                continue;
            }

            let mut accumulated_binding = vk::DescriptorSetLayoutBinding {
                stage_flags: vk::ShaderStageFlags::empty(),
                binding: binding_index,
                descriptor_type: vk::DescriptorType::from_raw(0),
                descriptor_count: 0,
                p_immutable_samplers: ptr::null(),
                ..Default::default()
            };

            if let Some(info) = vertex_binding_info {
                accumulated_binding = vk::DescriptorSetLayoutBinding {
                    descriptor_count: match info.binding_count {
                        rspirv_reflect::BindingCount::One => 1,
                        rspirv_reflect::BindingCount::StaticSized(x) => x as u32,
                        rspirv_reflect::BindingCount::Unbounded => 1024,
                    },
                    stage_flags: accumulated_binding.stage_flags | vk::ShaderStageFlags::VERTEX,
                    descriptor_type: vk::DescriptorType::from_raw(info.ty.0 as i32),
                    ..accumulated_binding
                }
            }
            if let Some(info) = fragment_binding_info {
                accumulated_binding = vk::DescriptorSetLayoutBinding {
                    descriptor_count: match info.binding_count {
                        rspirv_reflect::BindingCount::One => 1,
                        rspirv_reflect::BindingCount::StaticSized(x) => x as u32,
                        rspirv_reflect::BindingCount::Unbounded => 1024,
                    },
                    stage_flags: accumulated_binding.stage_flags | vk::ShaderStageFlags::FRAGMENT,
                    descriptor_type: vk::DescriptorType::from_raw(info.ty.0 as i32),
                    ..accumulated_binding
                }
            }
            bindings.push(accumulated_binding);
        }

        let set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: bindings.as_ptr(),
            binding_count: bindings.len() as u32,
            ..Default::default()
        };
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(&set_layout_create_info, None)
                .unwrap()
        };
        set_layouts.push(set_layout);
    }
    let mut push_constant_ranges = Vec::new();

    // PERF: better to make them
    // not overlap if posslbe tho later
    if let Some(range) = vertex_reflection.get_push_constant_range().unwrap() {
        let range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            size: range.size,
            offset: range.offset,
        };
        push_constant_ranges.push(range);
    }
    if let Some(range) = fragment_reflection.get_push_constant_range().unwrap() {
        let range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            size: range.size,
            offset: range.offset,
        };
        push_constant_ranges.push(range);
    }

    // NOTE: per shader, and can overlap between vert and frag so is fine,
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: push_constant_ranges.len() as u32,
        p_push_constant_ranges: push_constant_ranges.as_ptr(),

        ..Default::default()
    };
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .unwrap()
    };

    (
        vec![vertex_stage, fragment_stage],
        pipeline_layout,
        set_layouts,
    )
}

fn load_path_data(path: &path::Path) -> Reflection {
    let out_dir = PathBuf::from(env!("OUT_DIR"));
    let full_path: PathBuf = out_dir.join(&path);
    dbg!(&full_path);
    let code = fs::read(full_path).expect("failed to read file");
    rr::Reflection::new_from_spirv(&code).unwrap()
}
fn get_entry_name(reflection: &Reflection) -> ffi::CString {
    let entry_point_name =
        reflection.0.entry_points[0].operands[ASM_ENTRY_POINT_NAME_IDX].unwrap_literal_string();
    let entry_cstr = ffi::CString::new(entry_point_name).unwrap();
    entry_cstr
}
fn get_shader_kind(reflection: &Reflection) -> vk::ShaderStageFlags {
    let raw_stage = reflection.0.entry_points[0].operands[ASM_ENTRY_POINT_EXECUTION_MODEL_IDX]
        .unwrap_execution_model();
    let shader_kind = vk::ShaderStageFlags::from_raw(0b1 << (raw_stage as u32));
    shader_kind
}

/// helper for making descriptor writes
/// besides being extremely boilerplate heavy, making descriptor writes requires managing stable
/// memory addresses for the bufferinfo sub structs, so this abstracts that away
struct DescriptorWriteBuilder<'a> {
    buffer_infos: Vec<Box<vk::DescriptorBufferInfo>>,
    image_infos: Vec<Box<vk::DescriptorImageInfo>>,
    writes: Vec<vk::WriteDescriptorSet<'a>>,
}

impl DescriptorWriteBuilder<'_> {
    fn new() -> Self {
        Self {
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
        }
    }

    fn add_uniform_buffer<T: bytemuck::Pod>(
        &mut self,
        resource_manager: &mut ResourceManager,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binding: u32,
        data: &T,
    ) -> vk::DescriptorSet {
        let size = size_of::<T>();
        let (buffer, offset) = resource_manager.ring_buffer.allocate(size);
        resource_manager.ring_buffer.write(data, offset);

        let descriptor_set =
            resource_manager.allocate_temp_descriptor_set(descriptor_set_layout, descriptor_pool);

        // Store buffer info (we need stable addresses)
        self.buffer_infos.push(Box::new(vk::DescriptorBufferInfo {
            buffer,
            offset,
            range: size as u64,
        }));

        // Record the write
        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_buffer_info: self.buffer_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    fn add_texture(
        &mut self,
        resource_manager: &mut ResourceManager,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binding: u32,
        texture: Option<Material>,
    ) -> vk::DescriptorSet {
        let descriptor_set =
            resource_manager.allocate_temp_descriptor_set(descriptor_set_layout, descriptor_pool);

        let image = match texture {
            Some(x) => resource_manager
                .get_texture(x.albedo)
                .expect("texture should exist"),
            None => resource_manager.default_texture(),
        };

        self.image_infos.push(Box::new(vk::DescriptorImageInfo {
            sampler: image.sampler,
            image_view: image.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }));

        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_image_info: self.image_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    fn submit(self, device: &ash::Device) {
        unsafe {
            device.update_descriptor_sets(&self.writes, &[]);
        }
    }
}
