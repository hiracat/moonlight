use std::collections::BTreeMap;
use std::ffi::{self, c_void};
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs, marker};
use std::{
    path::{self, Path},
    ptr,
};

use ash::vk;
use bytemuck::bytes_of;
use educe::Educe;
use rspirv_reflect::{self as rr, Reflection, rspirv::binary::Assemble};

use crate::core::TerrainMap;
use crate::ecs::{EntityName, Not, NotM, Opt, OptM, ReqM, World};
use crate::renderers::world::descriptors::{BindingData, DescriptorManager};
use crate::renderers::world::draw::{DrawStyle, FRAMES_IN_FLIGHT};
use crate::renderers::world::rendergraph::{ImageDesc, ImageId, RenderGraph};
use crate::resources::{Animated, AnimatedVertex, IsVertex, ResourceManager, Vertex};
use crate::resources::{Material, Mesh, Skybox};
use crate::ubo::{
    AmbientLightUBO, CameraInverseUBO, CameraUBO, DirectionalLightUBO, MaterialUBO, ModelUBO,
    PointLightUBO, TerrainUBO,
};
use crate::vulkan::SharedAllocator;
use crate::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    ecs::Req,
    renderers::world::draw::DrawJob,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PipelineHandle {
    pub is_compute: bool,
    pub arr_index: usize,
    pub name: &'static str,
}

#[derive(Educe)]
#[educe(Debug)]
pub struct PipelineManager {
    pub pipelines: Vec<Option<PipelineBundle>>,
    pub pipeline_names: Vec<&'static str>,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,
}
impl PipelineManager {
    fn new(device: Arc<ash::Device>) -> Self {
        Self {
            pipeline_names: Vec::new(),
            pipelines: Vec::new(),
            device,
        }
    }
    fn allocate_handle(&mut self, name: &'static str) -> PipelineHandle {
        self.pipelines.push(None);
        self.pipeline_names.push(name);
        PipelineHandle {
            is_compute: false,
            name,
            arr_index: self.pipelines.len() - 1,
        }
    }
    fn allocate_compute_handle(&mut self, name: &'static str) -> PipelineHandle {
        self.pipelines.push(None);
        self.pipeline_names.push(name);
        PipelineHandle {
            is_compute: true,
            name,
            arr_index: self.pipelines.len() - 1,
        }
    }
    fn add_compute_pipeline(
        &mut self,
        handle: PipelineHandle,
        shader: ShaderStage,
        layout: vk::PipelineLayout,
        pipeline_fn: PipelineFn,
    ) {
        assert!(handle.is_compute);

        let pipeline = create_compute_pipeline(&self.device, shader, layout).unwrap();

        let bundle = PipelineBundle {
            name: handle.name,
            pipeline,
            is_compute: false,
            write_data_and_build_draw_jobs: pipeline_fn,
            device: self.device.clone(),
        };
        self.pipelines[handle.arr_index] = Some(bundle);
    }
    fn add_pipeline(
        &mut self,
        handle: PipelineHandle,
        desc: &GraphicsPipelineDesc,
        shaders: &[ShaderStage],
        layout: vk::PipelineLayout,
        pipeline_fn: PipelineFn,
    ) {
        assert!(!handle.is_compute);
        let pipeline = create_graphics_pipeline(&self.device, desc, shaders, layout).unwrap();

        let bundle = PipelineBundle {
            name: handle.name,
            is_compute: true,
            pipeline,
            write_data_and_build_draw_jobs: pipeline_fn,
            device: self.device.clone(),
        };
        self.pipelines[handle.arr_index] = Some(bundle);
    }
}

type PipelineFn = Box<
    dyn Fn(
        &mut World,
        &mut ResourceManager,
        &mut DescriptorManager,
        PipelineHandle,
    ) -> Vec<DrawJob>,
>;

#[derive(Educe)]
#[educe(Debug)]
pub struct PipelineBundle {
    pub name: &'static str,
    pub pipeline: vk::Pipeline,
    is_compute: bool,

    #[educe(Debug(ignore))]
    pub write_data_and_build_draw_jobs: PipelineFn,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,
}
impl Drop for PipelineBundle {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

// CREATE ALL THE ENGINES BUILTIN GRAPHICS PIPELINES(I want to extend to being able to add your
// own)
pub fn create_builtin_graphics_pipelines(
    device: Arc<ash::Device>,
    allocator: SharedAllocator,
    swapchain_image_format: vk::Format,
    depth_format: vk::Format,
    color_attachment_formats: &[vk::Format],
) -> (
    RenderGraph,
    PipelineManager,
    [DescriptorManager; FRAMES_IN_FLIGHT],
    ImageId,
) {
    let geometry_desc = GraphicsPipelineDesc {
        tesselation_state: None,
        viewport_state: None,
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
            cull_mode: vk::CullModeFlags::BACK,
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
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
        },
        depth_attachment_format: Some(depth_format),
        color_attachment_formats: color_attachment_formats.to_vec(),
    };

    let animated_geometry_desc = GraphicsPipelineDesc {
        vertex_input_state: VertexInputState {
            vertex_attribute_descriptions: AnimatedVertex::get_vertex_attributes(),
            vertex_binding_descriptions: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<AnimatedVertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
        },
        ..geometry_desc.clone()
    };
    let clipped_geometry_desc = GraphicsPipelineDesc {
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
        ..geometry_desc.clone()
    };

    let lighting_desc = GraphicsPipelineDesc {
        color_attachment_formats: vec![swapchain_image_format],
        color_blend_state: ColorBlendState {
            logic_op: None,
            blend_constants: [0.0; 4],
            attachments: vec![
                vk::PipelineColorBlendAttachmentState {
                    color_blend_op: vk::BlendOp::ADD,
                    src_color_blend_factor: vk::BlendFactor::ONE,
                    dst_color_blend_factor: vk::BlendFactor::ONE,
                    alpha_blend_op: vk::BlendOp::MAX,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ONE,
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    blend_enable: vk::TRUE,
                };
                1
            ],
        },
        depth_stencil_state: DepthStencilState {
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::ALWAYS,
            ..geometry_desc.depth_stencil_state.clone()
        },
        ..geometry_desc.clone()
    };

    let skybox_desc = GraphicsPipelineDesc {
        color_blend_state: ColorBlendState {
            logic_op: None,
            attachments: vec![
                vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::FALSE,
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                };
                1
            ],
            blend_constants: [0.0; 4],
        },
        depth_stencil_state: DepthStencilState {
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::EQUAL,
            ..geometry_desc.depth_stencil_state.clone()
        },
        ..lighting_desc.clone()
    };

    // --- reflect shaders ---
    let static_geometry_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/static_geometry_vert.spv"),
        Path::new("shaders/geometry_frag.spv"),
    );
    let animated_geometry_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/animated_geometry_vert.spv"),
        Path::new("shaders/geometry_frag.spv"),
    );
    let clipped_geometry_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/static_geometry_vert.spv"),
        Path::new("shaders/clipped_geometry_frag.spv"),
    );
    let terrain_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/terrain_vert.spv"),
        Path::new("shaders/terrain_frag.spv"),
    );
    let ambient_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/ambient_vert.spv"),
        Path::new("shaders/ambient_frag.spv"),
    );
    let directional_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/directional_vert.spv"),
        Path::new("shaders/directional_frag.spv"),
    );
    let point_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/point_vert.spv"),
        Path::new("shaders/point_frag.spv"),
    );
    let skybox_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/skybox_vert.spv"),
        Path::new("shaders/skybox_frag.spv"),
    );
    let mut pipeline_manager = PipelineManager::new(device.clone());

    // --- allocate handles ---
    let static_geometry = pipeline_manager.allocate_handle("static_geometry");
    let animated_geometry = pipeline_manager.allocate_handle("animated_geometry");
    let clipped_geometry = pipeline_manager.allocate_handle("clipped_geometry");
    let terrain = pipeline_manager.allocate_handle("terrain");
    let ambient = pipeline_manager.allocate_handle("ambient");
    let directional = pipeline_manager.allocate_handle("directional");
    let point = pipeline_manager.allocate_handle("point");
    let skybox = pipeline_manager.allocate_handle("skybox");

    // --- build descriptor managers, get layouts ---
    let mut descriptor_managers: Vec<DescriptorManager> = (0..FRAMES_IN_FLIGHT)
        .map(|_| DescriptorManager::new(device.clone(), allocator.clone()))
        .collect();

    let mut register = |handle: PipelineHandle,
                        vert: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
                        frag: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>|
     -> vk::PipelineLayout {
        let layout = descriptor_managers[0].add_pipeline(handle, vert.clone(), frag.clone());
        for dm in descriptor_managers.iter_mut().skip(1) {
            dm.add_pipeline(handle, vert.clone(), frag.clone());
        }
        layout
    };

    let static_geometry_layout = register(
        static_geometry,
        static_geometry_data.1,
        static_geometry_data.2,
    );
    let animated_geometry_layout = register(
        animated_geometry,
        animated_geometry_data.1,
        animated_geometry_data.2,
    );
    let clipped_geometry_layout = register(
        clipped_geometry,
        clipped_geometry_data.1,
        clipped_geometry_data.2,
    );
    let terrain_layout = register(terrain, terrain_data.1, terrain_data.2);
    let ambient_layout = register(ambient, ambient_data.1, ambient_data.2);
    let directional_layout = register(directional, directional_data.1, directional_data.2);
    let point_layout = register(point, point_data.1, point_data.2);
    let skybox_layout = register(skybox, skybox_data.1, skybox_data.2);

    let mut graph = RenderGraph::new();

    let mut final_color = graph.add_image(ImageDesc::Imported {
        name: "final_color",
        format: swapchain_image_format,
    });
    let mut albedo = graph.add_image(ImageDesc::Managed {
        name: "albedo",
        format: vk::Format::A2B10G10R10_UNORM_PACK32,
    });
    let mut normal = graph.add_image(ImageDesc::Managed {
        name: "normal",
        format: vk::Format::R16G16B16A16_SFLOAT,
    });
    let mut position = graph.add_image(ImageDesc::Managed {
        name: "position",
        format: vk::Format::R32G32B32A32_SFLOAT,
    });
    let mut depth = graph.add_image(ImageDesc::Managed {
        name: "depth",
        format: vk::Format::D32_SFLOAT,
    });

    let albedo_id = albedo.id;
    let normal_id = normal.id;
    let position_id = position.id;

    pipeline_manager.add_pipeline(
        static_geometry,
        &geometry_desc,
        static_geometry_data.0.as_ref(),
        static_geometry_layout,
        Box::new(
            |world: &mut World,
             _resource_manager: &mut ResourceManager,
             descriptor_manager: &mut DescriptorManager,
             handle: PipelineHandle| {
                let mut jobs = Vec::new();
                let camera = world.get_resource::<Camera>().unwrap();

                let camera_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&CameraUBO::from(camera)).to_vec(),
                    },
                );

                for entity in
                    world.query::<(Req<Mesh>, Req<Transform>, Opt<Material>, Not<Animated>)>()
                {
                    let (_entityid, (mesh, transform, material)) = entity;
                    if mesh.animated {
                        continue;
                    }
                    if let Some(material) = material
                        && material.alpha_clip.is_some()
                    {
                        continue;
                    }

                    let model_handle = descriptor_manager.request_bind(
                        handle,
                        0,
                        0,
                        BindingData::Uniform {
                            data: bytes_of(&ModelUBO::from(transform)).to_vec(),
                        },
                    );
                    let image_handle = descriptor_manager.request_bind(
                        handle,
                        2,
                        0,
                        BindingData::Texture {
                            texture: material.map(|x| x.albedo).unwrap_or_default(),
                        },
                    );

                    jobs.push(DrawJob {
                        mesh: DrawStyle::Mesh(*mesh),
                        descriptor_sets: vec![model_handle, camera_handle, image_handle],
                    });
                }

                jobs
            },
        ),
    );

    pipeline_manager.add_pipeline(
        animated_geometry,
        &animated_geometry_desc,
        &animated_geometry_data.0,
        animated_geometry_layout,
        Box::new(
            |world: &mut World,
             resource_manager: &mut ResourceManager,
             descriptor_manager: &mut DescriptorManager,
             handle: PipelineHandle| {
                let mut jobs = Vec::new();
                let camera = world.get_resource::<Camera>().unwrap();

                let camera_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&CameraUBO::from(camera)).to_vec(),
                    },
                );

                for entity in world
                    .query_mut::<(ReqM<Mesh>, ReqM<Transform>, OptM<Material>, OptM<Animated>)>()
                {
                    let (_entityid, (mesh, transform, material, _animation)) = entity;
                    if !mesh.animated {
                        continue;
                    }
                    if let Some(ref material) = material
                        && material.alpha_clip.is_some()
                    {
                        continue;
                    }

                    let model_handle = descriptor_manager.request_bind(
                        handle,
                        0,
                        0,
                        BindingData::Uniform {
                            data: bytes_of(&ModelUBO::from(&*transform)).to_vec(),
                        },
                    );
                    let image_handle = descriptor_manager.request_bind(
                        handle,
                        2,
                        0,
                        BindingData::Texture {
                            texture: material.map(|x| x.albedo).unwrap_or_default(),
                        },
                    );
                    let transform_handle = descriptor_manager.request_bind(
                        handle,
                        3,
                        0,
                        BindingData::Ssbo {
                            buffer: resource_manager
                                .animation_resources
                                .skeleton_transform_handle,
                        },
                    );
                    let normal_handle = descriptor_manager.request_bind(
                        handle,
                        3,
                        1,
                        BindingData::Ssbo {
                            buffer: resource_manager.animation_resources.skeleton_normal_handle,
                        },
                    );

                    jobs.push(DrawJob {
                        mesh: DrawStyle::Mesh(*mesh),
                        descriptor_sets: vec![
                            model_handle,
                            camera_handle,
                            image_handle,
                            transform_handle,
                            normal_handle,
                        ],
                    });
                }

                jobs
            },
        ),
    );

    pipeline_manager.add_pipeline(
        clipped_geometry,
        &clipped_geometry_desc,
        &clipped_geometry_data.0,
        clipped_geometry_layout,
        Box::new(
            |world: &mut World,
             resource_manager: &mut ResourceManager,
             descriptor_manager: &mut DescriptorManager,
             handle: PipelineHandle| {
                let mut jobs = Vec::new();
                let camera = world.get_resource::<Camera>().unwrap();

                let camera_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&CameraUBO::from(camera)).to_vec(),
                    },
                );

                for entity in world
                    .query_mut::<(ReqM<Mesh>, ReqM<Transform>, ReqM<Material>, NotM<Animated>)>()
                {
                    let (_entityid, (mesh, transform, material)) = entity;
                    if material.alpha_clip.is_none() {
                        continue;
                    }

                    if mesh.animated {
                        continue;
                    }

                    let model_handle = descriptor_manager.request_bind(
                        handle,
                        0,
                        0,
                        BindingData::Uniform {
                            data: bytes_of(&ModelUBO::from(&*transform)).to_vec(),
                        },
                    );
                    let image_handle = descriptor_manager.request_bind(
                        handle,
                        2,
                        0,
                        BindingData::Texture {
                            texture: material.albedo,
                        },
                    );
                    let material_handle = descriptor_manager.request_bind(
                        handle,
                        2,
                        1,
                        BindingData::Uniform {
                            data: bytes_of(&Into::<MaterialUBO>::into(&*material)).to_owned(),
                        },
                    );

                    jobs.push(DrawJob {
                        mesh: DrawStyle::Mesh(*mesh),
                        descriptor_sets: vec![
                            model_handle,
                            camera_handle,
                            image_handle,
                            material_handle,
                        ],
                    });
                }

                jobs
            },
        ),
    );

    pipeline_manager.add_pipeline(
        terrain,
        &geometry_desc,
        &terrain_data.0,
        terrain_layout,
        Box::new(
            |world: &mut World,
             _resource_manager: &mut ResourceManager,
             descriptor_manager: &mut DescriptorManager,
             handle: PipelineHandle| {
                let camera = world.get_resource::<Camera>().unwrap();
                let heightmap = world.get_resource::<TerrainMap>().unwrap();

                let height_map_handle = descriptor_manager.request_bind(
                    handle,
                    0,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&TerrainUBO::from(heightmap)).to_vec(),
                    },
                );
                let camera_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&CameraUBO::from(camera)).to_vec(),
                    },
                );
                let height_texture_handle = descriptor_manager.request_bind(
                    handle,
                    2,
                    0,
                    BindingData::Texture {
                        texture: heightmap.map,
                    },
                );

                vec![DrawJob {
                    mesh: DrawStyle::VertexCount(
                        (heightmap.resolution - 1) * (heightmap.resolution - 1) * 6,
                    ),
                    descriptor_sets: vec![height_map_handle, camera_handle, height_texture_handle],
                }]
            },
        ),
    );

    pipeline_manager.add_pipeline(
        ambient,
        &lighting_desc,
        &ambient_data.0,
        ambient_layout,
        Box::new(
            move |world: &mut World,
                  _resource_manager: &mut ResourceManager,
                  descriptor_manager: &mut DescriptorManager,
                  handle: PipelineHandle| {
                let ambient = world.get_resource::<AmbientLight>().unwrap();

                let gbuffer_albedo = descriptor_manager.request_bind(
                    handle,
                    0,
                    0,
                    BindingData::RenderGraphImage { id: albedo_id },
                );
                let gbuffer_normal = descriptor_manager.request_bind(
                    handle,
                    0,
                    1,
                    BindingData::RenderGraphImage { id: normal_id },
                );
                let gbuffer_position = descriptor_manager.request_bind(
                    handle,
                    0,
                    2,
                    BindingData::RenderGraphImage { id: position_id },
                );
                let ambient_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&AmbientLightUBO::from(ambient)).to_vec(),
                    },
                );

                vec![DrawJob {
                    mesh: DrawStyle::VertexCount(3),
                    descriptor_sets: vec![
                        gbuffer_albedo,
                        gbuffer_normal,
                        gbuffer_position,
                        ambient_handle,
                    ],
                }]
            },
        ),
    );

    pipeline_manager.add_pipeline(
        directional,
        &lighting_desc,
        &directional_data.0,
        directional_layout,
        Box::new(
            move |world: &mut World,
                  _resource_manager: &mut ResourceManager,
                  descriptor_manager: &mut DescriptorManager,
                  handle: PipelineHandle| {
                let directional = world.get_resource::<DirectionalLight>().unwrap();

                let gbuffer_albedo = descriptor_manager.request_bind(
                    handle,
                    0,
                    0,
                    BindingData::RenderGraphImage { id: albedo_id },
                );
                let gbuffer_normal = descriptor_manager.request_bind(
                    handle,
                    0,
                    1,
                    BindingData::RenderGraphImage { id: normal_id },
                );
                let gbuffer_position = descriptor_manager.request_bind(
                    handle,
                    0,
                    2,
                    BindingData::RenderGraphImage { id: position_id },
                );
                let directional_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&DirectionalLightUBO::from(directional)).to_vec(),
                    },
                );

                vec![DrawJob {
                    mesh: DrawStyle::VertexCount(3),
                    descriptor_sets: vec![
                        gbuffer_albedo,
                        gbuffer_normal,
                        gbuffer_position,
                        directional_handle,
                    ],
                }]
            },
        ),
    );

    pipeline_manager.add_pipeline(
        point,
        &lighting_desc,
        &point_data.0,
        point_layout,
        Box::new(
            move |world: &mut World,
                  _resource_manager: &mut ResourceManager,
                  descriptor_manager: &mut DescriptorManager,
                  handle: PipelineHandle| {
                let mut jobs = Vec::new();

                let gbuffer_albedo = descriptor_manager.request_bind(
                    handle,
                    0,
                    0,
                    BindingData::RenderGraphImage { id: albedo_id },
                );
                let gbuffer_normal = descriptor_manager.request_bind(
                    handle,
                    0,
                    1,
                    BindingData::RenderGraphImage { id: normal_id },
                );
                let gbuffer_position = descriptor_manager.request_bind(
                    handle,
                    0,
                    2,
                    BindingData::RenderGraphImage { id: position_id },
                );

                for entity in world.query::<(Req<PointLight>, Req<Transform>)>() {
                    let (_entityid, (point_light, transform)) = entity;

                    let point_handle = descriptor_manager.request_bind(
                        handle,
                        1,
                        0,
                        BindingData::Uniform {
                            data: bytes_of(&PointLightUBO::from((point_light, transform))).to_vec(),
                        },
                    );

                    jobs.push(DrawJob {
                        mesh: DrawStyle::VertexCount(3),
                        descriptor_sets: vec![
                            gbuffer_albedo,
                            gbuffer_normal,
                            gbuffer_position,
                            point_handle,
                        ],
                    });
                }

                jobs
            },
        ),
    );

    pipeline_manager.add_pipeline(
        skybox,
        &skybox_desc,
        &skybox_data.0,
        skybox_layout,
        Box::new(
            |world: &mut World,
             _resource_manager: &mut ResourceManager,
             descriptor_manager: &mut DescriptorManager,
             handle: PipelineHandle| {
                let cubemap = *world.get_resource::<Skybox>().unwrap();
                let camera = world.get_resource::<Camera>().unwrap();

                let camera_handle = descriptor_manager.request_bind(
                    handle,
                    0,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&CameraInverseUBO::from(camera)).to_vec(),
                    },
                );
                let cubemap_handle = descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::Texture {
                        texture: cubemap.material.albedo,
                    },
                );

                vec![DrawJob {
                    mesh: DrawStyle::VertexCount(3),
                    descriptor_sets: vec![camera_handle, cubemap_handle],
                }]
            },
        ),
    );

    let graph = graph
        .add_pipeline("static_geometry")
        .pipeline(static_geometry)
        .writes(&mut albedo)
        .writes(&mut normal)
        .writes(&mut position)
        .writes_depth(&mut depth)
        .build();
    let graph = graph
        .add_pipeline("animated_geometry")
        .pipeline(animated_geometry)
        .writes(&mut albedo)
        .writes(&mut normal)
        .writes(&mut position)
        .writes_depth(&mut depth)
        .build();
    let graph = graph
        .add_pipeline("clipped_geometry")
        .pipeline(clipped_geometry)
        .writes(&mut albedo)
        .writes(&mut normal)
        .writes(&mut position)
        .writes_depth(&mut depth)
        .build();
    let graph = graph
        .add_pipeline("terrain")
        .pipeline(terrain)
        .writes(&mut albedo)
        .writes(&mut normal)
        .writes(&mut position)
        .writes_depth(&mut depth)
        .build();
    let graph = graph
        .add_pipeline("ambient_light")
        .pipeline(ambient)
        .reads(&albedo)
        .reads(&normal)
        .reads(&position)
        .reads_depth(&depth)
        .writes(&mut final_color)
        .build();
    let graph = graph
        .add_pipeline("directional_light")
        .pipeline(directional)
        .reads(&albedo)
        .reads(&normal)
        .reads(&position)
        .reads_depth(&depth)
        .writes(&mut final_color)
        .build();
    let graph = graph
        .add_pipeline("point_light")
        .pipeline(point)
        .reads(&albedo)
        .reads(&normal)
        .reads(&position)
        .reads_depth(&depth)
        .writes(&mut final_color)
        .build();
    let graph = graph
        .add_pipeline("skybox")
        .pipeline(skybox)
        .writes(&mut final_color)
        .reads_depth(&depth)
        .build();

    (
        graph,
        pipeline_manager,
        descriptor_managers.try_into().unwrap(),
        final_color.id,
    )
}

/*--------------PIPELINE CREATION HELPERS-------------
-----------------------------------------------------*/
#[derive(Clone)]
pub struct VertexInputState {
    pub vertex_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
}
#[derive(Clone)]
pub struct InputAssemblyState {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool,
}
pub struct SpecalizationInfo {
    pub map_entries: Vec<vk::SpecializationMapEntry>,
    pub data: Vec<u8>,
}
#[derive(Clone)]
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

    device: Arc<ash::Device>,
}

impl Drop for ShaderStage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.shader, None);
        }
    }
}

#[derive(Clone)]
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

#[derive(Clone)]
pub struct MultisampleState {
    pub rasterization_samples: vk::SampleCountFlags,
    pub sample_shading_enable: bool,
    pub min_sample_shading: f32,
    pub sample_mask: Option<u32>,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

#[derive(Clone)]
pub struct ColorBlendState {
    pub logic_op: Option<vk::LogicOp>,
    pub attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    pub blend_constants: [f32; 4],
}
#[derive(Clone)]
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

#[derive(Clone)]
pub struct TesselationState {
    pub patch_control_points: u32,
}

#[derive(Clone)]
pub struct GraphicsPipelineDesc {
    pub vertex_input_state: VertexInputState,
    pub input_assembly: InputAssemblyState,
    pub viewport_state: Option<Viewport>,
    pub raster_state: RasterState,
    pub multisample_state: MultisampleState,
    pub depth_stencil_state: DepthStencilState,
    pub color_blend_state: ColorBlendState,
    pub dynamic_state: Vec<vk::DynamicState>,
    pub tesselation_state: Option<TesselationState>,
    pub color_attachment_formats: Vec<vk::Format>,
    pub depth_attachment_format: Option<vk::Format>,
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    desc: &GraphicsPipelineDesc,
    shaders: &[ShaderStage],
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, (vk::Pipeline, vk::Result)> {
    let mut stages = Vec::new();
    for shader in shaders {
        let specalization_info =
            shader
                .specalization_info
                .as_ref()
                .map(|info| vk::SpecializationInfo {
                    map_entry_count: info.map_entries.len() as u32,
                    p_map_entries: info.map_entries.as_ptr(),
                    p_data: info.data.as_ptr() as *const c_void,
                    data_size: info.data.len(),
                    _marker: marker::PhantomData,
                });

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

    let tesselation_state =
        desc.tesselation_state
            .as_ref()
            .map(|x| vk::PipelineTessellationStateCreateInfo {
                patch_control_points: x.patch_control_points,

                ..Default::default()
            });
    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo {
        color_attachment_count: desc.color_attachment_formats.len() as u32,
        p_color_attachment_formats: desc.color_attachment_formats.as_ptr(),
        depth_attachment_format: desc
            .depth_attachment_format
            .unwrap_or(vk::Format::UNDEFINED),
        ..Default::default()
    };

    let tessellation_state_info =
        tesselation_state.map(|x| vk::PipelineTessellationStateCreateInfo {
            patch_control_points: x.patch_control_points,
            ..Default::default()
        });
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
        p_tessellation_state: tessellation_state_info
            .as_ref()
            .map_or(ptr::null(), |x| x as *const _),

        layout: pipeline_layout,

        // if deriving from a graphics pipeline, the index
        base_pipeline_index: 0,
        // and the handle to that pipeline
        base_pipeline_handle: vk::Pipeline::null(),
        p_next: &mut pipeline_rendering_info as *mut _ as *const c_void,
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

pub fn create_compute_pipeline(
    device: &ash::Device,
    shader: ShaderStage,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, (vk::Pipeline, vk::Result)> {
    let specalization_info =
        shader
            .specalization_info
            .as_ref()
            .map(|info| vk::SpecializationInfo {
                map_entry_count: info.map_entries.len() as u32,
                p_map_entries: info.map_entries.as_ptr(),
                p_data: info.data.as_ptr() as *const c_void,
                data_size: info.data.len(),
                _marker: marker::PhantomData,
            });

    let stage = vk::PipelineShaderStageCreateInfo {
        p_specialization_info: specalization_info.as_ref().map_or(ptr::null(), |info| info),
        stage: shader.kind,
        module: shader.shader,
        p_name: shader.entry_point.as_ptr(),
        ..Default::default()
    };

    let create_info = vk::ComputePipelineCreateInfo {
        layout: pipeline_layout,
        // if deriving from a graphics pipeline, the index
        base_pipeline_index: 0,
        // and the handle to that pipeline
        base_pipeline_handle: vk::Pipeline::null(),
        stage,
        ..Default::default()
    };

    unsafe {
        Ok(
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) {
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

pub fn get_pipeline_data(
    device: Arc<ash::Device>,
    vertex_path: &path::Path,
    fragment_path: &path::Path,
) -> (
    Vec<ShaderStage>,
    BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
    BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
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
        device: device.clone(),
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
        device: device.clone(),
    };

    let vertex_descriptor_sets = vertex_reflection.get_descriptor_sets().unwrap();
    let fragment_descriptor_sets = fragment_reflection.get_descriptor_sets().unwrap();

    (
        vec![vertex_stage, fragment_stage],
        vertex_descriptor_sets,
        fragment_descriptor_sets,
    )
}

pub fn get_compute_data(
    device: Arc<ash::Device>,
    compute_path: &path::Path,
) -> (
    ShaderStage,
    BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
) {
    let compute_reflection = load_path_data(compute_path);
    assert_eq!(
        compute_reflection.0.entry_points.len(),
        1,
        "only single entry point supported"
    );
    let code = compute_reflection.0.assemble();
    let compute_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: code.as_ptr(),
        code_size: code.len() * 4,
        ..Default::default()
    };
    let compute_module = unsafe {
        device
            .create_shader_module(&compute_module_create_info, None)
            .unwrap()
    };
    let compute_stage = ShaderStage {
        shader: compute_module,
        kind: vk::ShaderStageFlags::COMPUTE,
        entry_point: get_entry_name(&compute_reflection),
        specalization_info: None,
        device: device.clone(),
    };
    let compute_descriptor_sets = compute_reflection.get_descriptor_sets().unwrap();
    (compute_stage, compute_descriptor_sets)
}

/*-------------------SHADER REFLECTION----------------
-----------------------------------------------------*/
const ASM_ENTRY_POINT_EXECUTION_MODEL_IDX: usize = 0;
const ASM_ENTRY_POINT_NAME_IDX: usize = 2;
pub fn create_pipeline_layout_from_vert_frag(
    device: Arc<ash::Device>,
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
        device: device.clone(),
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
        device: device.clone(),
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
            let max_vertex = vertex_bindings.and_then(|x| x.keys().max());
            let max_fragment = fragment_bindings.and_then(|x| x.keys().max());
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
    let full_path: PathBuf = out_dir.join(path);
    let code = fs::read(full_path).expect("failed to read file");
    rr::Reflection::new_from_spirv(&code).unwrap()
}
fn get_entry_name(reflection: &Reflection) -> ffi::CString {
    let entry_point_name =
        reflection.0.entry_points[0].operands[ASM_ENTRY_POINT_NAME_IDX].unwrap_literal_string();

    ffi::CString::new(entry_point_name).unwrap()
}
fn get_shader_kind(reflection: &Reflection) -> vk::ShaderStageFlags {
    let raw_stage = reflection.0.entry_points[0].operands[ASM_ENTRY_POINT_EXECUTION_MODEL_IDX]
        .unwrap_execution_model();

    vk::ShaderStageFlags::from_raw(0b1 << (raw_stage as u32))
}
