use std::collections::BTreeMap;
use std::ffi::{self, c_void};
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs, marker};
use std::{
    path::{self, Path},
    ptr,
};

use ash::vk::{self, Extent2D};
use bytemuck::bytes_of;
use educe::Educe;
use egui::TextBuffer;
use glam::{UVec3, Vec3};
use rspirv_reflect::{self as rr, Reflection};

use crate::core::TerrainMap;
use crate::ecs::{Not, NotM, Opt, OptM, ReqM, World};
use crate::renderers::world::descriptors::{BindingData, DescriptorManager};
use crate::renderers::world::draw::{DrawStyle, PipelineJob};
use crate::renderers::world::radiance::{self, RadianceCascadesConfiguration};
use crate::renderers::world::rendergraph::{ImageDesc, ImageId, RenderGraph};
use crate::resources::{Animated, AnimatedVertex, IsVertex, ResourceManager, Vertex};
use crate::resources::{Material, Mesh};
use crate::ubo::{CameraUBO, DirectionalLightUBO, MaterialUBO, ModelUBO, TerrainUBO};
use crate::vulkan::SharedAllocator;
use crate::{
    components::{Camera, DirectionalLight, Transform},
    ecs::Req,
    renderers::world::draw::DrawJob,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PipelineHandle {
    pub is_compute: bool,
    pub name: &'static str,
    arr_index: usize,
}

#[derive(Educe)]
#[educe(Debug)]
pub struct PipelineManager {
    pipelines: Vec<Option<PipelineBundle>>,
    pipeline_names: Vec<&'static str>,
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
    pub fn allocate_handle(&mut self, name: &'static str) -> PipelineHandle {
        self.pipelines.push(None);
        self.pipeline_names.push(name);
        PipelineHandle {
            is_compute: false,
            name,
            arr_index: self.pipelines.len() - 1,
        }
    }
    pub fn allocate_compute_handle(&mut self, name: &'static str) -> PipelineHandle {
        self.pipelines.push(None);
        self.pipeline_names.push(name);
        PipelineHandle {
            is_compute: true,
            name,
            arr_index: self.pipelines.len() - 1,
        }
    }
    pub fn add_compute_pipeline(
        &mut self,
        handle: PipelineHandle,
        shader: &ShaderStage,
        layout: vk::PipelineLayout,
        pipeline_fn: PipelineFn,
    ) {
        assert!(handle.is_compute);

        let pipeline = create_compute_pipeline(&self.device, shader, layout).unwrap();

        let bundle = PipelineBundle {
            name: handle.name,
            pipeline,
            is_compute: true,
            write_data_and_build_draw_jobs: pipeline_fn,
            device: self.device.clone(),
        };
        self.pipelines[handle.arr_index] = Some(bundle);
    }
    pub fn add_pipeline(
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
            is_compute: false,
            pipeline,
            write_data_and_build_draw_jobs: pipeline_fn,
            device: self.device.clone(),
        };
        self.pipelines[handle.arr_index] = Some(bundle);
    }
    pub fn get(&self, pipeline_handle: &PipelineHandle) -> &PipelineBundle {
        self.pipelines[pipeline_handle.arr_index]
            .as_ref()
            .expect("all pipeline handles should have valid pipelines")
    }
    pub fn all_pipelines(&self) -> impl Iterator<Item = (PipelineHandle, &PipelineBundle)> {
        self.pipelines
            .iter()
            .enumerate()
            .filter_map(|(index, bundle)| {
                let bundle = bundle.as_ref()?;
                Some((
                    PipelineHandle {
                        is_compute: bundle.is_compute,
                        arr_index: index,
                        name: bundle.name,
                    },
                    bundle,
                ))
            })
    }
}

pub type PipelineFn = Box<
    dyn Fn(
        &mut World,
        &mut ResourceManager,
        &mut DescriptorManager,
        PipelineHandle,
        vk::Extent2D,
    ) -> PipelineJob,
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
) -> (RenderGraph, PipelineManager, DescriptorManager, ImageId) {
    // let depth_format = vk::Format::D32_SFLOAT;
    // let color_formats = [
    //     vk::Format::A2B10G10R10_UNORM_PACK32,
    //     vk::Format::R16G16B16A16_SFLOAT,
    //     vk::Format::R32G32B32A32_SFLOAT,
    // ];
    // let geometry_desc = geometry_pipeline_desc(depth_format, &color_formats);
    //
    // let animated_geometry_desc = GraphicsPipelineDesc {
    //     vertex_input_state: VertexInputState {
    //         vertex_attribute_descriptions: AnimatedVertex::get_vertex_attributes(),
    //         vertex_binding_descriptions: vec![vk::VertexInputBindingDescription {
    //             binding: 0,
    //             stride: std::mem::size_of::<AnimatedVertex>() as u32,
    //             input_rate: vk::VertexInputRate::VERTEX,
    //         }],
    //     },
    //     ..geometry_desc.clone()
    // };
    // let clipped_geometry_desc = GraphicsPipelineDesc {
    //     raster_state: RasterState {
    //         cull_mode: vk::CullModeFlags::NONE,
    //         front_face: vk::FrontFace::COUNTER_CLOCKWISE,
    //         line_width: 1.0,
    //         depth_clamp_enable: false,
    //         rasterizer_discard_enable: false,
    //         polygon_mode: vk::PolygonMode::FILL,
    //         depth_bias_enable: false,
    //         depth_bias_constant_factor: 0.0,
    //         depth_bias_clamp: 0.0,
    //         depth_bias_slope_factor: 0.0,
    //     },
    //     ..geometry_desc.clone()
    // };
    //
    // let lighting_desc = additive_light_pass_desc(&geometry_desc, color_formats.to_vec());
    //
    // let tonemap_desc = fullscreen_opaque_pass_desc(&lighting_desc, vec![swapchain_image_format]);
    //
    // let terrain_desc = GraphicsPipelineDesc {
    //     vertex_input_state: VertexInputState {
    //         vertex_binding_descriptions: vec![],
    //         vertex_attribute_descriptions: vec![],
    //     },
    //     ..geometry_desc.clone()
    // };
    //
    // // --- reflect shaders ---
    // let static_geometry_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/static_geometry.vert.spv"),
    //     Path::new("shaders/geometry.frag.spv"),
    // );
    // let animated_geometry_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/animated_geometry.vert.spv"),
    //     Path::new("shaders/geometry.frag.spv"),
    // );
    // let clipped_geometry_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/static_geometry.vert.spv"),
    //     Path::new("shaders/clipped_geometry.frag.spv"),
    // );
    // let terrain_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/terrain.vert.spv"),
    //     Path::new("shaders/terrain.frag.spv"),
    // );
    // let tonemap_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/tonemap.vert.spv"),
    //     Path::new("shaders/tonemap.frag.spv"),
    // );
    // let directional_data = get_pipeline_data(
    //     device.clone(),
    //     Path::new("shaders/directional.vert.spv"),
    //     Path::new("shaders/directional.frag.spv"),
    // );
    //
    let mut pipeline_manager = PipelineManager::new(device.clone());
    //
    // // --- allocate handles ---
    // let static_geometry = pipeline_manager.allocate_handle("static_geometry");
    // let animated_geometry = pipeline_manager.allocate_handle("animated_geometry");
    // let clipped_geometry = pipeline_manager.allocate_handle("clipped_geometry");
    // let terrain = pipeline_manager.allocate_handle("terrain");
    // let tonemap = pipeline_manager.allocate_handle("tonemap");
    // let directional = pipeline_manager.allocate_handle("directional");
    //
    // // --- build descriptor managers, get layouts ---
    let mut descriptor_manager: DescriptorManager =
        DescriptorManager::new(device.clone(), allocator.clone());
    //
    // let static_geometry_layout = descriptor_manager.add_pipeline(
    //     static_geometry,
    //     static_geometry_data.vertex_sets,
    //     static_geometry_data.fragment_sets,
    // );
    // let animated_geometry_layout = descriptor_manager.add_pipeline(
    //     animated_geometry,
    //     animated_geometry_data.vertex_sets,
    //     animated_geometry_data.fragment_sets,
    // );
    // let clipped_geometry_layout = descriptor_manager.add_pipeline(
    //     clipped_geometry,
    //     clipped_geometry_data.vertex_sets,
    //     clipped_geometry_data.fragment_sets,
    // );
    // let terrain_layout = descriptor_manager.add_pipeline(
    //     terrain,
    //     terrain_data.vertex_sets,
    //     terrain_data.fragment_sets,
    // );
    // let tonemap_layout = descriptor_manager.add_pipeline(
    //     tonemap,
    //     tonemap_data.vertex_sets,
    //     tonemap_data.fragment_sets,
    // );
    // let directional_layout = descriptor_manager.add_pipeline(
    //     directional,
    //     directional_data.vertex_sets,
    //     directional_data.fragment_sets,
    // );
    //
    let mut graph = RenderGraph::new();
    //
    let mut final_color = graph.add_image(ImageDesc::Imported {
        name: "final_color",
        format: swapchain_image_format,
    });
    // let mut albedo = graph.add_image(ImageDesc::Managed {
    //     name: "albedo",
    //     format: vk::Format::A2B10G10R10_UNORM_PACK32,
    // });
    // let mut normal = graph.add_image(ImageDesc::Managed {
    //     name: "normal",
    //     format: vk::Format::R16G16B16A16_SFLOAT,
    // });
    // let mut position = graph.add_image(ImageDesc::Managed {
    //     name: "position",
    //     format: vk::Format::R32G32B32A32_SFLOAT,
    // });
    // let mut depth = graph.add_image(ImageDesc::Managed {
    //     name: "depth",
    //     format: vk::Format::D32_SFLOAT,
    // });
    // let mut hdr_color = graph.add_image(ImageDesc::Managed {
    //     name: "hdr_color",
    //     format: vk::Format::R32G32B32A32_SFLOAT,
    // });
    //
    // let albedo_id = albedo.id;
    // let normal_id = normal.id;
    // let position_id = position.id;
    // let hdr_color_id = hdr_color.id;
    //
    // pipeline_manager.add_pipeline(
    //     static_geometry,
    //     &geometry_desc,
    //     &static_geometry_data.stages,
    //     static_geometry_layout,
    //     Box::new(static_geometry_setup),
    // );
    //
    // pipeline_manager.add_pipeline(
    //     animated_geometry,
    //     &animated_geometry_desc,
    //     &animated_geometry_data.stages,
    //     animated_geometry_layout,
    //     Box::new(animated_geometry_setup),
    // );
    //
    // pipeline_manager.add_pipeline(
    //     clipped_geometry,
    //     &clipped_geometry_desc,
    //     &clipped_geometry_data.stages,
    //     clipped_geometry_layout,
    //     Box::new(clipped_geometry_setup),
    // );
    //
    // pipeline_manager.add_pipeline(
    //     terrain,
    //     &terrain_desc,
    //     &terrain_data.stages,
    //     terrain_layout,
    //     Box::new(terrainmap_setup),
    // );
    //
    // pipeline_manager.add_pipeline(
    //     directional,
    //     &lighting_desc,
    //     &directional_data.stages,
    //     directional_layout,
    //     make_directional_light_setup(albedo_id, normal_id, position_id),
    // );
    //
    // let graph = graph
    //     .add_pipeline("static_geometry")
    //     .pipeline(static_geometry)
    //     .writes(&mut albedo)
    //     .writes(&mut normal)
    //     .writes(&mut position)
    //     .writes_depth(&mut depth)
    //     .build();
    // let graph = graph
    //     .add_pipeline("animated_geometry")
    //     .pipeline(animated_geometry)
    //     .writes(&mut albedo)
    //     .writes(&mut normal)
    //     .writes(&mut position)
    //     .writes_depth(&mut depth)
    //     .build();
    // let graph = graph
    //     .add_pipeline("clipped_geometry")
    //     .pipeline(clipped_geometry)
    //     .writes(&mut albedo)
    //     .writes(&mut normal)
    //     .writes(&mut position)
    //     .writes_depth(&mut depth)
    //     .build();
    //
    // let graph = graph
    //     .add_pipeline("terrain")
    //     .pipeline(terrain)
    //     .writes(&mut albedo)
    //     .writes(&mut normal)
    //     .writes(&mut position)
    //     .writes_depth(&mut depth)
    //     .build();
    //
    // pipeline_manager.add_pipeline(
    //     tonemap,
    //     &tonemap_desc,
    //     &tonemap_data.stages,
    //     tonemap_layout,
    //     make_tonemap_setup(hdr_color_id),
    // );
    //
    // let config = RadianceCascadesConfiguration {
    //     volume_center: Vec3::ZERO,
    //     top_level_probe_count: UVec3::new(8, 6, 8),
    //     top_level_probe_gap: 2.0,
    //     cascade_count: 3,
    //     bottom_level_rays_per_probe: 16,
    //     base_interval_length_ratio: 1.7,
    // };
    //
    // let graph = radiance::setup(
    //     config,
    //     device,
    //     &mut pipeline_manager,
    //     &mut descriptor_manager,
    //     graph,
    //     allocator,
    //     &mut albedo,
    //     &mut position,
    //     &mut normal,
    //     &mut hdr_color,
    // );
    //
    // let graph = graph
    //     .add_pipeline("tonemap")
    //     .pipeline(tonemap)
    //     .reads(&hdr_color)
    //     .writes(&mut final_color)
    //     .build();

    (graph, pipeline_manager, descriptor_manager, final_color.id)
}

pub(crate) fn opaque_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        ..Default::default()
    }
}

pub(crate) fn additive_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState {
        color_blend_op: vk::BlendOp::ADD,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ONE,
        alpha_blend_op: vk::BlendOp::MAX,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        blend_enable: vk::TRUE,
    }
}

/// Base pipeline desc for opaque, depth-tested mesh geometry.
/// static/animated/clipped/terrain all derive from this with small overrides.
pub(crate) fn geometry_pipeline_desc(
    depth_format: vk::Format,
    color_attachment_formats: &[vk::Format],
) -> GraphicsPipelineDesc {
    GraphicsPipelineDesc {
        tesselation_state: None,
        viewport_state: None,
        dynamic_state: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        multisample_state: MultisampleState {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: false,
            min_sample_shading: 0.0,
            sample_mask: None,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
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
            attachments: vec![opaque_attachment(); 3],
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
    }
}

/// Base pipeline desc for a fullscreen-triangle pass that additively
/// blends light contributions into `attachment_count` outputs, sampling
/// (not writing) depth. lighting/ambient/skybox derive from this.
pub(crate) fn additive_light_pass_desc(
    base: &GraphicsPipelineDesc,
    color_attachment_formats: Vec<vk::Format>,
) -> GraphicsPipelineDesc {
    let attachment_count = color_attachment_formats.len();
    GraphicsPipelineDesc {
        color_attachment_formats,
        color_blend_state: ColorBlendState {
            logic_op: None,
            blend_constants: [0.0; 4],
            attachments: vec![additive_attachment(); attachment_count],
        },
        vertex_input_state: VertexInputState {
            vertex_attribute_descriptions: vec![],
            vertex_binding_descriptions: vec![],
        },
        depth_stencil_state: DepthStencilState {
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::ALWAYS,
            ..base.depth_stencil_state.clone()
        },
        ..base.clone()
    }
}

/// Base pipeline desc for a fullscreen-triangle pass with no blending
/// and no depth attachment at all. tonemap derives from this.
fn fullscreen_opaque_pass_desc(
    base: &GraphicsPipelineDesc,
    color_attachment_formats: Vec<vk::Format>,
) -> GraphicsPipelineDesc {
    GraphicsPipelineDesc {
        color_attachment_formats,
        color_blend_state: ColorBlendState {
            logic_op: None,
            blend_constants: [0.0; 4],
            attachments: vec![opaque_attachment()],
        },
        depth_attachment_format: None,
        depth_stencil_state: DepthStencilState {
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::ALWAYS,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
        },
        vertex_input_state: VertexInputState {
            vertex_attribute_descriptions: vec![],
            vertex_binding_descriptions: vec![],
        },
        ..base.clone()
    }
}

fn make_tonemap_setup(hdr_color_id: ImageId) -> PipelineFn {
    Box::new(
        move |_world: &mut World,
              _resource_manager: &mut ResourceManager,
              descriptor_manager: &mut DescriptorManager,
              handle: PipelineHandle,
              _extent: Extent2D| {
            let hdr_input = descriptor_manager.request_bind(
                handle,
                0,
                0,
                BindingData::RenderGraphImage { id: hdr_color_id },
            );
            PipelineJob::Graphics(vec![DrawJob {
                mesh: DrawStyle::VertexCount(3),
                descriptor_sets: vec![hdr_input],
            }])
        },
    )
}
fn make_directional_light_setup(
    albedo_id: ImageId,
    position_id: ImageId,
    normal_id: ImageId,
) -> PipelineFn {
    Box::new(
        move |world: &mut World,
              _resource_manager: &mut ResourceManager,
              descriptor_manager: &mut DescriptorManager,
              handle: PipelineHandle,
              _extent: Extent2D| {
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

            let jobs = vec![DrawJob {
                mesh: DrawStyle::VertexCount(3),
                descriptor_sets: vec![
                    gbuffer_albedo,
                    gbuffer_normal,
                    gbuffer_position,
                    directional_handle,
                ],
            }];
            PipelineJob::Graphics(jobs)
        },
    )
}
fn terrainmap_setup(
    world: &mut World,
    _resource_manager: &mut ResourceManager,
    descriptor_manager: &mut DescriptorManager,
    handle: PipelineHandle,
    _extent: Extent2D,
) -> PipelineJob {
    let camera = world.get_resource::<Camera>().unwrap();

    if let Some(heightmap) = world.get_resource::<TerrainMap>() {
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

        let jobs = vec![DrawJob {
            mesh: DrawStyle::VertexCount(
                (heightmap.resolution - 1) * (heightmap.resolution - 1) * 6,
            ),
            descriptor_sets: vec![height_map_handle, camera_handle, height_texture_handle],
        }];
        PipelineJob::Graphics(jobs)
    } else {
        PipelineJob::Graphics(vec![])
    }
}
fn clipped_geometry_setup(
    world: &mut World,
    _resource_manager: &mut ResourceManager,
    descriptor_manager: &mut DescriptorManager,
    handle: PipelineHandle,
    _extent: Extent2D,
) -> PipelineJob {
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

    for entity in world.query_mut::<(ReqM<Mesh>, ReqM<Transform>, ReqM<Material>, NotM<Animated>)>()
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
            descriptor_sets: vec![model_handle, camera_handle, image_handle, material_handle],
        });
    }

    PipelineJob::Graphics(jobs)
}

fn animated_geometry_setup(
    world: &mut World,
    resource_manager: &mut ResourceManager,
    descriptor_manager: &mut DescriptorManager,
    handle: PipelineHandle,
    _extent: Extent2D,
) -> PipelineJob {
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

    for entity in world.query_mut::<(ReqM<Mesh>, ReqM<Transform>, OptM<Material>, OptM<Animated>)>()
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

    PipelineJob::Graphics(jobs)
}

fn static_geometry_setup(
    world: &mut World,
    _resource_manager: &mut ResourceManager,
    descriptor_manager: &mut DescriptorManager,
    handle: PipelineHandle,
    _extent: Extent2D,
) -> PipelineJob {
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

    for entity in world.query::<(Req<Mesh>, Req<Transform>, Opt<Material>, Not<Animated>)>() {
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

    PipelineJob::Graphics(jobs)
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
    shader: &ShaderStage,
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

pub struct GetPipelineDataResult {
    pub stages: Vec<ShaderStage>,
    pub vertex_sets: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
    pub fragment_sets: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
}

pub fn get_pipeline_data(
    device: Arc<ash::Device>,
    vertex_path: &path::Path,
    fragment_path: &path::Path,
) -> GetPipelineDataResult {
    let (vertex_code, vertex_reflection) = load_path_data(vertex_path);
    let (fragment_code, fragment_reflection) = load_path_data(fragment_path);

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

    let vertex_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: vertex_code.as_ptr(),
        code_size: vertex_code.len() * 4,
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

    let fragment_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: fragment_code.as_ptr(),
        code_size: fragment_code.len() * 4,
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

    GetPipelineDataResult {
        stages: vec![vertex_stage, fragment_stage],
        vertex_sets: vertex_descriptor_sets,
        fragment_sets: fragment_descriptor_sets,
    }
}

pub fn get_compute_data(
    device: Arc<ash::Device>,
    compute_path: &path::Path,
) -> (
    ShaderStage,
    BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
) {
    let (compute_code, compute_reflection) = load_path_data(compute_path);
    assert_eq!(
        compute_reflection.0.entry_points.len(),
        1,
        "only single entry point supported"
    );
    let compute_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: compute_code.as_ptr(),
        code_size: compute_code.len() * 4,
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
    let (vertex_code, vertex_reflection) = load_path_data(vertex_path);
    let (fragment_code, fragment_reflection) = load_path_data(fragment_path);

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

    let vertex_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: vertex_code.as_ptr(),
        code_size: vertex_code.len() * 4,
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

    let fragment_module_create_info = vk::ShaderModuleCreateInfo {
        p_code: fragment_code.as_ptr(),
        code_size: fragment_code.len() * 4,
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

fn load_path_data(path: &path::Path) -> (Vec<u32>, Reflection) {
    let out_dir = PathBuf::from(env!("OUT_DIR"));
    let out_path = out_dir.join(path);
    let bytes = fs::read(&out_path)
        .unwrap_or_else(|_| panic!("failed to read file: {}", out_path.display()));
    let code: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let reflect_path = path.with_extension("reflect.spv");
    let reflect_bytes = fs::read(out_dir.join(&reflect_path)).expect("failed to read reflect file");
    let reflection = rr::Reflection::new_from_spirv(&reflect_bytes).unwrap();
    (code, reflection)
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
