use std::{path::Path, sync::Arc};

use ash::vk::{self, Extent2D};
use bytemuck::{Pod, Zeroable, bytes_of};
use glam::{UVec3, Vec3, Vec4, Vec4Swizzles};

use crate::{
    components::{Camera, DirectionalLight, PointLight, Transform},
    ecs::{Req, World},
    renderers::world::{
        descriptors::{BindingData, DescriptorManager},
        draw::{ComputeDispatch, DrawJob, DrawStyle, PipelineJob, alloc_buffers},
        pipelines::{
            ColorBlendState, DepthStencilState, GraphicsPipelineDesc, PipelineFn, PipelineHandle,
            PipelineManager, additive_attachment, additive_light_pass_desc, geometry_pipeline_desc,
            get_compute_data, get_pipeline_data, opaque_attachment,
        },
        rendergraph::{ImageDesc, ImageId, ImageVersion, RenderGraph},
    },
    resources::{Mesh, ResourceManager, SsboBinding, SsboHandle},
    ubo::{
        CameraInverseUBO, ComputeRadianceUBO, DirectionalLightUBO, LightDataUBO, MeshInfo, ModelUBO,
    },
    vulkan::SharedAllocator,
};

#[derive(Debug, Clone, Copy)]
pub struct RadianceCascadesConfiguration {
    pub volume_center: Vec3,
    /// probe volume is just this dimension * smallest_object_size
    pub top_level_probe_count: UVec3,
    pub top_level_probe_gap: f32,
    pub cascade_count: u32,
    /// must be a perfect square, if is not a perfect square,
    /// it will be rounded down to the nearest square
    pub bottom_level_rays_per_probe: u32,
    /// the ratio to mulitply the gap between probes by to get the interal length
    pub base_interval_length_ratio: f32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct RadianceLevelConfigUBO {
    grid_size: UVec3,
    grid_gap: f32,
    /// the grid origin, varies between each level because how how each probe level is offset from
    /// the previous
    grid_origin: Vec3,
    sqrt_ray_count: u32,
    interval_start: f32,
    interval_end: f32,
    is_top_cascade: u32, // zero for false, anything else for true
    _pad0: u32,
}

#[derive(Debug, Clone)]
pub struct RadianceMeshBuffers {
    pub position_ssbo: SsboHandle,
    pub index_ssbo: SsboHandle,
    pub mesh_infos: Vec<MeshInfo>,
}

pub fn setup(
    config: RadianceCascadesConfiguration,
    device: Arc<ash::Device>,
    pipeline_manager: &mut PipelineManager,
    descriptor_manager: &mut DescriptorManager,
    mut graph: RenderGraph,
    allocator: SharedAllocator,
    albedo: &mut ImageVersion,
    position: &mut ImageVersion,
    normal: &mut ImageVersion,
    hdr_color: &mut ImageVersion,
) -> RenderGraph {
    dbg!(&config);
    let compute_radiance_data = get_compute_data(
        device.clone(),
        Path::new("shaders/compute_cascade.comp.spv"),
    );
    let apply_radiance_data = get_pipeline_data(
        device.clone(),
        Path::new("shaders/ambient.vert.spv"),
        Path::new("shaders/ambient.frag.spv"),
    );

    let apply_radiance = pipeline_manager.allocate_handle("ambient");
    let apply_radiance_desc = additive_light_pass_desc(
        &geometry_pipeline_desc(vk::Format::from_raw(0), &[]),
        vec![vk::Format::R32G32B32A32_SFLOAT; 5],
    );

    let apply_radiance_layout = descriptor_manager.add_pipeline(
        apply_radiance,
        apply_radiance_data.vertex_sets,
        apply_radiance_data.fragment_sets,
    );
    let apply_radiance_desc = GraphicsPipelineDesc {
        depth_attachment_format: None,
        depth_stencil_state: DepthStencilState {
            depth_test_enable: false,
            ..apply_radiance_desc.depth_stencil_state
        },
        color_blend_state: ColorBlendState {
            attachments: std::iter::once(additive_attachment())
                .chain(std::iter::repeat_n(opaque_attachment(), 4))
                .collect(),
            ..apply_radiance_desc.color_blend_state
        },
        ..apply_radiance_desc
    };
    let mut tmp1 = graph.add_image(ImageDesc::Managed {
        name: "tmp1",
        format: vk::Format::R32G32B32A32_SFLOAT,
    });
    let mut tmp2 = graph.add_image(ImageDesc::Managed {
        name: "tmp2",
        format: vk::Format::R32G32B32A32_SFLOAT,
    });
    let mut tmp3 = graph.add_image(ImageDesc::Managed {
        name: "tmp3",
        format: vk::Format::R32G32B32A32_SFLOAT,
    });
    let mut tmp4 = graph.add_image(ImageDesc::Managed {
        name: "tmp4",
        format: vk::Format::R32G32B32A32_SFLOAT,
    });

    let mut configs = Vec::new();
    for cascade_level in 0..config.cascade_count {
        let config = RadianceLevelConfigUBO::new(config, cascade_level);
        configs.push(config);
    }

    // == PASS 1: Create all cascade images ==
    let mut cascade_images: Vec<ImageVersion> = Vec::with_capacity(config.cascade_count as usize);
    let mut dbg1_images: Vec<ImageVersion> = Vec::with_capacity(config.cascade_count as usize);
    let mut dbg2_images: Vec<ImageVersion> = Vec::with_capacity(config.cascade_count as usize);

    // from 0 (coarse but many to fine but few)
    for cascade_level in 0..config.cascade_count {
        let grid_size = configs[cascade_level as usize].grid_size;
        let sqrt_ray_count = configs[cascade_level as usize].sqrt_ray_count;

        // fold z into a 2D grid of z-blocks
        // z slices per row
        let z_cols = {
            let s = grid_size.z.isqrt();
            if s * s >= grid_size.z { s } else { s + 1 }
        };
        let z_rows = grid_size.z.div_ceil(z_cols);

        // total xy probes
        let xy = grid_size.x * grid_size.y;

        let xy_cols = {
            let s = xy.isqrt();
            if s * s >= xy { s } else { s + 1 }
        };
        let xy_rows = xy.div_ceil(xy_cols);

        let cascade_image = graph.add_image(ImageDesc::Custom {
            name: "cascade_image",
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent: vk::Extent3D {
                width: z_cols * xy_cols * sqrt_ray_count,
                height: z_rows * xy_rows * sqrt_ray_count,
                depth: 1,
            },
        });
        let dbg1 = graph.add_image(ImageDesc::Custom {
            name: "dbg1",
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent: vk::Extent3D {
                width: z_cols * xy_cols * sqrt_ray_count,
                height: z_rows * xy_rows * sqrt_ray_count,
                depth: 1,
            },
        });
        let dbg2 = graph.add_image(ImageDesc::Custom {
            name: "dbg2",
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent: vk::Extent3D {
                width: z_cols * xy_cols * sqrt_ray_count,
                height: z_rows * xy_rows * sqrt_ray_count,
                depth: 1,
            },
        });

        cascade_images.push(cascade_image);
        dbg1_images.push(dbg1);
        dbg2_images.push(dbg2);
    }

    let mut final_image_id: ImageId = cascade_images[0].id;

    // === PASS 2: Wire up pipelines ===
    // from fine to coarse because that is the order that they run and the image dependancies must
    // be delcared in the order that they run
    for cascade_level in (0..config.cascade_count).rev() {
        // Each cascade reads from the next coarser level (level + 1), except the top
        let (lower, upper) = cascade_images.split_at_mut(cascade_level as usize + 1);

        let cascade_image = &mut lower[cascade_level as usize];

        let above_image = upper.first_mut(); // None if top cascade, Some(&mut next) otherwise

        let radiance_comp = pipeline_manager.allocate_compute_handle("radiance");
        let radiance_layout =
            descriptor_manager.add_compute(radiance_comp, compute_radiance_data.1.clone());

        let dbg1_image = &mut dbg1_images[cascade_level as usize];
        let dbg2_image = &mut dbg2_images[cascade_level as usize];
        let radiance_level_config = RadianceLevelConfigUBO::new(config, cascade_level);

        pipeline_manager.add_compute_pipeline(
            radiance_comp,
            &compute_radiance_data.0,
            radiance_layout,
            make_radiance_setup(
                allocator.clone(),
                device.clone(),
                radiance_level_config,
                above_image.as_ref().map(|x| x.id),
                cascade_image.id,
                dbg1_image.id,
                dbg2_image.id,
            ),
        );

        graph = if cascade_level == config.cascade_count - 1 {
            graph
                .add_pipeline(format!("cascade_{}", cascade_level).as_str())
                .pipeline(radiance_comp)
                .writes(dbg1_image)
                .writes(dbg2_image)
                .writes(cascade_image)
                .build()
        } else {
            graph
                .add_pipeline(format!("cascade_{}", cascade_level).as_str())
                .pipeline(radiance_comp)
                .writes(cascade_image)
                .writes(dbg1_image)
                .writes(dbg2_image)
                .reads(above_image.as_deref().unwrap())
                .build()
        };
        if cascade_level == 0 {
            final_image_id = cascade_image.id;
        }
    }

    pipeline_manager.add_pipeline(
        apply_radiance,
        &apply_radiance_desc,
        &apply_radiance_data.stages,
        apply_radiance_layout,
        make_apply_radiance_pipline_setup(
            albedo.id,
            normal.id,
            position.id,
            final_image_id,
            configs[0],
        ),
    );

    graph
        .add_pipeline("ambient")
        .pipeline(apply_radiance)
        .reads(albedo)
        .reads(normal)
        .reads(position)
        .reads(&cascade_images[0])
        .writes(hdr_color)
        .writes(&mut tmp1)
        .writes(&mut tmp2)
        .writes(&mut tmp3)
        .writes(&mut tmp4)
        .build()
}

fn make_radiance_setup(
    allocator: SharedAllocator,
    device: Arc<ash::Device>,
    radiance_config: RadianceLevelConfigUBO,
    above_id: Option<ImageId>,
    cascade_id: ImageId,
    dbg1_id: ImageId,
    dbg2_id: ImageId,
) -> PipelineFn {
    Box::new(
        move |world: &mut World,
              resource_manager: &mut ResourceManager,
              descriptor_manager: &mut DescriptorManager,
              handle: PipelineHandle,
              _extent: Extent2D| {
            // if world.get_resource::<RadianceMeshBuffers>().is_none() {
            if true {
                let mut all_positions: Vec<Vec4> = Vec::new();
                let mut all_indices: Vec<u32> = Vec::new();
                let mut mesh_infos: Vec<MeshInfo> = Vec::new();

                {
                    for (_, (mesh_handle, transform)) in world
                        .query::<(Req<Mesh>, Req<Transform>)>()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                    {
                        if let Some(mesh) = resource_manager.get_mesh(*mesh_handle) {
                            let vertex_offset = all_positions.len() as u32;
                            let index_offset = all_indices.len() as u32;
                            all_positions.extend_from_slice(&mesh.positions);
                            all_indices.extend_from_slice(&mesh.indices);
                            let mut aabb_min = Vec3::splat(f32::MAX);
                            let mut aabb_max = Vec3::splat(f32::MIN);
                            for pos in &mesh.positions {
                                aabb_min = aabb_min.min(pos.xyz());
                                aabb_max = aabb_max.max(pos.xyz());
                            }

                            mesh_infos.push(MeshInfo {
                                vertex_offset,
                                index_offset,
                                index_count: mesh.index_count,
                                _pad: 0,
                                local_to_world: ModelUBO::from(transform).model,
                                world_to_local: ModelUBO::from(transform).model.inverse(),
                                aabb_local_min: aabb_min.extend(1.0),
                                aabb_local_max: aabb_max.extend(1.0),
                            });
                        }
                    }
                }
                let pos_size = (all_positions.len() * size_of::<Vec4>()) as u64;
                let idx_size = (all_indices.len() * size_of::<u32>()) as u64;

                if pos_size > 0 {
                    let (mut pos_buffers, mut pos_allocs) = alloc_buffers(
                        allocator.clone(),
                        1,
                        pos_size,
                        &device,
                        vk::SharingMode::EXCLUSIVE,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                        true,
                        bytemuck::cast_slice(&all_positions),
                        "radiance positions",
                    )
                    .unwrap();
                    let (mut idx_buffers, mut idx_allocs) = alloc_buffers(
                        allocator.clone(),
                        1,
                        idx_size,
                        &device,
                        vk::SharingMode::EXCLUSIVE,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                        true,
                        bytemuck::cast_slice(&all_indices),
                        "radiance indices",
                    )
                    .unwrap();

                    let position_ssbo = resource_manager.ssbo_registry.register_ssbo(SsboBinding {
                        buffer: pos_buffers.remove(0),
                        allocation: pos_allocs.remove(0),
                        offset: 0,
                        size: pos_size,
                    });
                    let index_ssbo = resource_manager.ssbo_registry.register_ssbo(SsboBinding {
                        buffer: idx_buffers.remove(0),
                        allocation: idx_allocs.remove(0),
                        offset: 0,
                        size: idx_size,
                    });
                    let resource = RadianceMeshBuffers {
                        position_ssbo,
                        index_ssbo,
                        mesh_infos,
                    };
                    if let Some(value) = world.get_mut_resource::<RadianceMeshBuffers>() {
                        *value = resource
                    } else {
                        world.add_resource(resource).unwrap();
                    }
                }
            }

            let mut meshes_array = [MeshInfo::default(); 64];
            let mut mesh_infos = None;
            let mut position_ssbo = None;
            let mut index_ssbo = None;
            if let Some(buffers) = world.get_resource::<RadianceMeshBuffers>() {
                let len = buffers.mesh_infos.len().min(64);
                position_ssbo = Some(buffers.position_ssbo);
                index_ssbo = Some(buffers.index_ssbo);
                mesh_infos = Some(buffers.mesh_infos.clone());
                meshes_array[..len].copy_from_slice(&buffers.mesh_infos[..len]);
            }

            // top reads itself, shader should ignore
            let resolved_above_id = above_id.unwrap_or(cascade_id);

            let config_ubo = ComputeRadianceUBO {
                level_config: radiance_config,
                mesh_count: mesh_infos.unwrap_or_default().len() as u32,
                meshes: meshes_array,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            tracing::trace!(?config_ubo.level_config);
            let directional = *world.get_resource::<DirectionalLight>().unwrap();
            let mut light_positions = [Vec4::ZERO; 32];
            let mut light_colors = [Vec4::ZERO; 32];
            let mut count = 0;
            for (idx, (_entityid, (light, transform))) in world
                .query::<(Req<PointLight>, Req<Transform>)>()
                .enumerate()
            {
                light_positions[idx] = Vec4::new(
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                    light.size,
                );
                // the w component is the radius
                light_colors[idx] = Vec4::new(light.color.x, light.color.y, light.color.z, 1.0);

                count += 1;
            }
            let lighting_ubo = LightDataUBO {
                point_light_count: count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                point_light_positions: light_positions,
                point_light_colors: light_colors,
                sky_light: DirectionalLightUBO::from(&directional),
            };

            let mut bindings = vec![
                // descriptor_manager.request_bind(
                //     handle,
                //     0,
                //     0,
                //     BindingData::Texture { texture: map.map },
                // ),
                // descriptor_manager.request_bind(
                //     handle,
                //     0,
                //     1,
                //     BindingData::Uniform {
                //         data: bytes_of(&TerrainUBO::from(map)).to_vec(),
                //     },
                // ),
                descriptor_manager.request_bind(
                    handle,
                    1,
                    0,
                    BindingData::StorageImage { id: cascade_id },
                ),
                descriptor_manager.request_bind(
                    handle,
                    1,
                    1,
                    BindingData::StorageImage {
                        id: resolved_above_id,
                    },
                ),
                descriptor_manager.request_bind(
                    handle,
                    1,
                    2,
                    BindingData::StorageImage { id: dbg1_id },
                ),
                descriptor_manager.request_bind(
                    handle,
                    1,
                    3,
                    BindingData::StorageImage { id: dbg2_id },
                ),
                descriptor_manager.request_bind(
                    handle,
                    2,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&config_ubo).to_vec(),
                    },
                ),
                descriptor_manager.request_bind(
                    handle,
                    3,
                    0,
                    BindingData::Uniform {
                        data: bytes_of(&lighting_ubo).to_vec(),
                    },
                ),
            ];

            if let Some(p_ssbo) = position_ssbo {
                bindings.push(descriptor_manager.request_bind(
                    handle,
                    2,
                    1,
                    BindingData::Ssbo { buffer: p_ssbo },
                ));
            }
            if let Some(i_ssbo) = index_ssbo {
                bindings.push(descriptor_manager.request_bind(
                    handle,
                    2,
                    2,
                    BindingData::Ssbo { buffer: i_ssbo },
                ));
            }

            // 64x1 chunk for each workgroup
            PipelineJob::Compute(ComputeDispatch {
                x: (radiance_config.grid_size.x
                    * radiance_config.grid_size.y
                    * radiance_config.grid_size.z
                    * radiance_config.sqrt_ray_count
                    * radiance_config.sqrt_ray_count)
                    .div_ceil(64),
                y: 1,
                z: 1,
                bindings,
            })
        },
    )
}
fn make_apply_radiance_pipline_setup(
    albedo_id: ImageId,
    normal_id: ImageId,
    position_id: ImageId,
    final_image_id: ImageId,
    radiance_info: RadianceLevelConfigUBO,
) -> PipelineFn {
    Box::new(
        move |world: &mut World,
              _resource_manager: &mut ResourceManager,
              descriptor_manager: &mut DescriptorManager,
              handle: PipelineHandle,
              _extent: Extent2D| {
            let directional = *world.get_resource::<DirectionalLight>().unwrap();
            let camera = *world.get_resource::<Camera>().unwrap();
            let mut light_positions = [Vec4::ZERO; 32];
            let mut light_colors = [Vec4::ZERO; 32];
            let mut count = 0;
            for (idx, (_entityid, (light, transform))) in world
                .query::<(Req<PointLight>, Req<Transform>)>()
                .enumerate()
            {
                light_positions[idx] = Vec4::new(
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                    light.size,
                );

                light_colors[idx] = Vec4::new(light.color.x, light.color.y, light.color.z, 0.0);

                count += 1;
            }
            let lighting_ubo = LightDataUBO {
                sky_light: DirectionalLightUBO::from(&directional),
                point_light_count: count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                point_light_positions: light_positions,
                point_light_colors: light_colors,
            };
            let camera_ubo = CameraInverseUBO::from(&camera);
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
            let final_color = descriptor_manager.request_bind(
                handle,
                0,
                3,
                BindingData::RenderGraphImage { id: final_image_id },
            );
            let radiance_info = descriptor_manager.request_bind(
                handle,
                1,
                0,
                BindingData::Uniform {
                    data: bytes_of(&radiance_info).to_vec(),
                },
            );
            let lighting_info = descriptor_manager.request_bind(
                handle,
                1,
                1,
                BindingData::Uniform {
                    data: bytes_of(&lighting_ubo).to_vec(),
                },
            );
            let camera_info = descriptor_manager.request_bind(
                handle,
                1,
                2,
                BindingData::Uniform {
                    data: bytes_of(&camera_ubo).to_vec(),
                },
            );

            let jobs = vec![DrawJob {
                mesh: DrawStyle::VertexCount(3),
                descriptor_sets: vec![
                    gbuffer_albedo,
                    gbuffer_normal,
                    gbuffer_position,
                    camera_info,
                    lighting_info,
                    radiance_info,
                    final_color,
                ],
            }];
            PipelineJob::Graphics(jobs)
        },
    )
}

impl RadianceLevelConfigUBO {
    ///  config and cascade level, where cascade0 is highest density lowest ray count, level is 0
    ///  indexed, and config.cascade_count is 1 indexed
    fn new(config: RadianceCascadesConfiguration, level: u32) -> Self {
        let volume_center = config.volume_center;
        let cascade_count = config.cascade_count;
        assert!(config.cascade_count != 0);

        let top_level_probe_count = config.top_level_probe_count;
        let top_level_probe_gap = config.top_level_probe_gap;

        let bottom_level_rays_per_probe = config.bottom_level_rays_per_probe;

        let grid_gap = get_grid_gap(top_level_probe_gap, cascade_count, level);
        let grid_size_ratio = 2_u32.pow(cascade_count - 1 - level);
        let grid_size = top_level_probe_count * grid_size_ratio;
        // by doing this ray count grows linearly with cascade count
        let sqrt_ray_count = (bottom_level_rays_per_probe * 8_u32.pow(level)).isqrt();

        let interval_start =
            get_interval_scale(level) * grid_gap * config.base_interval_length_ratio;
        let interval_end =
            get_interval_scale(level + 1) * grid_gap * config.base_interval_length_ratio;

        // set the grid origin at the place where the first probe should be, not where the corner of
        // the volume is, the math is volume center - (N points spanning N - 1 gaps, so you have to
        // subtract one from the probe count
        let grid_origin = volume_center
            - ((Vec3::from(grid_size.to_array().map(|x| x as f32)) - Vec3::ONE) * grid_gap / 2.0);

        let is_top_cascade = if level == cascade_count { 1 } else { 0 };

        Self {
            is_top_cascade,
            _pad0: 0,
            grid_origin,
            grid_size,
            grid_gap,
            sqrt_ray_count,
            interval_start,
            interval_end,
        }
    }
}
fn get_grid_gap(top_level_probe_gap: f32, cascade_count: u32, level: u32) -> f32 {
    top_level_probe_gap / 2_u32.pow(cascade_count - 1 - level) as f32
}

/// will overflow at around cascade 16, which better be more than enough
fn get_interval_scale(cascade_index: u32) -> f32 {
    if cascade_index == 0 {
        return 0.0;
    }
    return 4_u32.strict_pow(cascade_index) as f32;
}
