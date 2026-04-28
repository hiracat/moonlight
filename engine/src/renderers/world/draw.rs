#![allow(clippy::cast_possible_truncation)]

use super::pipelines::{PipelineBundle, PipelineKey, create_builtin_graphics_pipelines};
use crate::ecs::World;
use crate::renderers::world::pipelines::PipelineHandle;
use crate::renderers::world::rendergraph::{
    CompiledRenderGraph, GraphCommand, ImageDesc, ImageId, ImportedImageBinding, RenderGraph,
};
use crate::renderers::world::swapchain::update_attachment_descriptor_sets;
use crate::renderers::world::swapchain::{SwapchainResources, create_attachment_descriptor_sets};
use crate::resources::{Mesh, ResourceManager};
use crate::vulkan::{SharedAllocator, VulkanContext};
use ash::vk::{self};
use gpu_allocator::vulkan::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::{io::Write, ptr};

pub const GEOMETRY_SUBPASS: u32 = 0;
pub const LIGHTING_SUBPASS: u32 = 1;
pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct WorldRenderer {
    swapchain_image_id: ImageId,
    image_attachment_order: [ImageId; 3],

    // indexed by PipelineHandle
    pub pipelines: Vec<PipelineBundle>,
    pub rendergraph_config: RenderGraph,
    // one per swapchain image
    pub graph: Vec<CompiledRenderGraph>,

    pub per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,
    pub per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    pub image_descriptor_set_pool: vk::DescriptorPool,
    pub sampler: vk::Sampler,
    device: Arc<ash::Device>,
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: DrawStyle,
    // shaders set indices are required to be contiguous
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

#[derive(Debug)]
pub enum DrawStyle {
    Mesh(Mesh),
    VertexCount(u32),
}

impl Drop for WorldRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.image_descriptor_set_pool, None);
            self.device.destroy_sampler(self.sampler, None);
            // this is destroyed by pipelienbundle::drop, because its a random index that happens
            // to correspond with the final color output, but is really owned by them
            // self.device
            //     .destroy_descriptor_set_layout(self.per_swapchain_image_set_layout, None);
        }
    }
}

impl WorldRenderer {
    #[allow(clippy::too_many_lines)]
    pub fn record_commands(
        &mut self,
        world: &mut World,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        descriptor_pool: vk::DescriptorPool,
        window_size: vk::Extent2D,
        resource_manager: &mut ResourceManager,
        swapchain_image_index: usize,
    ) {
        let jobs = self.setup_gpu_build_draw_jobs(
            device,
            swapchain_image_index,
            resource_manager,
            descriptor_pool,
            world,
        );
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: window_size,
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size.width as f32,
            height: window_size.height as f32,
            max_depth: 1.0,
            min_depth: 0.0,
        };

        let graph = &self.graph[swapchain_image_index];
        for cmd in graph.commands() {
            match cmd {
                GraphCommand::BeginRendering {
                    color_attachments,
                    depth,
                } => unsafe {
                    let color_attachments_vk: Vec<_> =
                        color_attachments.iter().map(|x| x.to_vulkan()).collect();

                    device.cmd_begin_rendering(
                        command_buffer,
                        &vk::RenderingInfo {
                            render_area: vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: window_size,
                            },
                            layer_count: 1,
                            color_attachment_count: color_attachments_vk.len() as u32,
                            p_color_attachments: color_attachments_vk.as_ptr(),
                            p_depth_attachment: &depth
                                .as_ref()
                                .map(|x| x.to_vulkan())
                                .unwrap_or(vk::RenderingAttachmentInfo::default()),
                            ..Default::default()
                        },
                    );
                },
                GraphCommand::EndRendering => unsafe {
                    device.cmd_end_rendering(command_buffer);
                },
                GraphCommand::ImageBarrier {
                    image_id,
                    src_layout,
                    dst_layout,
                    src_stage,
                    dst_stage,
                    src_access,
                    dst_access,
                    aspect_mask,
                } => {
                    let image_barriers = [vk::ImageMemoryBarrier2 {
                        image: graph.get_image_from_id(image_id),
                        old_layout: *src_layout,
                        new_layout: *dst_layout,
                        src_stage_mask: *src_stage,
                        src_access_mask: *src_access,
                        dst_stage_mask: *dst_stage,
                        dst_access_mask: *dst_access,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: *aspect_mask,
                            level_count: 1,
                            layer_count: 1,
                            base_mip_level: 0,
                            base_array_layer: 0,
                        },
                        ..Default::default()
                    }];
                    let dependancy_info = vk::DependencyInfo {
                        //PERF: need to change this to come from rendergraph, cause it will know if
                        //its by region or not
                        dependency_flags: vk::DependencyFlags::empty(),
                        p_image_memory_barriers: image_barriers.as_ptr(),
                        image_memory_barrier_count: image_barriers.len() as u32,
                        ..Default::default()
                    };
                    unsafe {
                        device.cmd_pipeline_barrier2(command_buffer, &dependancy_info);
                    }
                }
                GraphCommand::BindPipeline(pipeline_bind_point, pipeline_handle) => {
                    let pipeline = self.pipelines[pipeline_handle.arr_index].pipeline;
                    unsafe {
                        device.cmd_bind_pipeline(command_buffer, *pipeline_bind_point, pipeline);
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                    }
                    let job_list = &jobs[pipeline_handle.arr_index];
                    for job in job_list {
                        assert_eq!(
                            job.descriptor_sets.len(),
                            self.pipelines[pipeline_handle.arr_index]
                                .descriptor_set_layouts
                                .len(),
                            "must be an equal amount of descriptor sets as descriptor set layouts"
                        );

                        unsafe {
                            device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipelines[pipeline_handle.arr_index].layout,
                                0,
                                &job.descriptor_sets,
                                &[],
                            );
                        }
                        match job.mesh {
                            DrawStyle::Mesh(x) => {
                                let mesh = resource_manager.get_mesh(x).unwrap();

                                unsafe {
                                    device.cmd_bind_vertex_buffers(
                                        command_buffer,
                                        0,
                                        &[mesh.vertex_buffer],
                                        &[0],
                                    );
                                    device.cmd_bind_index_buffer(
                                        command_buffer,
                                        mesh.index_buffer,
                                        0,
                                        vk::IndexType::UINT32,
                                    );
                                    device.cmd_draw_indexed(
                                        command_buffer,
                                        mesh.index_count,
                                        1,
                                        0,
                                        0,
                                        0,
                                    );
                                }
                            }
                            DrawStyle::VertexCount(count) => unsafe {
                                device.cmd_draw(command_buffer, count, 1, 0, 0);
                            },
                        }
                    }
                }
            }
        }
    }

    // this is sorta like a frame graph builder kinda thing
    fn setup_gpu_build_draw_jobs(
        &mut self,
        device: &ash::Device,
        swapchain_image_index: usize,
        resource_manager: &mut ResourceManager,
        pool: vk::DescriptorPool,
        world: &mut World,
    ) -> Vec<Vec<DrawJob>> {
        // indexed by pipelinekey, then just everything that belongs in that pipeline
        let mut jobs: Vec<Vec<DrawJob>> = Vec::with_capacity(self.pipelines.len());

        for pipeline in &self.pipelines {
            let job_set = (pipeline.write_data_and_build_draw_jobs)(
                device,
                resource_manager,
                pool,
                world,
                &pipeline.descriptor_set_layouts,
                &self.per_swapchain_image_descriptor_sets[swapchain_image_index],
            );
            jobs.push(job_set);
        }
        jobs
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(context: &VulkanContext, swapchain_resources: &SwapchainResources) -> Self {
        let color_formats = [
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
        ];
        let pipelines = create_builtin_graphics_pipelines(
            context.device.clone(),
            swapchain_resources.swapchain_image_format.format,
            vk::Format::D32_SFLOAT,
            &color_formats,
        );
        //HACK: yes this is a magic number, but its defined in the shader, so not resonable to fix
        // Index 2 = ambient pipeline (first lighting pipeline), set 0 = gbuffer input attachments
        let per_swapchain_image_set_layout =
            pipelines[PipelineKey::Ambient as usize].descriptor_set_layouts[0];
        let sampler = unsafe {
            context.device.create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();

        // one set per swapchain image, * 4 for color normal position and final_color * 3 for
        // safety
        let required_sets: u32 = (swapchain_resources.swapchain_images.len() * 4 * 3) as u32;

        let pool_sizes = [vk::DescriptorPoolSize {
            // per uniform buffer, so six descriptors per
            // set(2x what is needed for safety
            descriptor_count: 6 * required_sets,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }];

        let input_descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: required_sets,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        pool_size_count: pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        let _image_size = swapchain_resources.image_size;
        let mut graph = RenderGraph::new();

        let mut final_color = graph.add_image(ImageDesc::Imported {
            name: "final_color",
            format: swapchain_resources.swapchain_image_format.format,
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
        let graph = graph
            .add_pipeline("static_geometry")
            .pipeline(PipelineHandle { arr_index: 0 })
            .writes(&mut albedo)
            .writes(&mut normal)
            .writes(&mut position)
            .writes_depth(&mut depth)
            .build();
        let graph = graph
            .add_pipeline("animated_geometry")
            .pipeline(PipelineHandle { arr_index: 1 })
            .writes(&mut albedo)
            .writes(&mut normal)
            .writes(&mut position)
            .writes_depth(&mut depth)
            .build();
        let graph = graph
            .add_pipeline("terrain")
            .pipeline(PipelineHandle { arr_index: 2 })
            .writes(&mut albedo)
            .writes(&mut normal)
            .writes(&mut position)
            .writes_depth(&mut depth)
            .build();
        let graph = graph
            .add_pipeline("ambient_light")
            .pipeline(PipelineHandle { arr_index: 3 })
            .reads(&albedo)
            .reads(&normal)
            .reads(&position)
            .reads_depth(&depth)
            .writes(&mut final_color)
            .build();
        let graph = graph
            .add_pipeline("directional_light")
            .pipeline(PipelineHandle { arr_index: 4 })
            .reads(&albedo)
            .reads(&normal)
            .reads(&position)
            .reads_depth(&depth)
            .writes(&mut final_color)
            .build();
        let graph = graph
            .add_pipeline("point_light")
            .pipeline(PipelineHandle { arr_index: 5 })
            .reads(&albedo)
            .reads(&normal)
            .reads(&position)
            .reads_depth(&depth)
            .writes(&mut final_color)
            .build();

        let graph = graph
            .add_pipeline("skybox")
            .pipeline(PipelineHandle { arr_index: 6 })
            .writes(&mut final_color)
            .reads_depth(&depth)
            .build();

        let mut graphs = Vec::new();
        for i in 0..swapchain_resources.swapchain_images.len() {
            let mut imported_images = HashMap::new();
            imported_images.insert(
                final_color.id,
                ImportedImageBinding {
                    pre_initialized: false,
                    image: swapchain_resources.swapchain_images[i],
                    view: swapchain_resources.swapchain_image_views[i],
                    current_layout: vk::ImageLayout::UNDEFINED,
                    last_access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                },
            );
            graphs.push(graph.compile(
                context.device.clone(),
                context.allocator(),
                &imported_images,
                swapchain_resources.image_size.width,
                swapchain_resources.image_size.height,
            ));
        }

        let albedo_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&albedo.id))
            .collect();
        let normal_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&normal.id))
            .collect();
        let position_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&position.id))
            .collect();

        let descriptor_sets = create_attachment_descriptor_sets(
            &context.device,
            sampler,
            input_descriptor_pool,
            per_swapchain_image_set_layout,
            &[
                albedo_views.as_slice(),
                normal_views.as_slice(),
                position_views.as_slice(),
            ],
            &[0, 1, 2],
        );

        WorldRenderer {
            image_attachment_order: [albedo.id, normal.id, position.id],
            swapchain_image_id: final_color.id,
            graph: graphs,
            rendergraph_config: graph,
            device: context.device.clone(),
            pipelines,
            per_swapchain_image_descriptor_sets: descriptor_sets,
            image_descriptor_set_pool: input_descriptor_pool,
            per_swapchain_image_set_layout,
            sampler,
        }
    }

    pub fn update_swapchain_resources(
        &mut self,
        context: &VulkanContext,
        swapchain_resources: &SwapchainResources,
    ) {
        let mut graphs = Vec::new();
        for i in 0..swapchain_resources.swapchain_images.len() {
            let mut imported_images = HashMap::new();
            imported_images.insert(
                self.swapchain_image_id,
                ImportedImageBinding {
                    pre_initialized: false,
                    image: swapchain_resources.swapchain_images[i],
                    view: swapchain_resources.swapchain_image_views[i],
                    current_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    last_access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                },
            );
            graphs.push(self.rendergraph_config.compile(
                context.device.clone(),
                context.allocator(),
                &imported_images,
                swapchain_resources.image_size.width,
                swapchain_resources.image_size.height,
            ));
        }

        let albedo_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&self.image_attachment_order[0]))
            .collect();
        let normal_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&self.image_attachment_order[1]))
            .collect();
        let position_views: Vec<vk::ImageView> = graphs
            .iter()
            .map(|g| g.get_view_from_id(&self.image_attachment_order[2]))
            .collect();

        update_attachment_descriptor_sets(
            &context.device,
            &self.per_swapchain_image_descriptor_sets,
            &[
                albedo_views.as_slice(),
                normal_views.as_slice(),
                position_views.as_slice(),
            ],
            &[0, 1, 2],
            self.sampler,
        );

        self.graph = graphs;
    }
}

pub(crate) fn instant_submit_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_record: impl FnOnce(vk::CommandBuffer),
) {
    let begin_info = vk::CommandBufferBeginInfo {
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap()
    };

    command_record(command_buffer);

    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    let submits = vk::SubmitInfo {
        command_buffer_count: 1,
        p_wait_semaphores: ptr::null(),
        p_command_buffers: &command_buffer,
        p_signal_semaphores: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_dst_stage_mask: ptr::null(),
        signal_semaphore_count: 0,
        ..Default::default()
    };
    let create_info = vk::FenceCreateInfo {
        ..Default::default()
    };
    let fence = unsafe { device.create_fence(&create_info, None).unwrap() };

    unsafe { device.queue_submit(queue, &[submits], fence).unwrap() }

    let wait = [fence];
    unsafe { device.wait_for_fences(&wait, true, u64::MAX).unwrap() };
    unsafe {
        device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap()
    };
    unsafe { device.destroy_fence(fence, None) };
}

pub fn alloc_buffers(
    memory_allocator: SharedAllocator,
    buffer_count: usize,
    size: u64,
    device: &ash::Device,
    sharing: vk::SharingMode,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
    linear: bool,
    data: &[u8],
    name: &str,
) -> (Vec<vk::Buffer>, Vec<Allocation>) {
    let mut buffers = vec![];
    let mut allocations = vec![];
    for _ in 0..buffer_count {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    sharing_mode: sharing,
                    usage,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut alloc = memory_allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        alloc.mapped_slice_mut().unwrap().write_all(data).unwrap();
        unsafe {
            device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                .unwrap()
        }

        allocations.push(alloc);
        buffers.push(buffer);
    }
    (buffers, allocations)
}
