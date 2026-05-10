#![allow(clippy::cast_possible_truncation)]

use super::pipelines::create_builtin_graphics_pipelines;
use crate::ecs::World;
use crate::renderers::world::descriptors::{BindingHandle, DescriptorManager};
use crate::renderers::world::pipelines::{PipelineHandle, PipelineManager};
use crate::renderers::world::rendergraph::{
    CompiledRenderGraph, GraphCommand, ImageId, ImportedImageBinding, RenderGraph,
};
use crate::renderers::world::swapchain::SwapchainResources;
use crate::resources::{Mesh, ResourceManager};
use crate::vulkan::{SharedAllocator, VulkanContext};
use ash::vk::{self};
use gpu_allocator::vulkan::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::{io::Write, ptr};
use tracing::{instrument, trace};

pub const FRAMES_IN_FLIGHT: usize = 2;

//HACK: this is bad, but i just wanna get it to work, ill fix it later
pub struct Ids {
    pub albedo: ImageId,
    pub position: ImageId,
    pub normal: ImageId,
    pub final_color: ImageId,
}

pub struct WorldRenderer {
    swapchain_image_id: ImageId,

    // indexed by PipelineHandle
    pub pipeline_manager: PipelineManager,
    pub rendergraph_config: RenderGraph,
    // one per swapchain image
    pub graph: Vec<CompiledRenderGraph>,

    pub descriptor_manager: [DescriptorManager; FRAMES_IN_FLIGHT],

    device: Arc<ash::Device>,
}

#[derive(Debug)]
pub enum PipelineJob {
    Graphics(Vec<DrawJob>),
    Compute(ComputeDispatch),
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: DrawStyle,
    pub descriptor_sets: Vec<BindingHandle>,
}

#[derive(Debug)]
pub enum DrawStyle {
    Mesh(Mesh),
    VertexCount(u32),
}

#[derive(Debug)]
pub struct ComputeDispatch {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub bindings: Vec<BindingHandle>,
}

impl WorldRenderer {
    #[allow(clippy::too_many_lines)]
    pub fn record_commands(
        &mut self,
        world: &mut World,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        window_size: vk::Extent2D,
        resource_manager: &mut ResourceManager,
        swapchain_image_index: usize,
        frame_in_flight: usize,
    ) {
        self.descriptor_manager[frame_in_flight].begin_frame();
        let jobs =
            self.setup_gpu_build_draw_jobs(resource_manager, world, frame_in_flight, window_size);
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
        for cmd in &graph.commands {
            match cmd {
                GraphCommand::ClearImage {
                    image,
                    image_layout,
                    range,
                } => {
                    unsafe {
                        self.device.cmd_clear_color_image(
                            command_buffer,
                            graph.get_image_from_id(image),
                            *image_layout,
                            &vk::ClearColorValue::default(),
                            &[*range],
                        )
                    };
                }
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
                    let pipeline = self.pipeline_manager.get(pipeline_handle).pipeline;
                    unsafe {
                        device.cmd_bind_pipeline(command_buffer, *pipeline_bind_point, pipeline);
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                    }
                    let job_list = &jobs[pipeline_handle];
                    match job_list {
                        PipelineJob::Graphics(draw_jobs) => {
                            for job in draw_jobs {
                                let (pipeline_layout, set_layout) =
                                    &self.descriptor_manager[frame_in_flight].bind(
                                        resource_manager,
                                        graph,
                                        *pipeline_handle,
                                        &job.descriptor_sets,
                                    );
                                unsafe {
                                    device.cmd_bind_descriptor_sets(
                                        command_buffer,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        *pipeline_layout,
                                        0,
                                        set_layout.as_ref(),
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
                        PipelineJob::Compute(compute_dispatch) => {
                            let (pipeline_layout, descriptor_sets) =
                                &self.descriptor_manager[frame_in_flight].bind(
                                    resource_manager,
                                    graph,
                                    *pipeline_handle,
                                    &compute_dispatch.bindings,
                                );
                            unsafe {
                                device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::COMPUTE,
                                    *pipeline_layout,
                                    0,
                                    descriptor_sets.as_ref(),
                                    &[],
                                );
                            }
                            unsafe {
                                device.cmd_dispatch(
                                    command_buffer,
                                    compute_dispatch.x,
                                    compute_dispatch.y,
                                    compute_dispatch.z,
                                )
                            };
                        }
                    }
                }
            }
        }
    }

    #[instrument(
        level = "trace",
        name = "draw job building",
        skip(self, resource_manager, world)
    )]
    fn setup_gpu_build_draw_jobs(
        &mut self,
        resource_manager: &mut ResourceManager,
        world: &mut World,
        frame_in_flight: usize,
        extent: vk::Extent2D,
    ) -> HashMap<PipelineHandle, PipelineJob> {
        let mut jobs: HashMap<PipelineHandle, PipelineJob> = HashMap::new();

        for (handle, pipeline) in self.pipeline_manager.all_pipelines() {
            let job_set = (pipeline.write_data_and_build_draw_jobs)(
                world,
                resource_manager,
                &mut self.descriptor_manager[frame_in_flight],
                handle,
                extent,
            );
            jobs.insert(handle, job_set);
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
        let (graph, pipelines, descriptor_managers, final_color) =
            create_builtin_graphics_pipelines(
                context.device.clone(),
                context.allocator(),
                swapchain_resources.swapchain_image_format.format,
                vk::Format::D32_SFLOAT,
                &color_formats,
            );

        let mut graphs = Vec::new();
        for i in 0..swapchain_resources.swapchain_images.len() {
            let mut imported_images = HashMap::new();
            imported_images.insert(
                final_color,
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

        WorldRenderer {
            descriptor_manager: descriptor_managers,
            swapchain_image_id: final_color,
            graph: graphs,
            rendergraph_config: graph,
            device: context.device.clone(),
            pipeline_manager: pipelines,
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
