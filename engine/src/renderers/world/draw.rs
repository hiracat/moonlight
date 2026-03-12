#![allow(clippy::cast_possible_truncation)]

use super::pipelines::{PipelineBundle, PipelineKey, create_builtin_graphics_pipelines};
use super::swapchain::{framebuffers::GBufferResources, renderpass::create_renderpass};
use crate::ecs::World;
use crate::renderers::world::swapchain::framebuffers::create_gbuffer_resources;
use crate::renderers::world::swapchain::update_attachment_descriptor_sets;
use crate::renderers::world::swapchain::{SwapchainResources, create_attachment_descriptor_sets};
use crate::resources::{Mesh, ResourceManager};
use crate::vulkan::{SharedAllocator, VulkanContext};
use ash::vk::{self};
use gpu_allocator::vulkan::*;
use std::{io::Write, ptr};

pub const GEOMETRY_SUBPASS: u32 = 0;
pub const LIGHTING_SUBPASS: u32 = 1;
pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct WorldRenderer {
    // indexed by pipelineHandle
    pub render_pass: vk::RenderPass,
    pub pipelines: Vec<PipelineBundle>,

    pub per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,
    pub per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    pub image_descriptor_set_pool: vk::DescriptorPool,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub gbuffers: GBufferResources,
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: Option<Mesh>,
    // shaders set indices are required to be contiguous
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for WorldRenderer {
    fn drop(&mut self) {
        todo!();
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
        let clear_values = vec![
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        ];
        unsafe {
            device.cmd_begin_render_pass2(
                command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[swapchain_image_index],
                    p_clear_values: clear_values.as_ptr(),
                    clear_value_count: clear_values.len() as u32,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: window_size,
                    },
                    ..Default::default()
                },
                &vk::SubpassBeginInfo {
                    contents: vk::SubpassContents::INLINE,
                    ..Default::default()
                },
            );
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        };

        for (pipeline_index, job_list) in jobs.iter().enumerate() {
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[pipeline_index].pipeline,
                );
            }
            for job in job_list {
                assert_eq!(
                    job.descriptor_sets.len(),
                    self.pipelines[pipeline_index as usize]
                        .descriptor_set_layouts
                        .len(),
                    "must be an equal amount of descriptor sets as descriptor set layouts"
                );

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines[pipeline_index].layout,
                        0,
                        &job.descriptor_sets,
                        &[],
                    );
                }
                if let Some(mesh) = job.mesh {
                    let mesh = resource_manager.get_mesh(mesh).unwrap();

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
                        device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                    }
                } else {
                    unsafe {
                        device.cmd_draw(command_buffer, 3, 1, 0, 0);
                    }
                }
            }
            if pipeline_index == PipelineKey::AnimatedGeometry as usize {
                unsafe {
                    device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
                }
            }
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[pipeline_index as usize].pipeline,
                );
            }
        }
        unsafe {
            device.cmd_end_render_pass(command_buffer);
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
                &device,
                resource_manager,
                pool,
                world,
                &pipeline.descriptor_set_layouts,
                &self.per_swapchain_image_descriptor_sets[swapchain_image_index],
            );
            jobs.push(job_set);
        }
        return jobs;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(context: &VulkanContext, swapchain_resources: &SwapchainResources) -> Self {
        let render_pass = create_renderpass(
            &context.device,
            swapchain_resources.swapchain_image_format.format,
        );
        let pipelines = create_builtin_graphics_pipelines(&context.device, render_pass);
        //HACK: yes this is a magic number, but its defined in the shader, so not resonable to fix
        // Index 2 = ambient pipeline (first lighting pipeline), set 0 = gbuffer input attachments
        let per_swapchain_image_set_layout = pipelines[2].descriptor_set_layouts[0];

        let (framebuffers, gbuffer_resources) = create_gbuffer_resources(
            &context.device,
            &swapchain_resources.swapchain_images,
            render_pass,
            context.allocator.clone(),
            &swapchain_resources.swapchain_image_views,
            swapchain_resources.image_size,
        );

        // one set per swapchain image, * 4 for color normal position and final_color * 3 for
        // safety
        let required_sets: u32 = (swapchain_resources.swapchain_images.len() * 4 * 3) as u32;

        let pool_sizes = [vk::DescriptorPoolSize {
            // per uniform buffer, so six descriptors per
            // set(2x what is needed for safety
            descriptor_count: 6 * required_sets,
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
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

        let descriptor_sets = create_attachment_descriptor_sets(
            &context.device,
            input_descriptor_pool,
            per_swapchain_image_set_layout,
            &[
                gbuffer_resources.color_images.as_slice(),
                gbuffer_resources.normal_images.as_slice(),
                gbuffer_resources.position_images.as_slice(),
            ],
            &[0, 1, 2],
        );

        WorldRenderer {
            render_pass,
            pipelines,
            per_swapchain_image_descriptor_sets: descriptor_sets,
            image_descriptor_set_pool: input_descriptor_pool,
            per_swapchain_image_set_layout,
            framebuffers: framebuffers,
            gbuffers: gbuffer_resources,
        }
    }

    pub fn update_swapchain_resources(
        &mut self,
        context: &VulkanContext,
        swapchain_resources: &SwapchainResources,
    ) {
        let (framebuffers, gbuffer_resources) = create_gbuffer_resources(
            &context.device,
            &swapchain_resources.swapchain_images,
            self.render_pass,
            context.allocator.clone(),
            &swapchain_resources.swapchain_image_views,
            swapchain_resources.image_size,
        );
        self.framebuffers = framebuffers;

        update_attachment_descriptor_sets(
            &context.device,
            &self.per_swapchain_image_descriptor_sets,
            &[
                gbuffer_resources.color_images.as_slice(),
                gbuffer_resources.normal_images.as_slice(),
                gbuffer_resources.position_images.as_slice(),
            ],
            &[0, 1, 2],
        );
        self.gbuffers = gbuffer_resources;
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
