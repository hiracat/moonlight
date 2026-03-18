#![allow(clippy::cast_possible_truncation)]

use super::pipelines::{create_builtin_graphics_pipelines, PipelineBundle, PipelineKey};
use super::swapchain::image_attachments::GBufferResources;
use crate::ecs::World;
use crate::renderers::world::swapchain::image_attachments::create_gbuffer_resources;
use crate::renderers::world::swapchain::update_attachment_descriptor_sets;
use crate::renderers::world::swapchain::{create_attachment_descriptor_sets, SwapchainResources};
use crate::resources::{Mesh, ResourceManager};
use crate::vulkan::{SharedAllocator, VulkanContext};
use ash::vk::{self};
use gpu_allocator::vulkan::*;
use std::sync::Arc;
use std::{io::Write, ptr};

pub const GEOMETRY_SUBPASS: u32 = 0;
pub const LIGHTING_SUBPASS: u32 = 1;
pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct WorldRenderer {
    // indexed by pipelineHandle
    pub pipelines: Vec<PipelineBundle>,

    pub per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,
    pub per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    pub image_descriptor_set_pool: vk::DescriptorPool,
    pub gbuffers: GBufferResources,
    pub sampler: vk::Sampler,
    device: Arc<ash::Device>,
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: Option<Mesh>,
    // shaders set indices are required to be contiguous
    pub descriptor_sets: Vec<vk::DescriptorSet>,
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
        swapchain_image_view: vk::ImageView,
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
        //HACK: this is a magic array, because i store color normal and position, need
        //to do something else to fix this
        let color_images = vec![
            vk::RenderingAttachmentInfo {
                image_view: self.gbuffers.color_images[swapchain_image_index].view,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue::default(),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
            vk::RenderingAttachmentInfo {
                image_view: self.gbuffers.normal_images[swapchain_image_index].view,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue::default(),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
            vk::RenderingAttachmentInfo {
                image_view: self.gbuffers.position_images[swapchain_image_index].view,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue::default(),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let depth_info = vk::RenderingAttachmentInfo {
            image_view: self.gbuffers.depth_images[swapchain_image_index].view,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let image_barriers = [
            vk::ImageMemoryBarrier2 {
                image: self.gbuffers.depth_images[swapchain_image_index].image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                dst_access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    level_count: 1,
                    layer_count: 1,
                    base_mip_level: 0,
                    base_array_layer: 0,
                },
                ..Default::default()
            },
            vk::ImageMemoryBarrier2 {
                image: self.gbuffers.color_images[swapchain_image_index].image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    base_mip_level: 0,
                    base_array_layer: 0,
                },
                ..Default::default()
            },
            vk::ImageMemoryBarrier2 {
                image: self.gbuffers.normal_images[swapchain_image_index].image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    base_mip_level: 0,
                    base_array_layer: 0,
                },
                ..Default::default()
            },
            vk::ImageMemoryBarrier2 {
                image: self.gbuffers.position_images[swapchain_image_index].image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    base_mip_level: 0,
                    base_array_layer: 0,
                },
                ..Default::default()
            },
        ];
        let dependancy_info = vk::DependencyInfo {
            dependency_flags: vk::DependencyFlags::BY_REGION,
            p_image_memory_barriers: image_barriers.as_ptr(),
            image_memory_barrier_count: image_barriers.len() as u32,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependancy_info);
        }
        unsafe {
            device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo {
                    render_area: vk::Rect2D {
                        extent: window_size,
                        offset: vk::Offset2D { x: 0, y: 0 },
                    },
                    flags: vk::RenderingFlags::empty(),
                    layer_count: 1,
                    color_attachment_count: color_images.len() as u32,
                    p_color_attachments: color_images.as_ptr(),
                    p_depth_attachment: &depth_info as *const _,
                    p_stencil_attachment: ptr::null(),
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
                    device.cmd_end_rendering(command_buffer);
                }

                let image_barriers = [
                    vk::ImageMemoryBarrier2 {
                        image: self.gbuffers.depth_images[swapchain_image_index].image,
                        old_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                        src_stage_mask: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        src_access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                        dst_access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            level_count: 1,
                            layer_count: 1,
                            base_mip_level: 0,
                            base_array_layer: 0,
                        },
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier2 {
                        image: self.gbuffers.color_images[swapchain_image_index].image,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        dst_access_mask: vk::AccessFlags2::SHADER_READ,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: 1,
                            base_mip_level: 0,
                            base_array_layer: 0,
                        },
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier2 {
                        image: self.gbuffers.normal_images[swapchain_image_index].image,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        dst_access_mask: vk::AccessFlags2::SHADER_READ,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: 1,
                            base_mip_level: 0,
                            base_array_layer: 0,
                        },
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier2 {
                        image: self.gbuffers.position_images[swapchain_image_index].image,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        dst_access_mask: vk::AccessFlags2::SHADER_READ,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: 1,
                            base_mip_level: 0,
                            base_array_layer: 0,
                        },
                        ..Default::default()
                    },
                ];
                let dependancy_info = vk::DependencyInfo {
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                    p_image_memory_barriers: image_barriers.as_ptr(),
                    image_memory_barrier_count: image_barriers.len() as u32,
                    ..Default::default()
                };
                unsafe {
                    device.cmd_pipeline_barrier2(command_buffer, &dependancy_info);
                }
                let lighting_attachments = [vk::RenderingAttachmentInfo {
                    image_view: swapchain_image_view,
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    ..Default::default()
                }];
                let mut depth_attachment = vk::RenderingAttachmentInfo {
                    image_view: self.gbuffers.depth_images[swapchain_image_index].view,
                    image_layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                    load_op: vk::AttachmentLoadOp::LOAD,
                    store_op: vk::AttachmentStoreOp::NONE,
                    ..Default::default()
                };
                unsafe {
                    device.cmd_begin_rendering(
                        command_buffer,
                        &vk::RenderingInfo {
                            render_area: vk::Rect2D {
                                extent: window_size,
                                offset: vk::Offset2D { x: 0, y: 0 },
                            },
                            layer_count: 1,
                            color_attachment_count: 1,
                            p_color_attachments: lighting_attachments.as_ptr(),
                            p_depth_attachment: &mut depth_attachment as *mut _,
                            p_stencil_attachment: ptr::null(),
                            ..Default::default()
                        },
                    );
                }
            }
        }
        unsafe {
            device.cmd_end_rendering(command_buffer); // end lighting pass
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
        let per_swapchain_image_set_layout = pipelines[2].descriptor_set_layouts[0];
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

        let gbuffer_resources = create_gbuffer_resources(
            &context.device,
            &swapchain_resources.swapchain_images,
            context.allocator().clone(),
            swapchain_resources.image_size,
        );

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

        let descriptor_sets = create_attachment_descriptor_sets(
            &context.device,
            sampler,
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
            device: context.device.clone(),
            pipelines,
            per_swapchain_image_descriptor_sets: descriptor_sets,
            image_descriptor_set_pool: input_descriptor_pool,
            per_swapchain_image_set_layout,
            gbuffers: gbuffer_resources,
            sampler: sampler,
        }
    }

    pub fn update_swapchain_resources(
        &mut self,
        context: &VulkanContext,
        swapchain_resources: &SwapchainResources,
    ) {
        let gbuffer_resources = create_gbuffer_resources(
            &context.device,
            &swapchain_resources.swapchain_images,
            context.allocator().clone(),
            swapchain_resources.image_size,
        );

        update_attachment_descriptor_sets(
            &context.device,
            &self.per_swapchain_image_descriptor_sets,
            &[
                gbuffer_resources.color_images.as_slice(),
                gbuffer_resources.normal_images.as_slice(),
                gbuffer_resources.position_images.as_slice(),
            ],
            &[0, 1, 2],
            self.sampler,
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
