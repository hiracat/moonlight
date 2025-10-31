#![allow(clippy::cast_possible_truncation)]

use crate::ecs::World;
use crate::renderer::init::{
    create_device, create_instance, create_physical_device, setup_debug_utils,
};
use crate::renderer::pipelines::{create_builtin_graphics_pipelines, PipelineBundle, PipelineKey};
use crate::renderer::resources::{Mesh, ResourceManager};
use crate::renderer::swapchain::renderpass::create_renderpass;
use crate::{components::Camera, renderer::swapchain::SwapchainResources};
use ash::vk;
use gpu_allocator::vulkan::*;
use std::{default::Default, ptr};
use std::{
    hash::Hash,
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
    usize,
};
//INFO: idk how to fix this, one wants raw window handles and the other says no
#[allow(deprecated)]
use winit::{
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

pub const VALIDATION_ENABLE: bool = true;
pub const GEOMETRY_SUBPASS: u32 = 0;
pub const LIGHTING_SUBPASS: u32 = 1;
pub const FRAMES_IN_FLIGHT: usize = 2;

pub type SharedAllocator = Arc<Mutex<Allocator>>;
pub type QueueFamilyIndex = u32;

pub struct Renderer {
    // indexed by pipelineHandle
    pipelines: Vec<PipelineBundle>,

    // needs to be kept alive, dont forget is very important
    // anything that starts with ash:: and not vk:: impliments drop
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) surface_loader: ash::khr::surface::Instance,
    pub(crate) debug_utils_loader: ash::ext::debug_utils::Instance,

    pub window: Arc<Window>,
    surface: vk::SurfaceKHR,

    pub resource_manager: ResourceManager,

    pub framebuffer_resized: bool,
    pub start_time: Instant,
    swapchain_image_index: u32,
    frame_counter: u64,
    current_frame: usize,

    render_pass: vk::RenderPass,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family_index: QueueFamilyIndex,

    pub(crate) swapchain: SwapchainResources,
    per_frame: Vec<PerFrame>,
    pub(crate) allocator: SharedAllocator,

    pub(crate) one_time_submit_pool: vk::CommandPool,
    pub(crate) one_time_submit: vk::CommandBuffer,
}

#[derive(Debug)]
pub struct DrawJob {
    pub mesh: Option<Mesh>,
    // shaders set indices are required to be contiguous
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}
struct PerFrame {
    pub(crate) command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    // wait for the frame that was FRAMES_IN_FLIGHT frames ago, but has the same current_frame
    // since modulo
    in_flight: vk::Fence,

    transient_pool: vk::DescriptorPool,
}

impl PerFrame {
    fn create(device: &ash::Device, queue_family_index: QueueFamilyIndex) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()
                .first()
                .copied()
                .unwrap()
        };
        let in_flight = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        // One pool with multiple descriptor types
        let transient_pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 500,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 100,
            },
        ];

        let transient_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: 1000,
                        p_pool_sizes: transient_pool_sizes.as_ptr(),
                        pool_size_count: transient_pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let image_available = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        Self {
            command_pool,
            command_buffer,
            in_flight,
            transient_pool,
            image_available,
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION_ENABLE {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl Renderer {
    #[allow(clippy::too_many_lines)]
    pub fn draw2(&mut self, world: &mut World) {
        self.window.request_redraw();
        let window_size = self.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        let frame = &self.per_frame[self.current_frame];
        unsafe {
            self.device
                .wait_for_fences(&[frame.in_flight], true, u64::MAX)
                .unwrap();
        };

        self.frame_counter += 1;

        if self.framebuffer_resized {
            // Use the new dimensions of the window.
            self.swapchain.recreate(window_size);
            world.get_mut_resource::<Camera>().unwrap().aspect_ratio =
                window_size.width as f32 / window_size.height as f32;
        }

        let is_suboptimal;
        (self.swapchain_image_index, is_suboptimal) = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            ) {
                Ok((index, suboptimal)) => (index, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.framebuffer_resized = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };
        if is_suboptimal {
            self.framebuffer_resized = true;
        }

        // NOTE: RENDERING START

        let frame = &self.per_frame[self.current_frame];

        unsafe { self.device.reset_fences(&[frame.in_flight]).unwrap() };
        unsafe {
            self.device
                .reset_descriptor_pool(frame.transient_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }

        unsafe {
            self.device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }

        let jobs = Self::setup_gpu_build_draw_jobs(self, world);

        let frame = &self.per_frame[self.current_frame];

        unsafe {
            self.device
                .begin_command_buffer(
                    frame.command_buffer,
                    &vk::CommandBufferBeginInfo {
                        p_inheritance_info: &vk::CommandBufferInheritanceInfo {
                            pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
                            subpass: 0, // ingored
                            render_pass: self.render_pass,
                            framebuffer: self.swapchain.framebuffers.framebuffers
                                [self.swapchain_image_index as usize],
                            query_flags: vk::QueryControlFlags::empty(),
                            occlusion_query_enable: vk::FALSE,
                            ..Default::default()
                        },
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap()
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
            self.device.cmd_begin_render_pass2(
                frame.command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.swapchain.framebuffers.framebuffers
                        [self.swapchain_image_index as usize],
                    p_clear_values: clear_values.as_ptr(),
                    clear_value_count: clear_values.len() as u32,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: window_size.width,
                            height: window_size.height,
                        },
                    },
                    ..Default::default()
                },
                &vk::SubpassBeginInfo {
                    contents: vk::SubpassContents::INLINE,
                    ..Default::default()
                },
            );
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: window_size.width as f32,
                height: window_size.height as f32,
                max_depth: 1.0,
                min_depth: 0.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
            };
            self.device
                .cmd_set_viewport(frame.command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(frame.command_buffer, 0, &[scissor]);
        };

        for (pipeline_index, job_list) in jobs.iter().enumerate() {
            unsafe {
                self.device.cmd_bind_pipeline(
                    frame.command_buffer,
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
                    self.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines[pipeline_index].layout,
                        0,
                        &job.descriptor_sets,
                        &[],
                    );
                }
                if let Some(mesh) = job.mesh {
                    let mesh = self.resource_manager.get_mesh(mesh).unwrap();
                    unsafe {
                        self.device.cmd_bind_vertex_buffers(
                            frame.command_buffer,
                            0,
                            &[mesh.vertex_buffer],
                            &[0],
                        );
                        self.device.cmd_bind_index_buffer(
                            frame.command_buffer,
                            mesh.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        self.device.cmd_draw_indexed(
                            frame.command_buffer,
                            mesh.index_count,
                            1,
                            0,
                            0,
                            0,
                        );
                    }
                } else {
                    unsafe {
                        self.device.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
                    }
                }
            }
            if pipeline_index == PipelineKey::Geometry as usize {
                unsafe {
                    self.device
                        .cmd_next_subpass(frame.command_buffer, vk::SubpassContents::INLINE);
                }
            }
            unsafe {
                self.device.cmd_bind_pipeline(
                    frame.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[pipeline_index as usize].pipeline,
                );
            }
        }
        unsafe {
            self.device.cmd_end_render_pass(frame.command_buffer);
            self.device
                .end_command_buffer(frame.command_buffer)
                .unwrap();
        }

        let wait_dst_access_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let queue_submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &frame.image_available,
            p_wait_dst_stage_mask: &wait_dst_access_mask,
            command_buffer_count: 1,
            p_command_buffers: &frame.command_buffer,
            p_signal_semaphores: &self.swapchain.render_finished
                [self.swapchain_image_index as usize],
            signal_semaphore_count: 1,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.queue, &[queue_submit_info], frame.in_flight)
                .unwrap();
        }

        let mut present_results: [vk::Result; 1] = [Default::default(); 1];
        let present_info = vk::PresentInfoKHR {
            p_wait_semaphores: &self.swapchain.render_finished[self.swapchain_image_index as usize],
            wait_semaphore_count: 1,
            p_swapchains: &self.swapchain.swapchain,
            swapchain_count: 1,
            p_image_indices: &self.swapchain_image_index,
            p_results: present_results.as_mut_ptr(),
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.framebuffer_resized = false;
            self.swapchain.recreate(window_size);
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    // this is sorta like a frame graph builder kinda thing
    fn setup_gpu_build_draw_jobs(&mut self, world: &mut World) -> Vec<Vec<DrawJob>> {
        // indexed by pipelinekey, then just everything that belongs in that pipeline
        let mut jobs: Vec<Vec<DrawJob>> = Vec::with_capacity(self.pipelines.len());

        for pipeline in &self.pipelines {
            let job_set = (pipeline.write_data_and_build_draw_jobs)(
                &self.device,
                &mut self.resource_manager,
                self.per_frame[self.current_frame].transient_pool,
                world,
                &pipeline.descriptor_set_layouts,
                &self.swapchain.per_swapchain_image_descriptor_sets
                    [self.swapchain_image_index as usize],
            );
            jobs.push(job_set);
        }
        return jobs;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(event_loop: &ActiveEventLoop, window: &Arc<Window>) -> Self {
        let start_time = Instant::now();
        let entry = unsafe { ash::Entry::load().unwrap() };
        eprintln!("created entry");
        let instance = create_instance(&entry, event_loop);
        eprintln!("created instance");
        let surface = unsafe {
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                // idk what to do about this, i need the raw handle
                #[allow(deprecated)]
                event_loop.raw_display_handle().unwrap(),
                #[allow(deprecated)]
                window.raw_window_handle().unwrap(),
                None,
            )
            .unwrap();
            surface
        };

        eprintln!("created surface");
        //need to make this optional to put stuff inside
        if VALIDATION_ENABLE {}
        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = setup_debug_utils(&debug_utils_loader);
        eprintln!("set up debug utility");

        let required_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME,
            // ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME,
        ];
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, surface, &required_extensions);
        let (device, queue) = create_device(
            &instance,
            physical_device,
            queue_family_index,
            &required_extensions,
        );

        let window_size = window.inner_size();

        let memory_allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let swapchain_image_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let image_format =
            SwapchainResources::choose_swapchain_format(&swapchain_image_formats).unwrap();

        let render_pass = create_renderpass(&device, image_format.format);
        let shared_allocator = Arc::new(Mutex::new(memory_allocator));

        let pipelines = create_builtin_graphics_pipelines(&device, render_pass);
        let pool_create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
            ..Default::default()
        };
        let one_time_command_pool =
            unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: one_time_command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };
        let one_time_submit = unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] };
        let resource_manager = ResourceManager::init(
            &device,
            shared_allocator.clone(),
            one_time_command_pool,
            one_time_submit,
            queue,
            queue_family_index,
        );

        let swapchain = SwapchainResources::create(
            &surface_loader,
            &swapchain_loader,
            &device,
            physical_device,
            render_pass,
            None,
            image_format,
            window_size,
            shared_allocator.clone(),
            surface,
            //HACK: i need to find some other way than just having this a explicit number
            pipelines[2].descriptor_set_layouts[0],
            &[queue_family_index],
        );

        eprintln!("creating sync objects");

        let mut per_frame = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            per_frame.push(PerFrame::create(&device, queue_family_index));
        }

        Renderer {
            pipelines,
            one_time_submit,
            one_time_submit_pool: one_time_command_pool,
            _entry: entry,
            debug_messenger,
            debug_utils_loader,
            instance,
            framebuffer_resized: false,
            current_frame: 0,
            frame_counter: 0,
            swapchain,
            render_pass,
            allocator: shared_allocator,
            surface,
            surface_loader,
            swapchain_loader: swapchain_loader.clone(),
            device,
            queue,
            start_time,
            window: window.clone(),
            queue_family_index,
            physical_device,
            per_frame,
            swapchain_image_index: 0,
            resource_manager,
        }
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
