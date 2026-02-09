use std::time::Instant;

use ash::vk;
use winit::event_loop::ActiveEventLoop;

use crate::{
    ecs::World,
    renderer::{
        draw::{UIRenderer, WorldRenderer, FRAMES_IN_FLIGHT},
        resources::ResourceManager,
        swapchain::SwapchainResources,
    },
    vulkan::VulkanContext,
};

trait Game {
    fn on_start(&mut self, engine: &mut Engine);
    fn on_update(&mut self, engine: &mut Engine, delta_time: f32);
    fn on_close(&mut self, engine: &mut Engine);
}

struct Engine {
    vulkan_context: VulkanContext,
    world_renderer: WorldRenderer,
    ui_renderer: UIRenderer,

    resource_manager: ResourceManager,

    swapchain: SwapchainResources,
    per_frame: Vec<PerFrame>,
    swapchain_image_index: usize,
    current_frame: usize,

    world: World,

    prev_frame_end: Instant,
    delta_time: f32,
    frame_count: u64,
}

impl Engine {
    fn init(event_loop: &ActiveEventLoop) -> Self {
        let context = VulkanContext::init(event_loop);
        let swapchain = SwapchainResources::create(&context);
        let world_renderer = WorldRenderer::init(&context);
        let ui_renderer = UIRenderer::init(&context);
        let resource_manager = ResourceManager::init(&context);
        let mut per_frame = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            per_frame.push(PerFrame::create(&context));
        }
        Self {
            vulkan_context: context,
            world_renderer: world_renderer,
            ui_renderer: ui_renderer,
            swapchain: swapchain,
            resource_manager: resource_manager,
            world: World::init(),
            prev_frame_end: Instant::now(),
            delta_time: 0.0,
            current_frame: 0,
            per_frame: per_frame,
            frame_count: 0,
            swapchain_image_index: 0,
        }
    }
    fn draw(&mut self) {
        let full_output = self.ui.full_output.take().unwrap();
        self.ui
            .winit_egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        self.ui
            .handle_new_textures(&self.device, &full_output.textures_delta);
        let geometry = self
            .ui
            .ui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let window_size = self.vulkan_context.window.inner_size();

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
            self.ui.recreate_framebuffers(
                &self.device,
                &self.swapchain.framebuffers.swapchain_image_views,
                vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
            );
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

        let gui_clear_values = vec![];

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size.width as f32,
            height: window_size.height as f32,
            max_depth: 1.0,
            min_depth: 0.0,
        };

        unsafe {
            self.device.cmd_begin_render_pass(
                frame.command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.ui.renderpass,
                    framebuffer: self.ui.framebuffers[self.swapchain_image_index as usize],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: window_size.width,
                            height: window_size.height,
                        },
                    },
                    p_clear_values: gui_clear_values.as_ptr(),
                    clear_value_count: gui_clear_values.len() as u32,
                    ..Default::default()
                },
                vk::SubpassContents::INLINE,
            );
            self.device
                .cmd_set_viewport(frame.command_buffer, 0, &[viewport]);
            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.ui.pipeline,
            );
        }
        self.ui.draw_meshes(
            &geometry,
            &self.device,
            frame.command_buffer,
            full_output.pixels_per_point,
            window_size,
        );

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
            self.ui.recreate_framebuffers(
                &self.device,
                &self.swapchain.framebuffers.swapchain_image_views,
                vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
            );
            world.get_mut_resource::<Camera>().unwrap().aspect_ratio =
                window_size.width as f32 / window_size.height as f32;
        }

        self.ui.cleanup_old_textures(full_output.textures_delta);
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }
    fn run(&mut self, game: Box<dyn Game>) {
        todo!()
    }
}
pub struct PerFrame {
    pub command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    // wait for the frame that was FRAMES_IN_FLIGHT frames ago, but has the same current_frame
    // since modulo
    in_flight: vk::Fence,

    transient_pool: vk::DescriptorPool,
}

impl PerFrame {
    fn create(context: &VulkanContext) -> Self {
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
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 50,
            },
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
