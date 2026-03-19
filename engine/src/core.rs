use std::{collections::HashSet, ptr, sync::Arc, time::Instant};

use ash::vk;
use image::{ImageBuffer, Luma, Rgb, Rgb32FImage};
use ultraviolet::Vec3;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

use crate::{
    components::{Camera, Time},
    ecs::World,
    lua::{LuaVM, PrintOnError},
    renderers::{
        ui::UIRenderer,
        world::{
            draw::{WorldRenderer, FRAMES_IN_FLIGHT},
            swapchain::{create_semaphores, SwapchainResources},
        },
    },
    resources::{ResourceManager, Texture},
    vulkan::VulkanContext,
};
#[derive(Debug, Copy, Clone)]
pub struct Controllable;

pub struct Engine {
    world_renderer: WorldRenderer,
    ui_renderer: UIRenderer,

    swapchain: SwapchainResources,
    render_finished: Vec<vk::Semaphore>,
    per_frame: Vec<PerFrame>,
    swapchain_image_index: usize,
    frame_in_flight: usize,
    swapchain_resized: bool,

    pub resource_manager: ResourceManager,

    pub prev_frame_end: Instant,
    pub frame_count: u64,
    pub window_size: (u32, u32),

    vulkan_context: VulkanContext,
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.vulkan_context.device.device_wait_idle().unwrap();
        }

        // renderers first (framebuffers, pipelines, etc)
        // world_renderer and ui_renderer drop here via field order, but we need
        // semaphores gone before vulkan_context, so be explicit

        for semaphore in &self.render_finished {
            unsafe {
                self.vulkan_context
                    .device
                    .destroy_semaphore(*semaphore, None);
            }
        }

        // per_frame drops itself (PerFrame::drop handles semaphores, fences, pools)
        // swapchain drops itself (SwapchainResources::drop handles imageviews, swapchain)
        // world_renderer drops itself
        // ui_renderer drops itself
        // resource_manager drops itself

        // vulkan_context must be last — device, instance, surface all live here
    }
}

impl Engine {
    fn init(event_loop: &ActiveEventLoop) -> Self {
        let context = VulkanContext::init(event_loop);

        let swapchain = SwapchainResources::create(&context, None);
        let world_renderer = WorldRenderer::init(&context, &swapchain);
        let ui_renderer = UIRenderer::init(&context, &swapchain);
        let resource_manager = ResourceManager::init(&context);
        let mut per_frame = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            per_frame.push(PerFrame::create(&context));
        }
        let render_finished = create_semaphores(&context.device, swapchain.swapchain_images.len());
        let size = context.window.inner_size();
        let window_size = (size.width, size.height);

        Self {
            vulkan_context: context,
            world_renderer: world_renderer,
            ui_renderer: ui_renderer,
            swapchain: swapchain,
            render_finished: render_finished,
            resource_manager: resource_manager,
            prev_frame_end: Instant::now(),
            frame_in_flight: 0,
            per_frame: per_frame,
            frame_count: 0,
            swapchain_image_index: 0,
            swapchain_resized: false,
            window_size: window_size,
        }
    }
    fn draw(&mut self, world: &mut World) {
        let full_output = self.ui_renderer.full_output.take().unwrap();
        self.ui_renderer
            .winit_egui_state
            .handle_platform_output(&self.vulkan_context.window, full_output.platform_output);
        self.ui_renderer
            .handle_new_textures(&self.vulkan_context, &full_output.textures_delta);
        let geometry = self
            .ui_renderer
            .ui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let window_size_w = self.vulkan_context.window.inner_size();
        let window_size_v = vk::Extent2D {
            width: window_size_w.width,
            height: window_size_w.height,
        };

        if window_size_w.width == 0 || window_size_w.height == 0 {
            return;
        }

        let frame = &self.per_frame[self.frame_in_flight];

        unsafe {
            self.vulkan_context
                .device
                .wait_for_fences(&[frame.in_flight], true, u64::MAX)
                .unwrap();
        };

        self.frame_count += 1;

        if self.swapchain_resized {
            unsafe {
                self.vulkan_context.device.device_wait_idle().unwrap();
            }
            // Use the new dimensions of the window.
            self.swapchain =
                SwapchainResources::create(&self.vulkan_context, Some(self.swapchain.swapchain));
            self.world_renderer
                .update_swapchain_resources(&self.vulkan_context, &self.swapchain);
        }

        let is_suboptimal;
        (self.swapchain_image_index, is_suboptimal) = unsafe {
            match self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            ) {
                Ok((index, suboptimal)) => (index as usize, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_resized = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };
        if is_suboptimal {
            self.swapchain_resized = true;
        }

        // NOTE: RENDERING START

        unsafe {
            self.vulkan_context
                .device
                .reset_fences(&[frame.in_flight])
                .unwrap()
        };
        unsafe {
            self.vulkan_context
                .device
                .reset_descriptor_pool(frame.transient_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }

        unsafe {
            self.vulkan_context
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }

        unsafe {
            self.vulkan_context
                .device
                .begin_command_buffer(
                    frame.command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let mut swapchain_barrier = vk::ImageMemoryBarrier2 {
            image: self.swapchain.swapchain_images[self.swapchain_image_index],
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
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
        };
        unsafe {
            self.vulkan_context.device.cmd_pipeline_barrier2(
                frame.command_buffer,
                &vk::DependencyInfo {
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                    image_memory_barrier_count: 1,
                    p_image_memory_barriers: &mut swapchain_barrier as *mut _,
                    ..Default::default()
                },
            );
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size_v.width as f32,
            height: window_size_v.height as f32,
            max_depth: 1.0,
            min_depth: 0.0,
        };
        self.world_renderer.record_commands(
            world,
            &self.vulkan_context.device,
            frame.command_buffer,
            frame.transient_pool,
            window_size_v,
            &mut self.resource_manager,
            self.swapchain_image_index,
            self.swapchain.swapchain_image_views[self.swapchain_image_index],
        );

        let swapchain_image_view = self.swapchain.swapchain_image_views[self.swapchain_image_index];

        let ui_attachment = vk::RenderingAttachmentInfo {
            image_view: swapchain_image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::LOAD, // LOAD not CLEAR, UI draws on top of world
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        };

        unsafe {
            self.vulkan_context.device.cmd_begin_rendering(
                frame.command_buffer,
                &vk::RenderingInfo {
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: window_size_v,
                    },
                    layer_count: 1,
                    color_attachment_count: 1,
                    p_color_attachments: &ui_attachment,
                    p_depth_attachment: ptr::null(),
                    p_stencil_attachment: ptr::null(),
                    ..Default::default()
                },
            );
            self.vulkan_context
                .device
                .cmd_set_viewport(frame.command_buffer, 0, &[viewport]);
            self.vulkan_context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.ui_renderer.pipeline,
            );
        }

        self.ui_renderer.draw_meshes(
            &geometry,
            &self.vulkan_context.device,
            frame.command_buffer,
            full_output.pixels_per_point,
            window_size_w,
        );

        unsafe {
            self.vulkan_context
                .device
                .cmd_end_rendering(frame.command_buffer);
        }
        let mut swapchain_barrier = vk::ImageMemoryBarrier2 {
            image: self.swapchain.swapchain_images[self.swapchain_image_index],
            // you need the vk::Image handle here, not just the view
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access_mask: vk::AccessFlags2::empty(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                base_mip_level: 0,
                base_array_layer: 0,
            },
            ..Default::default()
        };

        unsafe {
            self.vulkan_context.device.cmd_pipeline_barrier2(
                frame.command_buffer,
                &vk::DependencyInfo {
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                    p_image_memory_barriers: &mut swapchain_barrier as *mut _,
                    image_memory_barrier_count: 1,
                    ..Default::default()
                },
            );
            self.vulkan_context
                .device
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
            p_signal_semaphores: &self.render_finished[self.swapchain_image_index as usize],
            signal_semaphore_count: 1,
            ..Default::default()
        };

        unsafe {
            self.vulkan_context
                .device
                .queue_submit(
                    self.vulkan_context.queue,
                    &[queue_submit_info],
                    frame.in_flight,
                )
                .unwrap();
        }

        let mut present_results: [vk::Result; 1] = [Default::default(); 1];
        let present_info = vk::PresentInfoKHR {
            p_wait_semaphores: &self.render_finished[self.swapchain_image_index as usize],
            wait_semaphore_count: 1,
            p_swapchains: &self.swapchain.swapchain,
            swapchain_count: 1,
            p_image_indices: &(self.swapchain_image_index as u32),
            p_results: present_results.as_mut_ptr(),
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain
                .swapchain_loader
                .queue_present(self.vulkan_context.queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.swapchain_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.swapchain_resized = false;
            unsafe {
                self.vulkan_context.device.device_wait_idle().unwrap();
            }
            self.swapchain =
                SwapchainResources::create(&self.vulkan_context, Some(self.swapchain.swapchain));
            self.world_renderer
                .update_swapchain_resources(&self.vulkan_context, &self.swapchain);
        }

        self.ui_renderer
            .cleanup_old_textures(full_output.textures_delta);
        self.frame_in_flight = (self.frame_in_flight + 1) % FRAMES_IN_FLIGHT;
        self.frame_count += 1;
    }
}
pub struct PerFrame {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub image_available: vk::Semaphore,
    // wait for the frame that was FRAMES_IN_FLIGHT frames ago, but has the same current_frame
    // since modulo
    pub in_flight: vk::Fence,

    pub transient_pool: vk::DescriptorPool,
    device: Arc<ash::Device>,
}
impl Drop for PerFrame {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_fence(self.in_flight, None);
            self.device
                .destroy_descriptor_pool(self.transient_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl PerFrame {
    fn create(context: &VulkanContext) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: context.queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe {
            context
                .device
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
            context
                .device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()
                .first()
                .copied()
                .unwrap()
        };
        let in_flight = unsafe {
            context.device.create_fence(
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
                descriptor_count: 2000,
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
            context
                .device
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
            context
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        Self {
            device: context.device.clone(),
            command_pool,
            command_buffer,
            in_flight,
            transient_pool,
            image_available,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MouseState {
    pub x: f32,
    pub y: f32,
    pub locked: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Keyboard {
    keys: HashSet<KeyCode>,
}
impl Keyboard {
    pub fn add(&mut self, key: KeyCode) {
        self.keys.insert(key);
    }
    pub fn remove(&mut self, key: KeyCode) {
        self.keys.remove(&key);
    }
    pub fn is_down(&self, key: KeyCode) -> bool {
        return self.keys.contains(&key);
    }
}
#[derive(Default)]
pub struct Game {
    pub on_start: Vec<System>,
    pub on_update: Vec<System>,
    pub on_close: Vec<System>,
    pub on_ui: Vec<fn(&mut World, &egui::Context)>,
}

pub enum System {
    Rust(fn(&mut World, &mut Engine)),
    Lua(String),
}

pub struct App {
    pub game: Game,
    engine: Option<Engine>,
    lua: LuaVM,
    world: World,
}

impl App {
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().expect("failed to init event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(self).unwrap();
    }

    pub fn new(lua_path: &str) -> Self {
        App {
            engine: None,
            game: Game::default(),
            world: World::init(),
            lua: LuaVM::new(lua_path),
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::too_many_lines)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // ───────────────────────────────────────────────────────
        // 1) Window & input setup
        // ───────────────────────────────────────────────────────
        if self.engine.is_some() {
            return;
        }
        //HACK: magic number, i dont care right now

        self.engine = Some(Engine::init(event_loop));

        let window_size = self
            .engine
            .as_ref()
            .unwrap()
            .vulkan_context
            .window
            .inner_size();

        // Keyboard + mouse input resources
        let keyboard = Keyboard::default();
        let mouse_movement = MouseState::default();
        let camera = Camera::create(
            Vec3::new(0.0, 5.0, -6.0),
            60.0,
            1.0,
            200.0,
            window_size.height as f32 / window_size.width as f32,
        );
        let time = Time::default();
        self.world.add_resource(camera).unwrap();
        self.world.add_resource(keyboard).unwrap();
        self.world.add_resource(mouse_movement).unwrap();
        self.world.add_resource(time).unwrap();

        for sys in &self.game.on_start {
            match sys {
                System::Lua(l) => {
                    let err =
                        self.lua
                            .run_script(&mut self.world, self.engine.as_mut().unwrap(), l);
                    err.print_on_error();
                }
                System::Rust(r) => r(&mut self.world, self.engine.as_mut().unwrap()),
            }
        }

        self.engine.as_mut().unwrap().prev_frame_end = Instant::now();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let window = self.engine.as_mut().unwrap().vulkan_context.window.clone();

        let response = self
            .engine
            .as_mut()
            .unwrap()
            .ui_renderer
            .winit_egui_state
            .on_window_event(&window, &event);

        if response.consumed == true {
            return;
        }
        if response.repaint == true {
            self.engine
                .as_ref()
                .unwrap()
                .vulkan_context
                .window
                .request_redraw();
        }
        let world = &mut self.world;

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                for sys in &self.game.on_update {
                    match sys {
                        System::Lua(l) => {
                            let err = self.lua.run_script(
                                &mut self.world,
                                self.engine.as_mut().unwrap(),
                                l,
                            );
                            err.print_on_error();
                        }
                        System::Rust(r) => r(&mut self.world, self.engine.as_mut().unwrap()),
                    }
                }

                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                let engine = self.engine.as_mut().unwrap();
                engine.swapchain_resized = true;
                let PhysicalSize { width, height } = window.inner_size();
                engine.window_size = (width, height);
                world.get_mut_resource::<Camera>().unwrap().aspect_ratio =
                    width as f32 / height as f32;
            }
            WindowEvent::RedrawRequested => {
                for sys in &self.game.on_update {
                    match sys {
                        System::Lua(l) => {
                            let err = self.lua.run_script(
                                &mut self.world,
                                self.engine.as_mut().unwrap(),
                                l,
                            );
                            err.print_on_error();
                        }
                        System::Rust(r) => r(&mut self.world, self.engine.as_mut().unwrap()),
                    }
                }
                let engine = self.engine.as_mut().unwrap();

                let raw_input = engine.ui_renderer.winit_egui_state.take_egui_input(&window);
                let full_output = engine.ui_renderer.ui_ctx.run(raw_input, |ctx| {
                    for sys in &self.game.on_ui {
                        sys(&mut self.world, ctx);
                    }
                });

                engine.ui_renderer.full_output = Some(full_output);
                engine.draw(&mut self.world);

                self.world.get_mut_resource::<Time>().unwrap().delta_time =
                    engine.prev_frame_end.elapsed().as_secs_f32();
                engine.prev_frame_end = Instant::now();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let mut release_mouse = false;
                let keyboard = self
                    .world
                    .get_mut_resource::<Keyboard>()
                    .expect("keyboard should have been added");
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            keyboard.add(code);
                            if code == KeyCode::Escape {
                                window
                                    .set_cursor_grab(winit::window::CursorGrabMode::None)
                                    .unwrap();
                                window.set_cursor_visible(true);
                                release_mouse = true;
                            }
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
                if event.state == ElementState::Released {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            keyboard.remove(code);
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
                if release_mouse {
                    self.world.get_mut_resource::<MouseState>().unwrap().locked = false;
                }
            }
            WindowEvent::MouseInput { .. } => {
                window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .unwrap();
                window.set_cursor_visible(false);
                self.world.get_mut_resource::<MouseState>().unwrap().locked = true;
            }
            WindowEvent::Focused(focused) => {
                if focused {
                    let engine = self.engine.as_mut().unwrap();
                    self.world.get_mut_resource::<Time>().unwrap().delta_time = 0.0;
                    engine.prev_frame_end = Instant::now();
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.engine
            .as_mut()
            .unwrap()
            .vulkan_context
            .window
            .request_redraw();
    }

    #[allow(unused_variables)]
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        #[allow(clippy::cast_possible_truncation)]
        if let DeviceEvent::MouseMotion { delta } = event {
            let state = self.world.get_mut_resource::<MouseState>().unwrap();
            if state.locked {
                state.x = delta.0 as f32;
                state.y = delta.1 as f32;
            }
        }
    }
}

pub struct TerrainMap {
    pub size: f32,
    pub height: f32,
    pub resolution: u32,
    pub map: Texture,
    pub cpu_map: ImageBuffer<Luma<f32>, Vec<f32>>,
}
impl TerrainMap {
    pub fn get_height_at(&self, x: f32, z: f32) -> f32 {
        let w = self.cpu_map.width() as f32;
        let h = self.cpu_map.height() as f32;

        // map world position to pixel coordinates
        // x/z are world coords, remap to 0..resolution
        let px = ((x / self.size) + 0.5) * w;
        let pz = ((z / self.size) + 0.5) * h;

        // clamp to image bounds
        let px = px.clamp(0.0, w - 1.0);
        let pz = pz.clamp(0.0, h - 1.0);

        // the 4 surrounding pixels
        let x0 = px.floor() as u32;
        let x1 = (px.ceil() as u32).min(self.cpu_map.width() - 1);
        let z0 = pz.floor() as u32;
        let z1 = (pz.ceil() as u32).min(self.cpu_map.height() - 1);

        // fractional part for interpolation
        let tx = px.fract();
        let tz = pz.fract();

        // sample red channel from all 4 corners
        let h00 = self.cpu_map.get_pixel(x0, z0).0[0];
        let h10 = self.cpu_map.get_pixel(x1, z0).0[0];
        let h01 = self.cpu_map.get_pixel(x0, z1).0[0];
        let h11 = self.cpu_map.get_pixel(x1, z1).0[0];

        // bilinear interpolation
        let top = h00 * (1.0 - tx) + h10 * tx;
        let bot = h01 * (1.0 - tx) + h11 * tx;
        let t = top * (1.0 - tz) + bot * tz;

        t * self.height
    }
}
