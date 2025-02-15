use std::{f32::consts::FRAC_PI_4, time::Instant};
use ultraviolet::{projection, Mat4, Vec3};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    pipeline::{Pipeline, PipelineBindPoint},
    swapchain::{acquire_next_image, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::WindowId,
};

mod app;
mod models;
mod renderer;
mod resources;

pub const FRAMES_IN_FLIGHT: usize = 2;
#[derive(Default)]
pub struct App {
    render_context: Option<renderer::Context>,
    app_context: Option<app::Context>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_context.is_some() || self.render_context.is_some() {
            eprintln!("resumed called while app is already some");
            return;
        }
        self.app_context = Some(app::Context::init(event_loop));
        let acx = self.app_context.as_ref().unwrap();
        self.render_context = Some(renderer::Context::init(
            &acx.device,
            &acx.window,
            &acx.surface,
        ));
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // INFO: RCX = render_context, acx is the app context
        let rcx = self.render_context.as_mut().unwrap();
        let acx = self.app_context.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => rcx.recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                println!("frame start");
                acx.window.request_redraw();
                let window_size = acx.window.inner_size();
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }
                rcx.frame_counter += 1;
                rcx.frames_resources_free[rcx.current_frame]
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();
                dbg!(rcx.frame_counter);

                // NOTE: RENDERING START
                if rcx.recreate_swapchain {
                    // Use the new dimensions of the window.
                    dbg!(rcx.recreate_swapchain);
                    let new_images;
                    (rcx.swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to create swapchain");
                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    (rcx.framebuffers, rcx.color_buffers, rcx.normal_buffers) =
                        renderer::create_framebuffers(
                            &new_images,
                            &rcx.render_pass,
                            rcx.memory_allocator.clone(),
                        );

                    (rcx.deferred_sets, rcx.lighting_sets) = renderer::create_descriptor_sets(
                        &acx.device,
                        &rcx.deferred_pipeline,
                        &rcx.lighting_pipeline,
                        &rcx.uniform_buffers,
                        &rcx.color_buffers,
                        &rcx.normal_buffers,
                        new_images.len(),
                    );

                    rcx.viewport.extent = window_size.into();
                    acx.aspect_ratio = window_size.width as f32 / window_size.height as f32;

                    rcx.recreate_swapchain = false;
                }

                dbg!(rcx.current_frame);

                // write needs to be dropped to free the lock on the uniform buffer

                let (swapchain_image_index, is_suboptimal, acquire_future) =
                    match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            rcx.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if is_suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut write = rcx.uniform_buffers[swapchain_image_index as usize]
                    .write()
                    .unwrap();
                *write = App::calculate_current_transform(acx.start_time, acx.aspect_ratio);
                drop(write);

                dbg!(is_suboptimal);
                dbg!(swapchain_image_index);

                let mut builder = AutoCommandBufferBuilder::primary(
                    &acx.command_buffer_allocator,
                    acx.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // Before we can draw, we have to *enter a render pass*.
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // others should use none as their value
                            clear_values: vec![
                                Some([1.0, 0.0, 0.0, 1.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some(1.0.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[swapchain_image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass. This can be either
                            // `Inline` or `SecondaryCommandBuffers`
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.deferred_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.deferred_pipeline.layout().clone(),
                        0,
                        rcx.deferred_sets[swapchain_image_index as usize].clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, acx.vertex_buffer.clone())
                    .unwrap()
                    .draw(acx.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .next_subpass(
                        SubpassEndInfo::default(),
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(rcx.lighting_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.lighting_pipeline.layout().clone(),
                        0,
                        rcx.lighting_sets[swapchain_image_index as usize].clone(),
                    )
                    .unwrap()
                    .draw(acx.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();
                let future = rcx.frames_resources_free[rcx.current_frame]
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(acx.queue.clone(), command_buffer)
                    .unwrap()
                    // dosent present imediately but submits a present command to
                    // the queue, so the triangle will finish rendering
                    .then_swapchain_present(
                        acx.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            swapchain_image_index,
                        ),
                    )
                    .then_signal_fence_and_flush(); // signal will tell gpu to finish and use fence

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.frames_resources_free[rcx.current_frame] = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.frames_resources_free[rcx.current_frame] =
                            Some(sync::now(acx.device.clone()).boxed());
                    }
                    Err(e) => {
                        rcx.frames_resources_free[rcx.current_frame] =
                            Some(Box::new(sync::now(acx.device.clone())).boxed());
                        println!("Failed to flush future: {:?}", e);
                    }
                }
                rcx.current_frame = (rcx.current_frame + 1) % FRAMES_IN_FLIGHT;
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let app_context = self.app_context.as_mut().unwrap();
        app_context.window.request_redraw();
    }
}
impl App {
    fn calculate_current_transform(
        start_time: Instant,
        aspect_ratio: f32,
    ) -> resources::TransformationUBO {
        let move_back = Mat4::from_translation(Vec3 {
            x: 0.0,
            y: 0.0,
            z: -5.0,
        });
        let rotation = Mat4::from_euler_angles(
            0.0,
            (start_time.elapsed().as_secs_f32() * 0.5) % 360.0,
            (start_time.elapsed().as_secs_f32() * 0.3) % 360.0,
        );

        resources::TransformationUBO {
            model: move_back * rotation,
            view: ultraviolet::Mat4::look_at(
                Vec3::new(0.0, 0.0, 3.0), // Move camera back slightly
                Vec3::new(0.0, 0.0, 0.0), // Look at origin
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            proj: projection::perspective_vk(FRAC_PI_4, aspect_ratio, 0.001, 100.0),
        }
    }
}
