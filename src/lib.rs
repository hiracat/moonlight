use obj::load_obj;
use renderer::{AmbientLight, Camera, Model, Renderer, Scene, Vertex};
use std::{fs::File, io::BufReader, sync::Arc, time::Instant};
use ultraviolet::Mat4;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::WindowId,
};

mod renderer;

#[derive(Default)]
pub struct App {
    renderer: Option<Renderer>,
    scene: Scene,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            eprintln!("resumed called while renderer is already some");
            return;
        }
        let input = BufReader::new(File::open("data/models/low poly fox.obj").unwrap());
        let model = load_obj::<obj::Vertex, _, u32>(input).unwrap();

        let vertices: Vec<Vertex> = model
            .vertices
            .iter()
            .map(|v| Vertex {
                position: v.position,
                normal: v.normal,
                color: [1.0, 1.0, 1.0], // map other fields as needed
            })
            .collect();
        let indices: Vec<u32> = model.indices.iter().map(|i| *i as u32).collect();

        let model = Model {
            vertices,
            indices,
            requires_update: true,
            model: App::calculate_current_transform(Instant::now()),

            ..Default::default()
        };

        self.scene = Scene {
            camera: Camera {
                view: Mat4::identity(),
                proj: Mat4::identity(),
            },
            ambient: AmbientLight {
                color: [1.0, 1.0, 1.0],
                intensity: 0.1,
            },
            lights: vec![],
            models: vec![model],
        };
        self.renderer = Some(Renderer::init(&event_loop, &mut self.scene));
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // INFO: RCX = render_context, acx is the app context

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => self.renderer.as_mut().unwrap().recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                println!("frame start");
                self.renderer.as_mut().unwrap().draw(&mut self.scene);
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.renderer.as_mut().unwrap().window.request_redraw();
    }
}
impl App {
    fn calculate_current_transform(start_time: Instant) -> Mat4 {
        let rotation = Mat4::from_euler_angles(
            0.0,
            // (start_time.elapsed().as_secs_f32() * 0.5) % 360.0,
            0.0,
            (start_time.elapsed().as_secs_f32() * 0.3) % 360.0,
        );
        rotation
    }
}
