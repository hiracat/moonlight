use obj::load_obj;
use renderer::{
    AmbientLight, Camera, DirectionalLight, Model, PointLight, Renderer, Scene, Vertex,
};
use std::{
    collections::HashSet,
    fs::File,
    io::BufReader,
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};
use ultraviolet::{Mat4, Rotor3, Vec3, Vec4};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

mod renderer;

pub struct App {
    renderer: Option<Renderer>,
    scene: Scene,
    keys: HashSet<KeyCode>,
    prev_frame_end: Instant,
    delta_time: Duration,
}
impl Default for App {
    fn default() -> Self {
        App {
            renderer: None,
            scene: Scene::default(),
            keys: HashSet::new(),
            prev_frame_end: Instant::now(),
            delta_time: Duration::default(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            eprintln!("resumed called while renderer is already some");
            return;
        }

        let fox = {
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

            Model::new(vertices, indices, Vec4::zero())
        };

        let ground = {
            let input = BufReader::new(File::open("data/models/groundplane.obj").unwrap());
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

            Model::new(vertices, indices, Vec4::zero())
        };

        // let sun = DirectionalLight::new([2.0, 10.0, 0.0, 1.0], [0.0, 0.0, 0.0]);

        let ambient = AmbientLight::new([0.9, 0.9, 1.0], 0.0);

        let window = renderer::create_window(event_loop);

        let red = PointLight::new([2.0, 2.0, 0.0, 1.0], [1.0, 0.0, 0.0], None, None, None);
        let green = PointLight::new([-2.0, 2.0, 0.0, 1.0], [0.0, 1.0, 0.0], None, None, None);
        let blue = PointLight::new([0.0, 2.0, -3.0, 1.0], [0.0, 0.0, 1.0], None, None, None);

        self.scene = Scene {
            camera: Camera::new(
                Vec3::new(0.0, 2.0, 6.0),
                Vec3::new(0.0, 1.0, 0.0),
                60.0,
                1.0,
                100.0,
                &window,
            ),
            ambient,
            points: vec![red, green, blue],
            directionals: vec![], //sun],
            models: vec![fox, ground],
        };
        self.renderer = Some(Renderer::init(&event_loop, &mut self.scene, &window));
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
                let mut velocity = Vec4::zero();
                let mut rotation = Rotor3::identity();

                let mut rotation_amount = 1.0;
                rotation_amount *= self.delta_time.as_secs_f32();
                if self.keys.contains(&KeyCode::KeyW) {
                    velocity += Vec4::new(0.0, 0.0, 1.0, 0.0)
                }
                if self.keys.contains(&KeyCode::KeyS) {
                    velocity += Vec4::new(0.0, 0.0, -1.0, 0.0)
                }

                if self.keys.contains(&KeyCode::KeyD) {
                    rotation = rotation * Rotor3::from_rotation_xz(rotation_amount)
                }
                if self.keys.contains(&KeyCode::KeyA) {
                    rotation = rotation * Rotor3::from_rotation_xz(-rotation_amount)
                }

                self.scene.models[0].rotation = rotation * self.scene.models[0].rotation;

                velocity *= self.delta_time.as_secs_f32();
                velocity =
                    (self.scene.models[0].rotation * velocity.xyz()).into_homogeneous_vector();

                self.scene.models[0].position += velocity;
                self.scene.models[0].requires_update = true;

                self.renderer.as_mut().unwrap().draw(&mut self.scene);

                self.delta_time = self.prev_frame_end.elapsed();
                self.prev_frame_end = Instant::now();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            self.keys.insert(code);
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
                if event.state == ElementState::Released {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            self.keys.remove(&code);
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.renderer.as_mut().unwrap().window.request_redraw();
    }
}
