use ecs::World;
use obj::load_obj;
use renderer::{AmbientLight, Camera, DirectionalLight, Model, PointLight, Renderer, Vertex};
use std::{
    collections::HashSet,
    fs::File,
    io::BufReader,
    time::{Duration, Instant},
};
use ultraviolet::{Rotor3, Vec3, Vec4};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

mod ecs;
mod renderer;

type Keyboard = HashSet<KeyCode>;
#[derive(Debug)]
struct Controllable;

pub struct App {
    renderer: Option<Renderer>,
    world: World,
    prev_frame_end: Instant,
    delta_time: Duration,
}
impl Default for App {
    fn default() -> Self {
        App {
            renderer: None,
            world: World::init(),
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
        let window = renderer::create_window(event_loop);

        let fox = self.world.entity_create();
        let ground = self.world.entity_create();

        let red = self.world.entity_create();
        let blue = self.world.entity_create();
        let green = self.world.entity_create();

        let _ = self.world.component_add(fox, {
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
            let indices: Vec<u32> = model.indices.clone();

            Model::new(vertices, indices, Vec4::zero())
        });
        let _ = self.world.component_add(fox, Controllable);
        let _ = self.world.component_add(ground, {
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
            let indices: Vec<u32> = model.indices.clone();

            Model::new(vertices, indices, Vec4::zero())
        });

        let _ = self.world.component_add(
            red,
            PointLight::new([2.0, 2.0, 0.0, 1.0], [1.0, 0.0, 0.0], None, None, None),
        );
        let _ = self.world.component_add(
            green,
            PointLight::new([-2.0, 2.0, 0.0, 1.0], [0.0, 1.0, 0.0], None, None, None),
        );
        let _ = self.world.component_add(
            blue,
            PointLight::new([0.0, 2.0, -3.0, 1.0], [0.0, 0.0, 1.0], None, None, None),
        );

        let camera = Camera::new(
            Vec3::new(0.0, 2.0, 6.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            1.0,
            100.0,
            &window,
        );
        let sun = DirectionalLight::new([2.0, 10.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.05);

        self.world.resource_add(camera);
        self.world.resource_add(sun);
        self.world.resource_add(ambient);

        self.world.resource_add(Keyboard::new());

        self.renderer = Some(Renderer::init(event_loop, &mut self.world, &window));
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => self.renderer.as_mut().unwrap().recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                println!("frame start");

                wasd_update(&mut self.world, self.delta_time);
                self.renderer.as_mut().unwrap().draw(&mut self.world);
                self.delta_time = self.prev_frame_end.elapsed();
                self.prev_frame_end = Instant::now();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let keyboard = self
                    .world
                    .resource_get_mut::<Keyboard>()
                    .expect("keyboard should have been added");
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            keyboard.insert(code);
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
                if event.state == ElementState::Released {
                    match event.physical_key {
                        PhysicalKey::Code(code) => {
                            keyboard.remove(&code);
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

fn wasd_update(world: &mut World, delta_time: Duration) {
    let mut velocity = Vec4::zero();
    let mut rotation = Rotor3::identity();
    let mut rotation_amount = 5.0;
    rotation_amount *= delta_time.as_secs_f32();
    let keyboard = world
        .resource_get::<Keyboard>()
        .expect("keyboard should have been added during resumed");
    if keyboard.contains(&KeyCode::KeyW) {
        velocity += Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
    if keyboard.contains(&KeyCode::KeyS) {
        velocity += Vec4::new(0.0, 0.0, -1.0, 0.0);
    }
    if keyboard.contains(&KeyCode::KeyD) {
        rotation = rotation * Rotor3::from_rotation_xz(rotation_amount);
    }
    if keyboard.contains(&KeyCode::KeyA) {
        rotation = rotation * Rotor3::from_rotation_xz(-rotation_amount);
    }
    let entities = world.query::<Controllable>();
    dbg!(&entities);
    let player = entities[0].0;
    dbg!(player);
    let player = world
        .component_get_mut::<Model>(player)
        .expect("player should have model component");

    player.rotation = rotation * player.rotation;
    velocity *= 5.0;
    velocity *= delta_time.as_secs_f32();
    velocity = (player.rotation * velocity.xyz()).into_homogeneous_vector();
    player.position += velocity;
    player.requires_update = true;
}
