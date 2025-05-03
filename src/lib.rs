use ecs::World;
use obj::load_obj;
use renderer::{AmbientLight, Camera, DirectionalLight, Model, PointLight, Renderer, Vertex};
use std::{
    collections::HashSet,
    f32::consts::PI,
    fs::File,
    io::BufReader,
    time::{Duration, Instant},
};
use ultraviolet::{Rotor3, Slerp, Vec3, Vec4};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

mod ecs;
mod renderer;

type Keyboard = HashSet<KeyCode>;
#[derive(Debug)]
struct Controllable;
#[derive(Debug, Default, Clone, Copy)]
struct MouseMovement {
    x: f32,
    y: f32,
}

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
    #[allow(clippy::too_many_lines)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            eprintln!("resumed called while renderer is already some");
            return;
        }
        let window = renderer::create_window(event_loop);
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        window.set_cursor_visible(false);

        let fox = self.world.entity_create();
        let ground = self.world.entity_create();

        let red = self.world.entity_create();
        let blue = self.world.entity_create();
        let green = self.world.entity_create();

        let x = self.world.entity_create();
        let y = self.world.entity_create();
        let z = self.world.entity_create();

        let _ = self.world.component_add(x, {
            let input = BufReader::new(File::open("data/models/square.obj").unwrap());
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

            Model::new(vertices, indices, Vec4::new(1.0, 0.0, 0.0, 1.0))
        });
        let _ = self.world.component_add(y, {
            let input = BufReader::new(File::open("data/models/square.obj").unwrap());
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

            Model::new(vertices, indices, Vec4::new(0.0, 1.0, 0.0, 1.0))
        });
        let _ = self.world.component_add(z, {
            let input = BufReader::new(File::open("data/models/smallsphere.obj").unwrap());
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

            Model::new(vertices, indices, Vec4::new(0.0, 0.0, 1.0, 1.0))
        });

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

        let camera = Camera::new(Vec3::new(0.0, 2.0, 6.0), 60.0, 1.0, 100.0, &window);
        let sun = DirectionalLight::new([2.0, 10.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.05);

        self.world.resource_add(camera);
        self.world.resource_add(sun);
        self.world.resource_add(ambient);
        self.world.resource_add(window.clone());

        self.world.resource_add(Keyboard::new());
        self.world.resource_add(MouseMovement::default());

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

                let delta_time = self.delta_time;
                let world = &mut self.world;
                player_update(world, delta_time);

                *self.world.resource_get_mut::<MouseMovement>().unwrap() = MouseMovement::default();
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

    #[allow(unused_variables)]
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        #[allow(clippy::cast_possible_truncation)]
        if let DeviceEvent::MouseMotion { delta } = event {
            let mvmt = self.world.resource_get_mut::<MouseMovement>().unwrap();
            mvmt.x = delta.0 as f32;
            mvmt.y = delta.1 as f32;
        }
    }
}

fn player_update(world: &mut World, delta_time: Duration) {
    let player_entity = world.query::<Controllable>().first().unwrap().0;
    let keyboard = world
        .resource_get::<Keyboard>()
        .expect("keyboard should have been added during resumed");

    let mut velocity = Vec3::zero();
    if keyboard.contains(&KeyCode::KeyW) {
        velocity += Vec3::new(0.0, 0.0, -1.0); // right handed eww
    }
    if keyboard.contains(&KeyCode::KeyS) {
        velocity += Vec3::new(0.0, 0.0, 1.0);
    }
    if keyboard.contains(&KeyCode::KeyD) {
        velocity += Vec3::new(1.0, 0.0, 0.0);
    }
    if keyboard.contains(&KeyCode::KeyA) {
        velocity += Vec3::new(-1.0, 0.0, 0.0);
    }
    let player = world
        .component_get_mut::<Model>(player_entity)
        .expect("player should have model component");

    let speed = 2.0 * delta_time.as_secs_f32();
    velocity = player.rotation * velocity;

    if velocity != Vec3::zero() {
        velocity.normalize();
    }
    velocity *= speed;

    let turn_speed = 9.0 * delta_time.as_secs_f32();

    let forward = Vec3::new(0.0, 0.0, -1.0);
    let forward = player.rotation * forward;

    let lean = Rotor3::from_rotation_between(forward, velocity);
    dbg!(lean, forward, velocity);

    player.lean = lean * player.lean;
    dbg!(player.lean);

    player.velocity = velocity.into_homogeneous_vector();

    player.position += player.velocity;
    player.requires_update = true;
}
