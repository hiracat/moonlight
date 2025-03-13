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
    x: f64,
    y: f64,
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
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            eprintln!("resumed called while renderer is already some");
            return;
        }
        let window = renderer::create_window(event_loop);
        window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        window.set_cursor_visible(false);

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
                camera_update(world, delta_time);
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

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let mvmt = self.world.resource_get_mut::<MouseMovement>().unwrap();
            mvmt.x = delta.0;
            mvmt.y = delta.1;
        }
    }
}

fn camera_update(world: &mut World, delta_time: Duration) {
    let mouse_movement = *world.resource_get::<MouseMovement>().unwrap();
    let player = world.query::<Controllable>().first().unwrap().0;
    let player = world.component_get_mut::<Model>(player).unwrap();

    let player_position = player.position.xyz();
    let player_rotation = player.rotation;

    dbg!(mouse_movement);
    let camera = world.resource_get_mut::<Camera>().unwrap();

    let offset = player_rotation * Vec3::new(0.0, 2.0, -4.0);

    let mouse_sensativity = 0.2;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    let rotation = Rotor3::from_euler_angles(
        0.0,
        mouse_movement.y as f32 * mouse_sensativity * delta_time.as_secs_f32(),
        mouse_movement.x as f32 * mouse_sensativity * delta_time.as_secs_f32(),
    );
    dbg!(rotation);

    camera.rotation = rotation * camera.rotation;
    camera.position = player_position + offset;
}
fn player_update(world: &mut World, delta_time: Duration) {
    let keyboard = world
        .resource_get::<Keyboard>()
        .expect("keyboard should have been added during resumed");

    let mut movement_direction = Vec3::zero();
    if keyboard.contains(&KeyCode::KeyW) {
        movement_direction += Vec3::new(0.0, 0.0, 1.0);
    }
    if keyboard.contains(&KeyCode::KeyS) {
        movement_direction += Vec3::new(0.0, 0.0, -1.0);
    }
    if keyboard.contains(&KeyCode::KeyD) {
        movement_direction += Vec3::new(-1.0, 0.0, 0.0);
    }
    if keyboard.contains(&KeyCode::KeyA) {
        movement_direction += Vec3::new(1.0, 0.0, 0.0);
    }
    if movement_direction == Vec3::zero() {
        return;
    }
    movement_direction.normalize();

    let camera_rotation = world.resource_get::<Camera>().unwrap().rotation;
    let yaw_only = get_yaw_rotation(camera_rotation);

    let player_entity = world.query::<Controllable>().first().unwrap().0;
    let player = world
        .component_get_mut::<Model>(player_entity)
        .expect("player should have model component");

    let speed = 5.0 * delta_time.as_secs_f32();

    let velocity = (yaw_only * movement_direction) * speed;

    player.position += velocity.into_homogeneous_vector();
    if movement_direction != Vec3::zero() {
        player.rotation = yaw_only;
    }
    player.requires_update = true;
}
fn get_yaw_rotation(rotation: Rotor3) -> Rotor3 {
    // This projects the rotation onto the XZ plane
    let forward = rotation * Vec3::new(0.0, 0.0, 1.0);
    let forward_xz = Vec3::new(forward.x, 0.0, forward.z).normalized();
    Rotor3::from_rotation_between(Vec3::new(0.0, 0.0, 1.0), forward_xz)
}
