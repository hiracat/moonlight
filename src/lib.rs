use ecs::{EntityId, World};
use engine::{
    components::{
        Aabb, AmbientLight, Collider, DirectionalLight, Dynamic, Model, PointLight, RigidBody,
        Transform,
    },
    renderer::{self, Camera, Renderer, Vertex},
};
use obj::load_obj;
use std::{
    collections::HashSet,
    f32::consts::PI,
    fs::File,
    io::BufReader,
    thread::sleep,
    time::{Duration, Instant},
};
use ultraviolet::{Rotor3, Slerp, Vec3};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

mod ecs;
mod engine;
mod shaders;

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

fn load_model(path: &str, renderer: &Renderer) -> Model {
    let input = BufReader::new(File::open(path).unwrap());
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

    Model::create(renderer, vertices, indices)
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
        let red_light = self.world.entity_create();
        let blue_light = self.world.entity_create();
        let green_light = self.world.entity_create();
        let x_axis = self.world.entity_create();
        let y_axis = self.world.entity_create();
        let z_axis = self.world.entity_create();

        let _ = self.world.component_add(fox, Transform::new());
        let _ = self.world.component_add(fox, Controllable);
        let _ = self.world.component_add(ground, Transform::new());

        let _ = self.world.component_add(
            z_axis,
            Collider::Aabb(Aabb::new(
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.0, 0.0, 0.0),
            )),
        );

        let _ = self.world.component_add(
            fox,
            Collider::Aabb(Aabb::new(
                Vec3::new(0.331, 0.88, 0.331),
                Vec3::new(0.0, 0.44, 0.0),
            )),
        );

        let camera = Camera::new(Vec3::new(0.0, 5.0, -6.0), 60.0, 1.0, 100.0, &window);
        let sun = DirectionalLight::new([2.0, 10.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.05);

        self.world.resource_add(camera);
        self.world.resource_add(sun);
        self.world.resource_add(ambient);
        self.world.resource_add(window.clone());
        self.world.resource_add(Keyboard::new());
        self.world.resource_add(MouseMovement::default());

        self.renderer = Some(Renderer::init(event_loop, &mut self.world, &window));
        let renderer = self.renderer.as_ref().unwrap();

        let _ = self.world.component_add(fox, RigidBody::new());

        let _ = self.world.component_add(
            red_light,
            PointLight::create(
                renderer,
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                10.0,
                None,
                None,
            ),
        );
        let _ = self.world.component_add(
            red_light,
            Transform::from(Some(Vec3::new(5.0, 2.0, 0.0)), None, None),
        );

        let _ = self.world.component_add(
            green_light,
            PointLight::create(
                renderer,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                10.0,
                None,
                None,
            ),
        );
        let _ = self.world.component_add(
            green_light,
            Transform::from(Some(Vec3::new(-5.0, 2.0, 0.0)), None, None),
        );

        let _ = self.world.component_add(
            blue_light,
            PointLight::create(
                renderer,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                10.0,
                None,
                None,
            ),
        );
        let _ = self.world.component_add(
            blue_light,
            Transform::from(Some(Vec3::new(0.0, 2.0, 5.0)), None, None),
        );

        let _ = self
            .world
            .component_add(x_axis, load_model("data/models/square.obj", renderer));
        let _ = self
            .world
            .component_add(y_axis, load_model("data/models/square.obj", renderer));
        let _ = self
            .world
            .component_add(z_axis, load_model("data/models/smallsphere.obj", renderer));
        let _ = self
            .world
            .component_add(fox, load_model("data/models/low poly fox.obj", renderer));
        let _ = self
            .world
            .component_add(ground, load_model("data/models/groundplane.obj", renderer));
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

                let delta_time = self.delta_time.as_secs_f32();
                let world = &mut self.world;
                camera_update(world, delta_time);
                player_update(world, delta_time);
                physics_update(world, delta_time);

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

fn physics_update(world: &mut World, delta_time: f32) {
    let dynamic_entities: HashSet<EntityId> =
        world.query_entities::<Dynamic>().into_iter().collect();

    let mut entities = world.query3_mut::<Collider, Model, Transform>();
    for i in 0..entities.len() {
        for j in i + 1..entities.len() {
            if !entities[i].1.intersects(entities[j].1) {
                continue;
            }
            let min = {
                let pen_vec3 = entities[i].1.penetration_vector(entities[j].1);
                let abs = pen_vec3.abs();
                let min_index = if abs.x <= abs.y && abs.x <= abs.z {
                    0
                } else if abs.y <= abs.z {
                    1
                } else {
                    2
                };
                // Construct the output Vec3
                match min_index {
                    0 => Vec3::new(pen_vec3.x, 0.0, 0.0),
                    1 => Vec3::new(0.0, pen_vec3.y, 0.0),
                    2 => Vec3::new(0.0, 0.0, pen_vec3.z),
                    _ => unreachable!(),
                }
            };

            if dynamic_entities.contains(&entities[i].0)
                && dynamic_entities.contains(&entities[j].0)
            {
                entities[i].3.position -= min * 0.5;
                entities[j].3.position += min * 0.5;
            } else {
                if dynamic_entities.contains(&entities[i].0) {
                    entities[i].3.position -= min;
                }
                if dynamic_entities.contains(&entities[i].0) {
                    entities[j].3.position += min;
                }
            }
        }
    }

    let mut entities = world.query3_mut::<Model, Transform, RigidBody>();
    for i in 0..entities.len() {
        dbg!(&entities[i].0);
        dbg!(&entities[i].2.position);
        let velocity = entities[i].3.velocity;
        entities[i].2.position += velocity;
        entities[i].3.velocity *= 0.5 * delta_time;

        entities[i].2.dirty = true;
    }
}

fn camera_update(world: &mut World, _delta_time: f32) {
    let mouse = *world.resource_get::<MouseMovement>().unwrap();

    let player = world.query::<Controllable>().first().unwrap().0;

    let player_position = world.component_get::<Transform>(player).unwrap().position;

    // let player_rotation = world.component_get::<Model>(player).unwrap().rotation;
    let camera = world.resource_get_mut::<Camera>().unwrap();
    // let target_offset = Vec3 {
    //     x: 0.0,
    //     y: 5.0,
    //     z: -10.0,
    // };
    // let local_target_offset = target_offset.rotated_by(player_rotation);
    //
    // let target = player_position.xyz() + local_target_offset;
    if player_position.x > 10.0 {
        sleep(Duration::from_nanos(1));
    }

    let sensativity = 0.002;
    camera.pitch += mouse.y * sensativity;
    camera.yaw -= mouse.x * sensativity;
    if camera.pitch < -PI / 2.0 {
        camera.pitch = -PI / 2.0 + 0.001;
    }
    if camera.pitch > PI / 2.0 {
        camera.pitch = PI / 2.0 - 0.01;
    }

    camera.rotation = Rotor3::from_euler_angles(0.0, -camera.pitch, -camera.yaw);
    dbg!(camera.pitch, camera.yaw);

    let target_distance = 20.0;

    let backward = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    }
    .rotated_by(camera.rotation)
    .normalized();

    let offset = backward * target_distance;

    camera.position = player_position + offset;
}

fn player_update(world: &mut World, delta_time: f32) {
    let keyboard = world
        .resource_get::<Keyboard>()
        .expect("keyboard should have been added during resumed");

    let mut velocity = Vec3::zero();
    let mut speed = 2.0 * delta_time;
    if keyboard.contains(&KeyCode::ShiftLeft) {
        speed *= 4.0;
    }
    if keyboard.contains(&KeyCode::KeyW) {
        velocity += Vec3::new(0.0, 0.0, -1.0);
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

    let camera_rotation = world.resource_get::<Camera>().unwrap().rotation;

    let mut binding = world.query3_mut::<Controllable, Transform, RigidBody>();
    let (_, _, transform, rigidbody) = binding.first_mut().expect("player should exist!");

    if velocity == Vec3::zero() {
        rigidbody.velocity = Vec3::zero();
    } else {
        velocity = camera_rotation * velocity;
        let mut velocity = Vec3 {
            x: velocity.x,
            y: 0.0,
            z: velocity.z,
        };

        velocity = velocity.normalized();

        let forward = -Vec3::unit_z();
        let face_direction = if forward.dot(velocity) < -1.0 + 0.0001 {
            Rotor3::from_rotation_xz(PI)
        } else {
            Rotor3::from_rotation_between(forward, velocity)
        };

        dbg!(face_direction);
        let rotation_speed = 5.0;
        velocity *= speed;
        rigidbody.velocity = velocity;
        dbg!(transform.rotation.dot(face_direction));
        transform.rotation = transform
            .rotation
            .slerp(face_direction, rotation_speed * delta_time)
            .normalized();
        dbg!(transform.rotation);
    }
}
