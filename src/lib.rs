use core::f32;
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
    f32::{consts::PI, EPSILON},
    fs::File,
    io::BufReader,
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

pub mod ecs;
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
        // ───────────────────────────────────────────────────────
        // 1) Window & input setup
        // ───────────────────────────────────────────────────────
        if self.renderer.is_some() {
            return;
        }

        let window = renderer::create_window(event_loop);
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        window.set_cursor_visible(false);

        // ───────────────────────────────────────────────────────
        // 2) Create “static” resources (camera, lights, input, etc.)
        // ───────────────────────────────────────────────────────
        // Camera at (0, 5, -6), fov=60°, near=1.0, far=100.0
        let camera = Camera::new(Vec3::new(0.0, 5.0, -6.0), 60.0, 1.0, 200.0, &window);
        // Sun (directional) and ambient light
        let sun = DirectionalLight::new([2.0, 10.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.05);

        // Keyboard + mouse input resources
        let keyboard = Keyboard::new();
        let mouse_movement = MouseMovement::default();

        // Register resources into the world
        self.world.resource_add(camera);
        self.world.resource_add(sun);
        self.world.resource_add(ambient);
        self.world.resource_add(window.clone());
        self.world.resource_add(keyboard);
        self.world.resource_add(mouse_movement);

        // Initialize renderer after resources exist in the world
        self.renderer = Some(Renderer::init(event_loop, &mut self.world, &window));
        let renderer = self.renderer.as_ref().unwrap();

        // ───────────────────────────────────────────────────────
        // 3) Spawn entities
        // ───────────────────────────────────────────────────────
        let fox = self.world.entity_create();
        let ground = self.world.entity_create();
        let red_light = self.world.entity_create();
        let blue_light = self.world.entity_create();
        let green_light = self.world.entity_create();
        let x_axis = self.world.entity_create();
        let y_axis = self.world.entity_create();
        let z_axis = self.world.entity_create();

        // ───────────────────────────────────────────────────────
        // 4) Add components to “fox”
        // ───────────────────────────────────────────────────────
        // Give the fox a transform at origin, make it controllable, add physics collider/body
        let _ = self.world.component_add(
            fox,
            Transform::from(Some(Vec3::new(4.0, 20.0, 1.0)), None, None),
        );
        let _ = self.world.component_add(fox, Controllable);
        let half_extents = Vec3::new(0.2, 0.55, 0.2);
        let center_offset = Vec3::new(0.0, 0.55, 0.0);

        let _ = self
            .world
            .component_add(fox, Collider::Aabb(Aabb::new(half_extents, center_offset)));

        let _ = self.world.component_add(fox, RigidBody::new());
        let _ = self.world.component_add(fox, Dynamic);
        // Finally, attach the visual model for the fox
        let _ = self
            .world
            .component_add(fox, load_model("data/models/low poly fox.obj", renderer));

        // ───────────────────────────────────────────────────────
        // 5) Add components to “ground”
        // ───────────────────────────────────────────────────────
        let _ = self.world.component_add(ground, Transform::new());
        let _ = self
            .world
            .component_add(ground, load_model("data/models/groundplane.obj", renderer));

        let _ = self.world.component_add(
            ground,
            Collider::Aabb(Aabb::new(
                Vec3::new(20.0, 0.5, 20.0),
                Vec3::new(0.0, -0.5, 0.0),
            )),
        );

        // ───────────────────────────────────────────────────────
        // 6) Set up point‐lights (red, green, blue)
        // ───────────────────────────────────────────────────────
        // --- Red light ---
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
            Transform::from(Some(Vec3::new(5.0, 5.0, 0.0)), None, None),
        );

        // --- Green light ---
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
            Transform::from(Some(Vec3::new(-5.0, 5.0, 0.0)), None, None),
        );

        // --- Blue light ---
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
            Transform::from(Some(Vec3::new(0.0, 5.0, 5.0)), None, None),
        );

        // ───────────────────────────────────────────────────────
        // 7) Set up coordinate‐axis cubes with transforms
        // ───────────────────────────────────────────────────────
        // Position each “unit cube” along its respective axis:
        //   x_axis at (1, 0, 0), y_axis at (0, 1, 0), z_axis at (0, 0, 1)
        let _ = self.world.component_add(
            x_axis,
            Transform::from(Some(Vec3::new(1.0, 0.0, 0.0)), None, None),
        );
        let _ = self.world.component_add(
            y_axis,
            Transform::from(Some(Vec3::new(0.0, 1.0, 0.0)), None, None),
        );
        let _ = self.world.component_add(
            z_axis,
            Transform::from(Some(Vec3::new(0.0, 0.0, 1.0)), None, None),
        );

        // Optionally add a collider to each axis cube (if you want them to be collidable)
        let unit_collider = Collider::Aabb(Aabb::new(
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(0.0, 0.0, 0.0),
        ));
        let _ = self.world.component_add(x_axis, unit_collider);
        let _ = self.world.component_add(y_axis, unit_collider);
        let _ = self.world.component_add(z_axis, unit_collider);

        // Attach the models for each axis cube
        let _ = self
            .world
            .component_add(x_axis, load_model("data/models/square.obj", renderer));
        let _ = self
            .world
            .component_add(y_axis, load_model("data/models/square.obj", renderer));
        let _ = self
            .world
            .component_add(z_axis, load_model("data/models/square.obj", renderer));
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
                player_update(world, delta_time);
                physics_update(world, delta_time);
                camera_update(world, delta_time);

                *self.world.resource_get_mut::<MouseMovement>().unwrap() = MouseMovement::default();
                self.renderer.as_mut().unwrap().draw(&mut self.world);
                self.delta_time = self.prev_frame_end.elapsed();
                eprintln!("fps: {}", 1.0 / self.delta_time.as_secs_f32());

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
    apply_gravity(world, delta_time);
    integrate_movement(world, delta_time);
    resolve_colisions(world);
}
fn apply_gravity(world: &mut World, delta_time: f32) {
    let grounded_entities = get_entity_grounded(world);
    let dynamic_entities: HashSet<EntityId> =
        world.query_entities::<Dynamic>().into_iter().collect();

    for (entity, grounded) in grounded_entities {
        if !dynamic_entities.contains(&entity) {
            continue;
        }
        match world.component_get_mut::<RigidBody>(entity) {
            Ok(rigidbody) => {
                if grounded {
                    rigidbody.velocity.y = rigidbody.velocity.y.clamp(0.0, f32::MAX);
                    rigidbody.grounded = true;
                } else {
                    rigidbody.velocity.y -= 9.8 * delta_time;
                    rigidbody.grounded = false;
                }
                dbg!(rigidbody);
            }
            Err(_) => {}
        };
    }
}
fn resolve_colisions(world: &mut World) {
    let dynamic_entities: HashSet<EntityId> =
        world.query_entities::<Dynamic>().into_iter().collect();

    let mut entities = world.query3_mut::<Collider, Model, Transform>();

    for i in 0..entities.len() {
        for j in i + 1..entities.len() {
            match Collider::penetration_vector(
                entities[i].1,
                entities[j].1,
                entities[i].3,
                entities[j].3,
            ) {
                Some(pen_vec) => {
                    if dynamic_entities.contains(&entities[i].0)
                        && dynamic_entities.contains(&entities[j].0)
                    {
                        entities[i].3.position += pen_vec * 0.5;
                        entities[j].3.position -= pen_vec * 0.5;
                    } else {
                        if dynamic_entities.contains(&entities[i].0) {
                            entities[i].3.position += pen_vec;
                        }
                        if dynamic_entities.contains(&entities[j].0) {
                            entities[j].3.position -= pen_vec;
                        }
                    }
                }
                None => (),
            }
        }
    }
}
fn integrate_movement(world: &mut World, delta_time: f32) {
    let mut entities = world.query2_mut::<Transform, RigidBody>();

    for i in 0..entities.len() {
        dbg!(i, &entities[i].1.position, &entities[i].2);
        let decel: f32 = 20.0; // NOTE: how many units of speed to remove per second
        let horizontal_velocity =
            Vec3::new(entities[i].2.velocity.x, 0.0, entities[i].2.velocity.z);

        if horizontal_velocity.mag() > 0.0 {
            // compute how much to drop this frame:
            let frame_decel = decel * delta_time;

            if horizontal_velocity.mag() <= frame_decel {
                entities[i].2.velocity.x = 0.0;
                entities[i].2.velocity.z = 0.0;
            } else {
                let direction = horizontal_velocity.normalized();
                entities[i].2.velocity -= direction * frame_decel;
            }
        }

        let velocity = entities[i].2.velocity;
        entities[i].1.position += velocity * delta_time;
        entities[i].1.dirty = true;
        dbg!(entities[i].2.velocity);
    }
}

fn get_entity_grounded(world: &mut World) -> Vec<(EntityId, bool)> {
    let entities = world.query2_mut::<Collider, Transform>();
    let mut grounded_entities = Vec::with_capacity(entities.len());

    for i in 0..entities.len() {
        let (entity, collider1, transform1) = &entities[i];
        let mut grounded;
        for j in 0..entities.len() {
            if i == j {
                continue;
            }

            let (_, collider2, transform2) = &entities[j];

            let epsilon = EPSILON;
            grounded = Collider::penetration_vector(
                &collider1,
                collider2,
                &Transform::from(
                    Some(Vec3 {
                        y: transform1.position.y - epsilon,
                        ..transform1.position
                    }),
                    Some(transform1.rotation),
                    Some(transform1.scale),
                ),
                transform2,
            )
            .is_some();

            if grounded {
                grounded_entities.push((*entity, true));
                break;
            }
        }
        grounded_entities.push((*entity, false));
    }
    grounded_entities
}

fn camera_update(world: &mut World, _delta_time: f32) {
    let mouse = *world.resource_get::<MouseMovement>().unwrap();

    let player = world.query::<Controllable>().first().unwrap().0;

    let player_position = world.component_get::<Transform>(player).unwrap().position;

    // let player_rotation = world.component_get::<Model>(player).unwrap().rotation;
    let camera = world.resource_get_mut::<Camera>().unwrap();

    let sensativity = 0.002;
    camera.pitch += mouse.y * sensativity;
    camera.yaw -= mouse.x * sensativity;
    if camera.pitch < -PI / 2.0 + 0.01 {
        camera.pitch = -PI / 2.0 + 0.02;
    }
    if camera.pitch > PI / 2.0 - 0.01 {
        camera.pitch = PI / 2.0 - 0.02;
    }

    camera.rotation = Rotor3::from_euler_angles(0.0, -camera.pitch, -camera.yaw);

    let target_distance = 10.0;

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
        .expect("keyboard should have been added during resumed")
        .clone();

    let mut delta_v = Vec3::zero();
    let mut jump = 0.0;
    let mut max_speed = 3.0;
    if keyboard.contains(&KeyCode::ShiftLeft) {
        max_speed *= 2.0;
    }
    if keyboard.contains(&KeyCode::KeyW) {
        delta_v += Vec3::new(0.0, 0.0, -1.0);
    }
    if keyboard.contains(&KeyCode::KeyS) {
        delta_v += Vec3::new(0.0, 0.0, 1.0);
    }
    if keyboard.contains(&KeyCode::KeyD) {
        delta_v += Vec3::new(1.0, 0.0, 0.0);
    }
    if keyboard.contains(&KeyCode::KeyA) {
        delta_v += Vec3::new(-1.0, 0.0, 0.0);
    }

    let camera_rotation = world.resource_get::<Camera>().unwrap().rotation;
    let mut binding = world.query3_mut::<Controllable, Transform, RigidBody>();
    let (_, _, transform, rigidbody) = binding.first_mut().expect("player should exist!");
    if keyboard.contains(&KeyCode::Space) && rigidbody.grounded {
        jump = 20.0
    }
    if !(delta_v == Vec3::zero()) {
        delta_v = camera_rotation * delta_v;
        delta_v = Vec3 {
            x: delta_v.x,
            y: 0.0,
            z: delta_v.z,
        };

        delta_v.normalize();

        let forward = -Vec3::unit_z();
        let face_direction = if forward.dot(delta_v) < -1.0 + 0.0001 {
            Rotor3::from_rotation_xz(PI)
        } else {
            Rotor3::from_rotation_between(forward, delta_v)
        };

        let mut horizontal_velocity = rigidbody.velocity;
        horizontal_velocity.y = 0.0;

        let speed_remaining = (max_speed - horizontal_velocity.mag()).clamp(0.0, max_speed);
        let acceleration = 20.0;

        delta_v *= speed_remaining * delta_time * acceleration;
        rigidbody.velocity += delta_v;

        let rotation_speed = 5.0;
        transform.rotation = transform
            .rotation
            .slerp(face_direction, rotation_speed * delta_time)
            .normalized();
    }
    rigidbody.velocity.y += jump;
    dbg!(rigidbody);
}
