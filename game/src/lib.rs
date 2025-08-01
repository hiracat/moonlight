use core::f32;
use std::path::Path;

use gltf;
use image::ImageReader;
use moonlight::components::Texture;
use moonlight::ecs::World;
use moonlight::{
    components::{
        Aabb, AmbientLight, Collider, DirectionalLight, Model, PointLight, RigidBody, Transform,
    },
    renderer::{self, Camera, Renderer, Vertex},
};
use std::{
    collections::HashSet,
    f32::consts::PI,
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

///Panics:
///panics if the number of meshes in the gltf asset is != 1, or if the mesh does not contain
///positions, normals and texture coordinates
fn load_model(path: &str, renderer: &mut Renderer) -> Model {
    let (document, buffers, _images) = gltf::import(path).unwrap();
    assert_eq!(document.meshes().len(), 1);
    let mesh = &document.meshes().next().unwrap();
    let primative: &gltf::Primitive = &mesh.primitives().next().unwrap();

    let reader = primative.reader(|buffer| Some(&buffers[buffer.index()]));

    let indices = reader.read_indices().unwrap().into_u32().collect();
    let positions = reader.read_positions().unwrap();
    let normals = reader.read_normals().unwrap();
    let tex_coords = reader.read_tex_coords(0).unwrap().into_f32();

    let vertices = positions
        .zip(normals)
        .zip(tex_coords)
        .map(|((position, normal), uv)| Vertex {
            position,
            normal,
            uv,
        })
        .collect();

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
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .unwrap();
        window.set_cursor_visible(false);
        self.renderer = Some(Renderer::init(event_loop, &window));
        let renderer = unsafe { self.renderer.as_mut().unwrap_unchecked() };

        // ───────────────────────────────────────────────────────
        // 2) Create “static” resources (camera, lights, input, etc.)
        // ───────────────────────────────────────────────────────
        // Camera at (0, 5, -6), fov=60°, near=1.0, far=100.0
        let camera = Camera::create(Vec3::new(0.0, 5.0, -6.0), 60.0, 1.0, 200.0, renderer);
        // Sun (directional) and ambient light
        let directional =
            DirectionalLight::create(renderer, [200.0, 10.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let ambient = AmbientLight::create(renderer, [1.0, 1.0, 1.0], 0.05);

        // Keyboard + mouse input resources
        let keyboard = Keyboard::new();
        let mouse_movement = MouseMovement::default();
        let empty_image = Texture::default_image(renderer);

        // Register resources into the world
        self.world.resource_add(camera);
        self.world.resource_add(directional);
        self.world.resource_add(ambient);
        self.world.resource_add(window.clone());
        self.world.resource_add(keyboard);
        self.world.resource_add(mouse_movement);
        self.world.resource_add(empty_image);

        let renderer = self.renderer.as_mut().unwrap();

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
        let standing_block = self.world.entity_create();

        let image = ImageReader::open(Path::new("data/textures/fox_texture.png"))
            .unwrap()
            .decode()
            .unwrap();

        let image = Texture::create_image(&image, renderer);
        self.world.component_add(fox, image).unwrap();

        let image = ImageReader::open(Path::new("data/textures/ground_soft.jpg"))
            .unwrap()
            .decode()
            .unwrap();

        let image = Texture::create_image(&image, renderer);
        self.world.component_add(ground, image).unwrap();

        // ───────────────────────────────────────────────────────
        // THE FOX - Center stage, elevated on a mystical platform
        // ───────────────────────────────────────────────────────
        self.world
            .component_add(
                fox,
                Transform::from(
                    Some(Vec3::new(0.0, 3.0, 0.0)),            // Elevated at center
                    Some(Rotor3::from_rotation_xz(PI * 0.25)), // Slight rotation for dramatic pose
                    Some(Vec3::new(1.0, 1.0, 1.0)),
                ),
            )
            .unwrap();
        self.world.component_add(fox, Controllable).unwrap();

        // Fox collision - smaller and more precise
        let fox_half_extents = Vec3::new(0.4, 0.588, 0.4);
        let fox_center_offset = Vec3::new(0.0, 0.0, 0.0);
        self.world
            .component_add(
                fox,
                Collider::Aabb(Aabb::new(fox_half_extents, fox_center_offset)),
            )
            .unwrap();
        self.world.component_add(fox, RigidBody::new()).unwrap();
        self.world
            .component_add(fox, load_model("data/models/low_poly_fox.glb", renderer))
            .unwrap();

        self.world
            .component_add(
                ground,
                Transform::from(None, None, Some(Vec3::new(100.0, 1.0, 100.0))),
            )
            .unwrap();
        self.world
            .component_add(ground, load_model("data/models/ground_plane.glb", renderer))
            .unwrap();

        // Ground collision - the full 40x40 area with some depth
        self.world
            .component_add(
                ground,
                Collider::Aabb(Aabb::new(
                    Vec3::new(20.0, 4.0, 20.0), // Half-extents: 40x2x40 total size
                    Vec3::new(0.0, -4.0, 0.0),  // Center offset: buried 1 unit down
                )),
            )
            .unwrap();

        self.world
            .component_add(
                x_axis,
                Transform::from(
                    Some(Vec3::new(8.0, 0.0, 8.0)), // High and forward-right
                    None,
                    Some(Vec3::new(10.0, 9.0, 10.0)),
                ),
            )
            .unwrap();
        self.world
            .component_add(
                y_axis,
                Transform::from(
                    Some(Vec3::new(-12.0, 0.0, -3.0)), // Tallest, to the left and slightly back
                    None,
                    Some(Vec3::new(2.0, 2.0, 2.0)),
                ),
            )
            .unwrap();
        self.world
            .component_add(
                z_axis,
                Transform::from(
                    Some(Vec3::new(5.0, 0.0, -10.0)), // Behind the fox, watching over
                    None,
                    Some(Vec3::new(3.0, 3.0, 3.0)),
                ),
            )
            .unwrap();

        self.world
            .component_add(
                standing_block,
                Transform::from(
                    Some(Vec3::new(4.0, 5.5, -3.0)),
                    None,
                    Some(Vec3::new(3.0, 0.3, 3.0)),
                ),
            )
            .unwrap();

        // Colliders for the floating monuments - smaller, more precise
        let cube_collider = Collider::Aabb(Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::zero()));
        self.world.component_add(x_axis, cube_collider).unwrap();
        self.world.component_add(y_axis, cube_collider).unwrap();
        self.world.component_add(z_axis, cube_collider).unwrap();
        self.world
            .component_add(standing_block, cube_collider)
            .unwrap();

        // Attach models to monuments
        self.world
            .component_add(x_axis, load_model("data/models/large_cube.glb", renderer))
            .unwrap();
        self.world
            .component_add(y_axis, load_model("data/models/large_cube.glb", renderer))
            .unwrap();
        self.world
            .component_add(z_axis, load_model("data/models/large_cube.glb", renderer))
            .unwrap();
        self.world
            .component_add(
                standing_block,
                load_model("data/models/large_cube.glb", renderer),
            )
            .unwrap();

        // ───────────────────────────────────────────────────────
        // DRAMATIC LIGHTING - Three-point mystical lighting setup
        // ───────────────────────────────────────────────────────

        // RED LIGHT: "Crimson Flame" - Key light, high intensity, following the red monument
        self.world
            .component_add(
                red_light,
                PointLight::create(
                    renderer,
                    Vec3::new(1.0, 0.3, 0.2), // Warm red-orange
                    23.0,                     // High intensity for drama
                    None,
                    None,
                ),
            )
            .unwrap();
        self.world
            .component_add(
                red_light,
                Transform::from(Some(Vec3::new(10.0, 8.0, 6.0)), None, None), // Near the red monument, but higher
            )
            .unwrap();

        // GREEN LIGHT: "Emerald Glow" - Fill light, medium intensity, illuminating the left side
        self.world
            .component_add(
                green_light,
                PointLight::create(
                    renderer,
                    Vec3::new(0.2, 1.0, 0.3), // Vibrant green
                    15.0,                     // Medium intensity
                    None,
                    None,
                ),
            )
            .unwrap();
        self.world
            .component_add(
                green_light,
                Transform::from(Some(Vec3::new(-10.0, 7.0, -2.0)), None, None), // Left side, near green monument
            )
            .unwrap();

        // BLUE LIGHT: "Azure Whisper" - Rim light, subtle, creating mystical atmosphere
        self.world
            .component_add(
                blue_light,
                PointLight::create(
                    renderer,
                    Vec3::new(0.3, 0.4, 1.0), // Cool blue
                    12.0,                     // Lower intensity for subtle rim lighting
                    None,
                    None,
                ),
            )
            .unwrap();
        self.world
            .component_add(
                blue_light,
                Transform::from(Some(Vec3::new(2.0, 5.0, -12.0)), None, None), // Behind fox, creating rim light
            )
            .unwrap();
        self.prev_frame_end = Instant::now();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => self.renderer.as_mut().unwrap().framebuffer_resized = true,
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
            WindowEvent::Focused(focused) => {
                if focused {
                    self.delta_time = Duration::from_secs(0);
                    self.prev_frame_end = Instant::now();
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
    //NOTE: MOVEMENT INTEGRATION
    let mut rigidbody = world.query2_mut::<Transform, RigidBody>();
    for i in 0..rigidbody.len() {
        //NOTE:COMPUTES THE HORIZONTAL SPEED REDUCTION
        let horizontal_velocity =
            Vec3::new(rigidbody[i].2.velocity.x, 0.0, rigidbody[i].2.velocity.z);
        let horizontal_friction: f32 = 20.0; // how many units of speed to remove per second
        if horizontal_velocity.mag() > 0.0 {
            // compute how much to drop this frame:
            let frame_decel = horizontal_friction * delta_time;

            if horizontal_velocity.mag() <= frame_decel {
                rigidbody[i].2.velocity.x = 0.0;
                rigidbody[i].2.velocity.z = 0.0;
            } else {
                let direction = horizontal_velocity.normalized();
                rigidbody[i].2.velocity -= direction * frame_decel;
            }
        }
        // NOTE: APPLIES GRAVITY, GROUNDED FLAGS DONT MATTER BECAUSE WILL BE CORRECTED FOR IN
        // RESOLUTION FOR POSITION AND THEN VELOCITY
        rigidbody[i].2.velocity.y -= 9.8 * delta_time;

        let velocity = rigidbody[i].2.velocity;
        rigidbody[i].1.position += velocity * delta_time;
        rigidbody[i].1.dirty = true;
        dbg!(rigidbody[i].2.velocity);
    }

    let rigidbody = world.query2_entities::<Transform, RigidBody>();
    let mut collidable = world.query2_mut::<Transform, Collider>();
    let mut collision_penetrations = Vec::new();

    //NOTE: COLLISION RESOLUTION
    for i in 0..collidable.len() {
        for j in i + 1..collidable.len() {
            match Collider::penetration_vector(
                collidable[i].2,
                collidable[j].2,
                collidable[i].1,
                collidable[j].1,
            ) {
                Some(pen_vec) => {
                    if rigidbody.contains(&collidable[i].0) && rigidbody.contains(&collidable[j].0)
                    {
                        collidable[i].1.position += pen_vec * 0.5;
                        collision_penetrations.push((collidable[i].0, pen_vec * 0.5));
                        collidable[j].1.position -= pen_vec * 0.5;
                        collision_penetrations.push((collidable[j].0, pen_vec * -0.5));
                    } else {
                        if rigidbody.contains(&collidable[i].0) {
                            collidable[i].1.position += pen_vec;
                            collision_penetrations.push((collidable[i].0, pen_vec));
                        }
                        if rigidbody.contains(&collidable[j].0) {
                            collidable[j].1.position -= pen_vec;
                            collision_penetrations.push((collidable[j].0, pen_vec * -1.0));
                        }
                    }
                }
                None => (),
            }
        }
    }
    //NOTE: VELOCITY UPDATE
    for collision in collision_penetrations {
        let rigidbody = world
            .component_get_mut::<RigidBody>(collision.0)
            .expect("thing with collision regesterd should have rigidbody");
        rigidbody.velocity = set_axis_component(rigidbody.velocity, collision.1, 0.0)
    }
}

// accepts a velocity, a minimum vector to resolve the collision, a coefficient of restitution, and returns the new
// velocity
fn set_axis_component(velocity: Vec3, collision_vector: Vec3, restitution: f32) -> Vec3 {
    let collision = collision_vector.normalized();
    let projection = collision * velocity.dot(collision);
    return velocity - (1.0 + restitution) * projection;
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

    // BUG: allows the player to jump at the apex of their jump, but the period of time which the
    // bug is viable to exploit is almost zero so not going to fix
    if keyboard.contains(&KeyCode::Space) && rigidbody.velocity.y.abs() < 0.00001 {
        jump = 8.0
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
