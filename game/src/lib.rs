use core::{f32, panic};
use moonlight;
use moonlight::components::Camera;
use moonlight::components::{AmbientLight, DirectionalLight, PointLight, Transform};
use moonlight::ecs::OptM;
use moonlight::ecs::ReqM;
use moonlight::ecs::World;
use moonlight::physics::Aabb;
use moonlight::physics::Collider;
use moonlight::physics::RigidBody;
use moonlight::renderer::draw::Renderer;
use moonlight::renderer::init::create_window;
use moonlight::renderer::resources::Material;
use moonlight::renderer::resources::Skybox;
use std::{
    collections::HashSet,
    f32::consts::PI,
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
static mut OFFSET_X: f32 = 0.0;
static mut OFFSET_Y: f32 = 0.0;
static mut OFFSET_Z: f32 = 0.0;

type Keyboard = HashSet<KeyCode>;

#[derive(Debug)]
struct Controllable;
#[derive(Debug, Default, Clone, Copy)]
struct MouseState {
    x: f32,
    y: f32,
    locked: bool,
}

pub struct App {
    renderer: Option<Renderer>,
    world: World,
    prev_frame_end: Instant,
    delta_time: Duration,
    current_frame: u64,
}
impl Default for App {
    fn default() -> Self {
        App {
            renderer: None,
            world: World::init(),
            prev_frame_end: Instant::now(),
            delta_time: Duration::default(),
            current_frame: 0,
        }
    }
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

        let window = create_window(event_loop);
        //HACK: magic number, i dont care right now

        self.renderer = Some(Renderer::init(event_loop, &window));

        // ───────────────────────────────────────────────────────
        // 2) Create “static” resources (camera, lights, input, etc.)
        // ───────────────────────────────────────────────────────
        // Camera at (0, 5, -6), fov=60°, near=1.0, far=100.0
        let camera = Camera::create(
            Vec3::new(0.0, 5.0, -6.0),
            60.0,
            1.0,
            200.0,
            window.inner_size().height as f32 / window.inner_size().width as f32,
        );
        // Sun (directional) and ambient light
        let directional = DirectionalLight::create(Vec4::new(200.0, 10.0, 0.0, 1.0), Vec3::new(0.2, 0.2, 0.2));
        let ambient = AmbientLight::create(Vec3::new(1.0, 1.0, 1.0), 0.05);

        // Keyboard + mouse input resources
        let keyboard = Keyboard::new();
        let mouse_movement = MouseState::default();

        let renderer = self.renderer.as_mut().unwrap();

        // Register resources into the world
        self.world.add_resource(camera).unwrap();
        self.world.add_resource(directional).unwrap();
        self.world.add_resource(ambient).unwrap();
        self.world.add_resource(window.clone()).unwrap();
        self.world.add_resource(keyboard).unwrap();
        self.world.add_resource(mouse_movement).unwrap();

        let skybox = renderer.resource_manager.create_cubemap([
            "data/skybox/px.png",
            "data/skybox/nx.png",
            "data/skybox/py.png",
            "data/skybox/ny.png",
            "data/skybox/pz.png",
            "data/skybox/nz.png",
        ]);
        let skybox = Skybox::new(skybox);
        self.world.add_resource(skybox).unwrap();

        // ───────────────────────────────────────────────────────
        // 3) Spawn entities
        // ───────────────────────────────────────────────────────
        let fox = self.world.spawn();
        let ground = self.world.spawn();
        let red_light = self.world.spawn();
        let blue_light = self.world.spawn();
        let green_light = self.world.spawn();
        let x_axis = self.world.spawn();
        let y_axis = self.world.spawn();
        let z_axis = self.world.spawn();
        let standing_block = self.world.spawn();

        // ───────────────────────────────────────────────────────
        // THE FOX - Center stage, elevated on a mystical platform
        // ───────────────────────────────────────────────────────
        self.world
            .add(
                fox,
                Transform::from(
                    Some(Vec3::new(0.0, 3.0, 0.0)), // Elevated at center
                    Some(Rotor3::identity()),       // Slight rotation for dramatic pose
                    Some(Vec3::new(1.0, 1.0, 1.0)),
                ),
            )
            .unwrap();
        self.world.add(fox, Controllable).unwrap();

        // Fox collision - smaller and more precise
        let fox_half_extents = Vec3::new(0.4, 0.588, 0.4);
        let fox_center_offset = Vec3::new(0.0, 0.588, 0.0);
        self.world
            .add(
                fox,
                Collider::Aabb(Aabb::new(fox_half_extents, fox_center_offset)),
            )
            .unwrap();
        self.world.add(fox, RigidBody::new()).unwrap();
        let (fox_model, fox_animations) = renderer
            .resource_manager
            .load_gltf_asset("data/models/animated_fox.glb");
        let mut fox_animations = fox_animations.unwrap();
        fox_animations.current_playing = Some(fox_animations.animations[0].clone());

        self.world.add(fox, fox_model).unwrap();
        self.world.add(fox, fox_animations).unwrap();

        let albedo = renderer
            .resource_manager
            .create_texture("data/models/textures/animated_fox_texture.png");

        self.world.add(fox, Material::create(albedo)).unwrap();

        let albedo = renderer
            .resource_manager
            .create_texture("data/models/textures/ground.jpg");
        self.world.add(ground, Material::create(albedo)).unwrap();
        self.world.add(x_axis, Material::create(albedo)).unwrap();
        self.world.add(y_axis, Material::create(albedo)).unwrap();
        self.world.add(z_axis, Material::create(albedo)).unwrap();
        let albedo = renderer
            .resource_manager
            .create_texture("data/models/textures/ground_soft.jpg");
        self.world
            .add(standing_block, Material::create(albedo))
            .unwrap();

        self.world
            .add(
                ground,
                Transform::from(None, None, Some(Vec3::new(100.0, 1.0, 100.0))),
            )
            .unwrap();
        self.world
            .add(
                ground,
                renderer
                    .resource_manager
                    .load_gltf_asset("data/models/ground_plane.glb").0,
            )
            .unwrap();

        // Ground collision - the full 40x40 area with some depth
        self.world
            .add(
                ground,
                Collider::Aabb(Aabb::new(
                    Vec3::new(1.0, 4.0, 1.0),  // Half-extents: 40x2x40 total size
                    Vec3::new(0.0, -4.0, 0.0), // Center offset: buried 1 unit down
                )),
            )
            .unwrap();

        self.world
            .add(
                x_axis,
                Transform::from(
                    Some(Vec3::new(8.0, 0.0, 8.0)), // High and forward-right
                    None,
                    Some(Vec3::new(10.0, 9.0, 10.0)),
                ),
            )
            .unwrap();
        self.world
            .add(
                y_axis,
                Transform::from(
                    Some(Vec3::new(-12.0, 0.0, -3.0)), // Tallest, to the left and slightly back
                    None,
                    Some(Vec3::new(2.0, 2.0, 2.0)),
                ),
            )
            .unwrap();
        self.world
            .add(
                z_axis,
                Transform::from(
                    Some(Vec3::new(5.0, 0.0, -10.0)), // Behind the fox, watching over
                    None,
                    Some(Vec3::new(3.0, 3.0, 3.0)),
                ),
            )
            .unwrap();

        self.world
            .add(
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
        self.world.add(x_axis, cube_collider).unwrap();
        self.world.add(y_axis, cube_collider).unwrap();
        self.world.add(z_axis, cube_collider).unwrap();
        self.world.add(standing_block, cube_collider).unwrap();

        let cube = renderer.resource_manager.load_gltf_asset("data/models/large_cube.glb").0;
        // Attach models to monuments
        self.world
            .add(
                x_axis,
               cube,
            )
            .unwrap();
        self.world
            .add(
                y_axis,
                cube,
            )
            .unwrap();
        self.world
            .add(
                z_axis,
                cube,
            )
            .unwrap();
        self.world
            .add(
                standing_block,
                cube,
            )
            .unwrap();

        self.world
            .add(
                red_light,
                PointLight::new(
                    Vec3::new(1.0, 0.3, 0.2), // Warm red-orange
                    35.0,                     // Medium intensity
                    None,
                    Some(2.0),
                ),
            )
            .unwrap();
        self.world
            .add(
                red_light,
                Transform::from(Some(Vec3::new(10.0, 8.0, 6.0)), None, None), // Near the red monument, but higher
            )
            .unwrap();

        self.world
            .add(
                green_light,
                PointLight::new(
                    Vec3::new(0.2, 1.0, 0.3), // Vibrant green
                    35.0,                     // Medium intensity
                    None,
                    Some(2.0),
                ),
            )
            .unwrap();
        self.world
            .add(
                green_light,
                Transform::from(Some(Vec3::new(-10.0, 7.0, -2.0)), None, None), // Left side, near green monument
            )
            .unwrap();

        self.world
            .add(
                blue_light,
                PointLight::new(
                    Vec3::new(0.1, 0.1, 1.0), // Cool blue
                    55.0,                     // Medium intensity
                    Some(1.2),
                    Some(0.4),
                ),
            )
            .unwrap();
        self.world
            .add(
                blue_light,
                Transform::from(Some(Vec3::new(2.0, 5.0, -12.0)), None, None), // Behind fox, creating rim light
            )
            .unwrap();
        self.prev_frame_end = Instant::now();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let window = self.renderer.as_mut().unwrap().window.clone();
        let response = self
            .renderer
            .as_mut()
            .unwrap()
            .ui
            .winit_egui_state
            .on_window_event(&window, &event);
        if response.consumed == true {
            return;
        }
        if response.repaint == true {
            self.renderer.as_ref().unwrap().window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => self.renderer.as_mut().unwrap().framebuffer_resized = true,
            WindowEvent::RedrawRequested => {
                // println!("frame {} starts here", self.current_frame);
                let window = self.renderer.as_ref().unwrap().window.clone();
                let mut offset_x = unsafe { OFFSET_X };
                let mut offset_y = unsafe { OFFSET_Y };
                let mut offset_z = unsafe { OFFSET_Z };
                let raw_input = self
                    .renderer
                    .as_mut()
                    .unwrap()
                    .ui
                    .winit_egui_state
                    .take_egui_input(&window);
                let full_output = self
                    .renderer
                    .as_mut()
                    .unwrap()
                    .ui
                    .ui_ctx
                    .run(raw_input, |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            egui::Window::new("Camera Offset").show(&ctx, |ui| {
                                ui.add(
                                    egui::Slider::new(&mut offset_x, -10.0..=10.0).text("offset x"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut offset_y, -10.0..=10.0).text("offset y"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut offset_z, -10.0..=10.0).text("offset z"),
                                );
                            })
                        });
                    });

                unsafe {
                    OFFSET_X = offset_x;
                    OFFSET_Y = offset_y;
                    OFFSET_Z = offset_z;
                }

                let delta_time = self.delta_time.as_secs_f32();
                let world = &mut self.world;
                player_update(world, delta_time);
                physics_update(world, delta_time);
                camera_update(world, delta_time, Vec3::new(offset_x, offset_y, offset_z));
                // camera_update(world, delta_time);

                // reset mouse movment after camera handles it
                self.world.get_mut_resource::<MouseState>().unwrap().x = 0.0;
                self.world.get_mut_resource::<MouseState>().unwrap().y = 0.0;

                self.renderer.as_mut().unwrap().ui.full_output = Some(full_output);
                self.renderer.as_mut().unwrap().draw2(&mut self.world);
                self.delta_time = self.prev_frame_end.elapsed();
                // println!("\x1b[H\x1b[J");
                // eprintln!(
                //     "fps for frame {} is {}",
                //     self.current_frame,
                //     1.0 / self.delta_time.as_secs_f32()
                // );

                self.current_frame += 1;
                self.prev_frame_end = Instant::now();
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
                            keyboard.insert(code);
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
                            keyboard.remove(&code);
                        }
                        PhysicalKey::Unidentified(_) => {}
                    }
                }
                if release_mouse {
                    self.world.get_mut_resource::<MouseState>().unwrap().locked = false;
                }
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .unwrap();
                window.set_cursor_visible(false);
                self.world.get_mut_resource::<MouseState>().unwrap().locked = true;
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
            let state = self.world.get_mut_resource::<MouseState>().unwrap();
            if state.locked {
                state.x = delta.0 as f32;
                state.y = delta.1 as f32;
            }
        }
    }
}

fn physics_update(world: &mut World, delta_time: f32) {
    //NOTE: MOVEMENT INTEGRATION
    let mut rigidbodies = world.query_mut::<(ReqM<Transform>, ReqM<RigidBody>)>();
    for (_, (transform, rigidbody)) in rigidbodies {
        //NOTE:COMPUTES THE HORIZONTAL SPEED REDUCTION
        let horizontal_velocity = Vec3::new(rigidbody.velocity.x, 0.0, rigidbody.velocity.z);
        let horizontal_friction: f32 = 20.0; // how many units of speed to remove per second
        if horizontal_velocity.mag() > 0.0 {
            // compute how much to drop this frame:
            let frame_decel = horizontal_friction * delta_time;

            if horizontal_velocity.mag() <= frame_decel {
                rigidbody.velocity.x = 0.0;
                rigidbody.velocity.z = 0.0;
            } else {
                let direction = horizontal_velocity.normalized();
                rigidbody.velocity -= direction * frame_decel;
            }
        }
        // NOTE: APPLIES GRAVITY, GROUNDED FLAGS DONT MATTER BECAUSE WILL BE CORRECTED FOR IN
        // RESOLUTION FOR POSITION AND THEN VELOCITY
        rigidbody.velocity.y -= 9.8 * delta_time;

        let velocity = rigidbody.velocity;
        transform.position += velocity * delta_time;
    }

    let mut collidable: Vec<_> = world
        .query_mut::<(ReqM<Transform>, ReqM<Collider>, OptM<RigidBody>)>()
        .collect();
    let mut collision_penetrations = Vec::new();

    //NOTE: COLLISION RESOLUTION
    for i in 0..collidable.len() {
        for j in i + 1..collidable.len() {
            match Collider::penetration_vector(
                collidable[i].1.1,
                collidable[j].1.1,
                collidable[i].1.0,
                collidable[j].1.0,
            ) {
                Some(pen_vec) => {
                    if let Some(body1) = &collidable[i].1.2
                        && let Some(body2) = &collidable[j].1.2
                    {
                        collidable[i].1.0.position += pen_vec * 0.5;
                        collision_penetrations.push((collidable[i].0, pen_vec * 0.5));
                        collidable[j].1.0.position -= pen_vec * 0.5;
                        collision_penetrations.push((collidable[j].0, pen_vec * -0.5));
                    } else {
                        if let Some(body) = &collidable[i].1.2 {
                            collidable[i].1.0.position += pen_vec;
                            collision_penetrations.push((collidable[i].0, pen_vec));
                        }
                        if let Some(body) = &collidable[j].1.2 {
                            collidable[j].1.0.position -= pen_vec;
                            collision_penetrations.push((collidable[j].0, pen_vec * -1.0));
                        }
                    }
                }
                None => (),
            }
        }
    }
    //NOTE: VELOCITY UPDATE
    for (entity, pen_vec) in collision_penetrations {
        let rigidbody = world
            .get_mut::<(RigidBody,)>(entity)
            .expect("thing with collision regesterd should have rigidbody");
        let mut restitution = 0.3;
        if rigidbody.velocity.y.abs() < 0.5 {
            restitution = 0.0;
        }
        rigidbody.velocity = set_axis_component(rigidbody.velocity, pen_vec, restitution);
    }
}

// accepts a velocity, a minimum vector to resolve the collision, a coefficient of restitution, and returns the new
// velocity
fn set_axis_component(velocity: Vec3, collision_vector: Vec3, restitution: f32) -> Vec3 {
    let collision = collision_vector.normalized();
    let projection = collision * velocity.dot(collision);
    return velocity - (1.0 + restitution) * projection;
}
fn camera_update(world: &mut World, _delta_time: f32, offset: Vec3) {
    let mouse = *world.get_resource::<MouseState>().unwrap();
    let player = world.query::<(Controllable,)>().next().unwrap().0;
    let player_transform = world.get::<(Transform,)>(player).unwrap().clone();
    let camera = world.get_mut_resource::<Camera>().unwrap();

    let sensativity = 0.002;
    camera.pitch += mouse.y * sensativity;
    camera.yaw -= mouse.x * sensativity;
    if camera.pitch / PI < -0.499999 {
        camera.pitch = -(PI / 2.0) + 0.000001;
    }
    if camera.pitch / PI > 0.500001 {
        camera.pitch = (PI / 2.0) - 0.000001;
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

    let camera_relative_offset = camera.rotation * offset;
    let target = player_transform.position + camera_relative_offset;

    let offset = backward * target_distance;
    camera.position = target + offset;
}

fn player_update(world: &mut World, delta_time: f32) {
    let keyboard = world
        .get_resource::<Keyboard>()
        .expect("keyboard should have been added during resumed")
        .clone();

    let mut delta_v = Vec3::zero();
    let mut jump = 0.0;
    let mut max_speed = 7.0;
    if keyboard.contains(&KeyCode::ShiftLeft) {
        max_speed = 20.0;
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

    let camera_rotation = world.get_resource::<Camera>().unwrap().rotation;
    let mut binding = world.query_mut::<(ReqM<Controllable>, ReqM<Transform>, ReqM<RigidBody>)>();
    let (_, (_, transform, rigidbody)) = binding.next().expect("player should exist!");

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
    if transform.position.y < -50.0 {
        transform.position.y = 20.0;
    }
    rigidbody.velocity.y += jump;
}
