use std::{collections::HashSet, f32::consts::PI};

use moonlight::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    ecs::{OptM, ReqM, World},
    physics::{Aabb, Collider, RigidBody},
    resources::{Material, Skybox},
};
use ultraviolet::{Rotor3, Slerp, Vec3, Vec4};
use winit::keyboard::KeyCode;

pub struct GameImpl {}

impl moonlight::core::Game for GameImpl {
    fn on_ui(&mut self, world: &mut moonlight::ecs::World, context: &egui::Context) {
        let mut offset_x = unsafe { OFFSET_X };
        let mut offset_y = unsafe { OFFSET_Y };
        let mut offset_z = unsafe { OFFSET_Z };
        egui::CentralPanel::default().show(context, |ui| {
            egui::Window::new("Camera Offset").show(context, |ui| {
                ui.add(egui::Slider::new(&mut offset_x, -10.0..=10.0).text("offset x"));
                ui.add(egui::Slider::new(&mut offset_y, -10.0..=10.0).text("offset y"));
                ui.add(egui::Slider::new(&mut offset_z, -10.0..=10.0).text("offset z"));
            })
        });
        unsafe {
            OFFSET_X = offset_x;
            OFFSET_Y = offset_y;
            OFFSET_Z = offset_z;
        }
    }
    fn on_update(
        &mut self,
        world: &mut moonlight::ecs::World,
        engine: &mut moonlight::core::Engine,
        delta_time: f32,
    ) {
        let offset_x = unsafe { OFFSET_X };
        let offset_y = unsafe { OFFSET_Y };
        let offset_z = unsafe { OFFSET_Z };

        let delta_time = engine.delta_time;
        player_update(world, delta_time);
        physics_update(world, delta_time);
        camera_update(world, delta_time, Vec3::new(offset_x, offset_y, offset_z));

        // reset mouse movment after camera handles it
        world.get_mut_resource::<MouseState>().unwrap().x = 0.0;
        world.get_mut_resource::<MouseState>().unwrap().y = 0.0;
    }
    fn on_close(
        &mut self,
        world: &mut moonlight::ecs::World,
        engine: &mut moonlight::core::Engine,
    ) {
    }
    fn on_start(
        &mut self,
        world: &mut moonlight::ecs::World,
        engine: &mut moonlight::core::Engine,
    ) {
        // ───────────────────────────────────────────────────────
        // 2) Create "static" resources (camera, lights, input, etc.)
        // ───────────────────────────────────────────────────────
        // Camera at (0, 5, -6), fov=60°, near=1.0, far=100.0
        let (width, height) = engine.window_size;
        let camera = Camera::create(
            Vec3::new(0.0, 5.0, -6.0),
            60.0,
            1.0,
            200.0,
            height as f32 / width as f32,
        );
        // Sun (directional) and ambient light
        let directional =
            DirectionalLight::create(Vec4::new(200.0, 10.0, 0.0, 1.0), Vec3::new(0.2, 0.2, 0.2));
        let ambient = AmbientLight::create(Vec3::new(1.0, 1.0, 1.0), 0.05);

        // Keyboard + mouse input resources
        let keyboard = Keyboard::new();
        let mouse_movement = MouseState::default();

        // Register resources into the world
        world.add_resource(camera).unwrap();
        world.add_resource(directional).unwrap();
        world.add_resource(ambient).unwrap();
        world.add_resource(keyboard).unwrap();
        world.add_resource(mouse_movement).unwrap();

        let skybox = engine.resource_manager.create_cubemap([
            "data/skybox/px.png",
            "data/skybox/nx.png",
            "data/skybox/py.png",
            "data/skybox/ny.png",
            "data/skybox/pz.png",
            "data/skybox/nz.png",
        ]);
        let skybox = Skybox::new(skybox);
        world.add_resource(skybox).unwrap();

        // ───────────────────────────────────────────────────────
        // 3) Spawn entities
        // ───────────────────────────────────────────────────────
        let fox = world.spawn();
        let ground = world.spawn();
        let red_light = world.spawn();
        let blue_light = world.spawn();
        let green_light = world.spawn();
        let x_axis = world.spawn();
        let y_axis = world.spawn();
        let z_axis = world.spawn();
        let standing_block = world.spawn();

        // ───────────────────────────────────────────────────────
        // THE FOX - Center stage, elevated on a mystical platform
        // ───────────────────────────────────────────────────────
        world
            .add(
                fox,
                Transform::from(
                    Some(Vec3::new(0.0, 3.0, 0.0)), // Elevated at center
                    Some(Rotor3::identity()),        // Slight rotation for dramatic pose
                    Some(Vec3::new(1.0, 1.0, 1.0)),
                ),
            )
            .unwrap();
        world.add(fox, Controllable).unwrap();

        // Fox collision - smaller and more precise
        let fox_half_extents = Vec3::new(0.4, 0.588, 0.4);
        let fox_center_offset = Vec3::new(0.0, 0.588, 0.0);
        world
            .add(
                fox,
                Collider::Aabb(Aabb::new(fox_half_extents, fox_center_offset)),
            )
            .unwrap();
        world.add(fox, RigidBody::new()).unwrap();
        let (fox_model, fox_animations) = engine
            .resource_manager
            .load_gltf_asset("data/models/animated_fox.slb");
        let mut fox_animations = fox_animations.unwrap();
        fox_animations.current_playing = Some(fox_animations.animations[0].clone());

        world.add(fox, fox_model).unwrap();
        world.add(fox, fox_animations).unwrap();

        let albedo = engine
            .resource_manager
            .create_texture("data/models/textures/animated_fox_texture.png");

        world.add(fox, Material::create(albedo)).unwrap();

        let albedo = engine
            .resource_manager
            .create_texture("data/models/textures/ground.jpg");
        world.add(ground, Material::create(albedo)).unwrap();
        world.add(x_axis, Material::create(albedo)).unwrap();
        world.add(y_axis, Material::create(albedo)).unwrap();
        world.add(z_axis, Material::create(albedo)).unwrap();
        let albedo = engine
            .resource_manager
            .create_texture("data/models/textures/ground_soft.jpg");
        world
            .add(standing_block, Material::create(albedo))
            .unwrap();

        world
            .add(
                ground,
                Transform::from(None, None, Some(Vec3::new(100.0, 1.0, 100.0))),
            )
            .unwrap();
        world
            .add(
                ground,
                engine
                    .resource_manager
                    .load_gltf_asset("data/models/ground_plane.glb")
                    .0,
            )
            .unwrap();

        // Ground collision - the full 40x40 area with some depth
        world
            .add(
                ground,
                Collider::Aabb(Aabb::new(
                    Vec3::new(1.0, 4.0, 1.0),  // Half-extents: 40x2x40 total size
                    Vec3::new(0.0, -4.0, 0.0), // Center offset: buried 1 unit down
                )),
            )
            .unwrap();

        world
            .add(
                x_axis,
                Transform::from(
                    Some(Vec3::new(8.0, 0.0, 8.0)), // High and forward-right
                    None,
                    Some(Vec3::new(10.0, 9.0, 10.0)),
                ),
            )
            .unwrap();
        world
            .add(
                y_axis,
                Transform::from(
                    Some(Vec3::new(-12.0, 0.0, -3.0)), // Tallest, to the left and slightly back
                    None,
                    Some(Vec3::new(2.0, 2.0, 2.0)),
                ),
            )
            .unwrap();
        world
            .add(
                z_axis,
                Transform::from(
                    Some(Vec3::new(5.0, 0.0, -10.0)), // Behind the fox, watching over
                    None,
                    Some(Vec3::new(3.0, 3.0, 3.0)),
                ),
            )
            .unwrap();

        world
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
        world.add(x_axis, cube_collider).unwrap();
        world.add(y_axis, cube_collider).unwrap();
        world.add(z_axis, cube_collider).unwrap();
        world.add(standing_block, cube_collider).unwrap();

        let cube = engine
            .resource_manager
            .load_gltf_asset("data/models/large_cube.glb")
            .0;
        // Attach models to monuments
        world.add(x_axis, cube).unwrap();
        world.add(y_axis, cube).unwrap();
        world.add(z_axis, cube).unwrap();
        world.add(standing_block, cube).unwrap();

        world
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
        world
            .add(
                red_light,
                Transform::from(Some(Vec3::new(10.0, 8.0, 6.0)), None, None), // Near the red monument, but higher
            )
            .unwrap();

        world
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
        world
            .add(
                green_light,
                Transform::from(Some(Vec3::new(-10.0, 7.0, -2.0)), None, None), // Left side, near green monument
            )
            .unwrap();

        world
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
        world
            .add(
                blue_light,
                Transform::from(Some(Vec3::new(2.0, 5.0, -12.0)), None, None), // Behind fox, creating rim light
            )
            .unwrap();
    }
}

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
                    if collidable[i].1.2.is_some() && collidable[j].1.2.is_some() {
                        collidable[i].1.0.position += pen_vec * 0.5;
                        collision_penetrations.push((collidable[i].0, pen_vec * 0.5));
                        collidable[j].1.0.position -= pen_vec * 0.5;
                        collision_penetrations.push((collidable[j].0, pen_vec * -0.5));
                    } else {
                        if collidable[i].1.2.is_some() {
                            collidable[i].1.0.position += pen_vec;
                            collision_penetrations.push((collidable[i].0, pen_vec));
                        }
                        if collidable[j].1.2.is_some() {
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
