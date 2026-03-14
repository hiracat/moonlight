use std::f32::consts::PI;

use moonlight::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Transform},
    core::{Game, Keyboard, MouseState},
    ecs::{OptM, Req, ReqM, World},
    physics::{Aabb, Collider, RigidBody},
    resources::{Material, Skybox},
};
use ultraviolet::{Rotor3, Slerp, Vec3, Vec4};
use winit::keyboard::KeyCode;

// ai generated test game, will be replaced, just wanted an interesting test scene to have
#[derive(Default)]
pub struct GameImpl {
    pub camera_offset: Vec3,
    pub time: f32,
    pub gem_collected: bool,
    pub platform_entities: Vec<moonlight::ecs::EntityId>,
    pub gem_entity: moonlight::ecs::EntityId,
}
impl Game for GameImpl {
    fn on_close(&mut self, world: &mut World, engine: &mut moonlight::core::Engine) {}
    fn on_start(
        &mut self,
        world: &mut moonlight::ecs::World,
        engine: &mut moonlight::core::Engine,
    ) {
        let (width, height) = engine.window_size;
        let camera = Camera::create(
            Vec3::new(0.0, 5.0, -10.0),
            60.0,
            1.0,
            200.0,
            height as f32 / width as f32,
        );
        let directional =
            DirectionalLight::create(Vec4::new(200.0, 10.0, 0.0, 1.0), Vec3::new(0.6, 0.2, 0.2));
        let ambient = AmbientLight::create(Vec3::new(1.0, 1.0, 0.8), 0.05);

        world.name("platformer");
        *world.get_mut_resource().unwrap() = camera;
        world.add_resource(directional).unwrap();
        world.add_resource(ambient).unwrap();

        let skybox = engine.resource_manager.create_cubemap([
            "data/skybox/px.png",
            "data/skybox/nx.png",
            "data/skybox/py.png",
            "data/skybox/ny.png",
            "data/skybox/pz.png",
            "data/skybox/nz.png",
        ]);
        world.add_resource(Skybox::new(skybox)).unwrap();

        // ── Fox ──────────────────────────────────────────────────
        let fox = world.spawn();
        world
            .add(
                fox,
                Transform::from(
                    Some(Vec3::new(0.0, 1.0, 0.0)),
                    Some(Rotor3::identity()),
                    Some(Vec3::new(1.0, 1.0, 1.0)),
                ),
            )
            .unwrap();
        world.add(fox, Controllable).unwrap();
        world
            .add(
                fox,
                Collider::Aabb(Aabb::new(
                    Vec3::new(0.4, 0.588, 0.4),
                    Vec3::new(0.0, 0.588, 0.0),
                )),
            )
            .unwrap();
        world.add(fox, RigidBody::new()).unwrap();

        let (fox_model, fox_animations) = engine
            .resource_manager
            .load_gltf_asset("data/models/animated_fox.glb");
        let mut fox_animations = fox_animations.unwrap();
        fox_animations.current_playing = Some(fox_animations.animations[0].clone());
        world.add(fox, fox_model).unwrap();
        world.add(fox, fox_animations).unwrap();
        let fox_albedo = engine
            .resource_manager
            .create_texture("data/models/textures/animated_fox_texture.png");
        world.add(fox, Material::create(fox_albedo)).unwrap();

        // ── Ground ───────────────────────────────────────────────
        let ground = world.spawn();
        let ground_tex = engine
            .resource_manager
            .create_texture("data/models/textures/ground_roots.png");
        world.add(ground, Material::create(ground_tex)).unwrap();
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
        world
            .add(
                ground,
                Collider::Aabb(Aabb::new(
                    Vec3::new(1.0, 4.0, 1.0),
                    Vec3::new(0.0, -4.0, 0.0),
                )),
            )
            .unwrap();

        // ── Platforms ────────────────────────────────────────────
        // Each entry: (x, base_y, z, bob_speed, bob_phase, bob_amplitude)
        let platform_defs: &[(f32, f32, f32, f32, f32, f32)] = &[
            (3.0, 2.0, 3.0, 1.0, 0.0, 0.8),   // easy first hop
            (6.0, 4.0, 5.0, 1.3, 1.0, 1.2),   // starts rising
            (9.0, 5.0, 3.0, 0.8, 2.5, 1.5),   // slow, high amplitude
            (12.0, 7.0, 1.0, 1.8, 0.5, 0.6),  // fast, small bob
            (10.0, 9.0, -3.0, 1.1, 3.0, 1.0), // turn left
            (7.0, 11.0, -6.0, 0.6, 1.5, 1.8), // big slow swing
            (4.0, 13.0, -4.0, 2.0, 0.0, 0.5), // fast small
            (1.0, 15.0, -2.0, 1.0, 4.0, 1.0), // final approach
        ];

        let cube = engine
            .resource_manager
            .load_gltf_asset("data/models/large_cube.glb")
            .0;
        let platform_tex = engine
            .resource_manager
            .create_texture("data/models/textures/ground.jpg");

        for &(x, base_y, z, _speed, _phase, _amp) in platform_defs {
            let e = world.spawn();
            world
                .add(
                    e,
                    Transform::from(
                        Some(Vec3::new(x, base_y, z)),
                        None,
                        Some(Vec3::new(2.5, 0.3, 2.5)),
                    ),
                )
                .unwrap();
            world
                .add(
                    e,
                    Collider::Aabb(Aabb::new(Vec3::new(1.0, 0.15, 1.0), Vec3::zero())),
                )
                .unwrap();
            world.add(e, cube).unwrap();
            world.add(e, Material::create(platform_tex)).unwrap();
            self.platform_entities.push(e);
        }

        // ── Gem (goal) ───────────────────────────────────────────
        let gem = world.spawn();
        world
            .add(
                gem,
                Transform::from(
                    Some(Vec3::new(1.0, 17.0, -2.0)),
                    None,
                    Some(Vec3::new(0.5, 0.5, 0.5)),
                ),
            )
            .unwrap();
        let gem_model = engine
            .resource_manager
            .load_gltf_asset("data/models/diamond.glb")
            .0;
        let gem_tex = engine
            .resource_manager
            .create_texture("data/models/textures/diamond.png");
        world.add(gem, gem_model).unwrap();
        world.add(gem, Material::create(gem_tex)).unwrap();
        self.gem_entity = gem;

        // ── Lights ───────────────────────────────────────────────
        let blue_light = world.spawn();
        world
            .add(
                blue_light,
                PointLight::new(Vec3::new(0.1, 0.4, 1.0), 60.0, Some(1.2), Some(0.4)),
            )
            .unwrap();
        world
            .add(
                blue_light,
                Transform::from(Some(Vec3::new(1.0, 18.0, -2.0)), None, None),
            )
            .unwrap();

        let warm_light = world.spawn();
        world
            .add(
                warm_light,
                PointLight::new(Vec3::new(1.0, 0.6, 0.2), 40.0, None, Some(1.5)),
            )
            .unwrap();
        world
            .add(
                warm_light,
                Transform::from(Some(Vec3::new(0.0, 3.0, 0.0)), None, None),
            )
            .unwrap();
    }
    fn on_update(
        &mut self,
        world: &mut moonlight::ecs::World,
        engine: &mut moonlight::core::Engine,
        _delta_time: f32,
    ) {
        let delta_time = engine.delta_time;
        self.time += delta_time;

        player_update(world, delta_time);
        physics_update(world, delta_time);
        camera_update(world, delta_time, self.camera_offset);

        // ── Bob platforms ────────────────────────────────────────
        let platform_defs: &[(f32, f32, f32, f32, f32, f32)] = &[
            (3.0, 2.0, 3.0, 1.0, 0.0, 0.8),
            (6.0, 4.0, 5.0, 1.3, 1.0, 1.2),
            (9.0, 5.0, 3.0, 0.8, 2.5, 1.5),
            (12.0, 7.0, 1.0, 1.8, 0.5, 0.6),
            (10.0, 9.0, -3.0, 1.1, 3.0, 1.0),
            (7.0, 11.0, -6.0, 0.6, 1.5, 1.8),
            (4.0, 13.0, -4.0, 2.0, 0.0, 0.5),
            (1.0, 15.0, -2.0, 1.0, 4.0, 1.0),
        ];

        for (i, &entity) in self.platform_entities.iter().enumerate() {
            let (x, base_y, z, speed, phase, amp) = platform_defs[i];
            let new_y = base_y + (self.time * speed + phase).sin() * amp;

            if let Ok(transform) = world.get_mut::<(Transform,)>(entity) {
                transform.position = Vec3::new(x, new_y, z);
            }
        }

        // ── Spin gem and check collection ────────────────────────
        if !self.gem_collected {
            if let Ok(gem_transform) = world.get_mut::<(Transform,)>(self.gem_entity) {
                // spin
                gem_transform.rotation = Rotor3::from_rotation_xz(self.time * 2.0);

                // check if fox is close enough
                let gem_pos = gem_transform.position;
                // get fox position via query
            }

            // need fox pos separately since we already borrowed gem
            let mut fox_pos = Vec3::zero();
            for (_, (t, c)) in world.query::<(Req<Transform>, Req<Controllable>)>() {
                fox_pos = t.position;
            }
            if let Ok(gem_transform) = world.get_mut::<(Transform,)>(self.gem_entity) {
                let dist = (gem_transform.position - fox_pos).mag();
                if dist < 1.5 {
                    self.gem_collected = true;
                    // hide gem by scaling to zero
                    gem_transform.scale = Vec3::zero();
                    println!("🦊 Gem collected! You win!");
                }
            }
        }

        world.get_mut_resource::<MouseState>().unwrap().x = 0.0;
        world.get_mut_resource::<MouseState>().unwrap().y = 0.0;
    }
    fn on_ui(&mut self, _world: &mut moonlight::ecs::World, context: &egui::Context) {
        egui::CentralPanel::default().show(context, |_ui| {
            if self.gem_collected {
                egui::Window::new("🦊 You Win!")
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(context, |ui| {
                        ui.heading("You collected the gem!");
                        ui.label("Jump across the platforms to reach the top.");
                    });
            } else {
                egui::Window::new("Goal").show(context, |ui| {
                    ui.label("Reach the glowing gem at the top!");
                });
            }

            egui::Window::new("Camera").show(context, |ui| {
                ui.add(egui::Slider::new(&mut self.camera_offset.x, -10.0..=10.0).text("x"));
                ui.add(egui::Slider::new(&mut self.camera_offset.y, -10.0..=10.0).text("y"));
                ui.add(egui::Slider::new(&mut self.camera_offset.z, -10.0..=10.0).text("z"));
            });
        });
    }
}

// this on down is human written code though
#[derive(Debug)]
struct Controllable;

fn physics_update(world: &mut World, delta_time: f32) {
    //NOTE: MOVEMENT INTEGRATION
    let rigidbodies = world.query_mut::<(ReqM<Transform>, ReqM<RigidBody>)>();
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
                collidable[i].1 .1,
                collidable[j].1 .1,
                collidable[i].1 .0,
                collidable[j].1 .0,
            ) {
                Some(pen_vec) => {
                    if collidable[i].1 .2.is_some() && collidable[j].1 .2.is_some() {
                        collidable[i].1 .0.position += pen_vec * 0.5;
                        collision_penetrations.push((collidable[i].0, pen_vec * 0.5));
                        collidable[j].1 .0.position -= pen_vec * 0.5;
                        collision_penetrations.push((collidable[j].0, pen_vec * -0.5));
                    } else {
                        if collidable[i].1 .2.is_some() {
                            collidable[i].1 .0.position += pen_vec;
                            collision_penetrations.push((collidable[i].0, pen_vec));
                        }
                        if collidable[j].1 .2.is_some() {
                            collidable[j].1 .0.position -= pen_vec;
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
    if keyboard.is_down(KeyCode::ShiftLeft) {
        max_speed = 20.0;
    }
    if keyboard.is_down(KeyCode::KeyW) {
        delta_v += Vec3::new(0.0, 0.0, -1.0);
    }
    if keyboard.is_down(KeyCode::KeyS) {
        delta_v += Vec3::new(0.0, 0.0, 1.0);
    }
    if keyboard.is_down(KeyCode::KeyD) {
        delta_v += Vec3::new(1.0, 0.0, 0.0);
    }
    if keyboard.is_down(KeyCode::KeyA) {
        delta_v += Vec3::new(-1.0, 0.0, 0.0);
    }

    let camera_rotation = world.get_resource::<Camera>().unwrap().rotation;
    let mut binding = world.query_mut::<(ReqM<Controllable>, ReqM<Transform>, ReqM<RigidBody>)>();
    let (_, (_, transform, rigidbody)) = binding.next().expect("player should exist!");

    // BUG: allows the player to jump at the apex of their jump, but the period of time which the
    // bug is viable to exploit is almost zero so not going to fix
    if keyboard.is_down(KeyCode::Space) && rigidbody.velocity.y.abs() < 0.00001 {
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
