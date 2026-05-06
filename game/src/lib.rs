use moonlight::animations::PlaybackMode;
use moonlight::ecs;
use moonlight::lua;
use moonlight::resources::Texture;
use proc_macros::LuaUnion;
use proc_macros::LuaVal;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::OnceLock;
use ultraviolet::Bivec3;

use image::ImageReader;
use moonlight::{
    components::{AmbientLight, Camera, DirectionalLight, PointLight, Time, Transform},
    core::{App, Controllable, Engine, Keyboard, MouseState, System, TerrainMap},
    ecs::{OptM, ReqM, World},
    physics::{Collider, Obb, RigidBody},
    resources::{self, Material, Skybox},
};
use proc_macros::LuaRef;
use tracing::{trace, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use ultraviolet::{Rotor3, Slerp, Vec3};
use winit::keyboard::KeyCode;

type FilterHandle = tracing_subscriber::reload::Handle<
    EnvFilter,
    tracing_subscriber::layer::Layered<
        tracing_subscriber::fmt::Layer<tracing_subscriber::Registry>,
        tracing_subscriber::Registry,
    >,
>;
pub static FILTER_HANDLE: OnceLock<FilterHandle> = OnceLock::new();

// ── Resources ────────────────────────────────────────────

#[derive(Debug, LuaRef, Default, Clone)]
struct CameraOffset {
    offset: Vec3,
}
fn init_tracing() {
    let filter = EnvFilter::new("trace");

    let (filter_layer, handle) = tracing_subscriber::reload::Layer::new(filter);

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter_layer)
        .init();

    FILTER_HANDLE.set(handle).ok();
}

pub fn setup_game() -> App {
    init_tracing();
    let mut app = App::new("data/scripts/main.lua");
    app.game.on_start.push(System::Rust(start));
    app.game.on_update.push(System::Rust(game_update));
    app.game.on_update.push(System::Lua("Update".to_string()));
    app.game.on_ui.push(ui);
    app
}

#[derive(LuaRef, Clone, Copy)]
struct Config {
    gravity_strength: f32,
}
impl Default for Config {
    fn default() -> Self {
        Config {
            gravity_strength: 9.8,
        }
    }
}

#[derive(LuaRef, Clone, Debug)]
pub struct TracingConfig {
    pub level: String,
    pub player: Option<String>,
    pub physics: Option<String>,
    pub pipeline: Option<String>,
    pub ui: Option<String>,
    pub gpu: Option<String>,
}
impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            player: None,
            physics: None,
            pipeline: None,
            ui: None,
            gpu: None,
        }
    }
}

fn start(world: &mut World, engine: &mut Engine) {
    let (width, height) = engine.window_size;
    let camera = Camera::create(
        Vec3::new(0.0, 5.0, -10.0),
        60.0,
        0.1,
        200.0,
        height as f32 / width as f32,
    );
    let directional =
        DirectionalLight::create(Vec3::new(200.0, 10.0, 0.0), Vec3::new(2.0, 1.6, 1.5));
    let ambient = AmbientLight::create(Vec3::new(1.0, 1.0, 0.8), 0.35);

    let heightmap = TerrainMap {
        map: engine
            .resource_manager
            .create_texture("data/heightmap.exr", resources::TextureFormat::HeightF32),
        cpu_map: ImageReader::open("data/heightmap.exr")
            .unwrap()
            .decode()
            .unwrap()
            .to_luma32f(),
        size: 2000.0,
        height: 300.0,
        resolution: 1000,
    };

    world.add_resource(UIStuff::default()).unwrap();
    world.add_resource(heightmap).unwrap();
    let config = Config::default();
    world.add_resource(config).unwrap();

    *world.get_mut_resource().unwrap() = camera;
    world.add_resource(directional).unwrap();
    world.add_resource(ambient).unwrap();
    world
        .add_resource(CameraOffset {
            offset: Vec3::zero(),
        })
        .unwrap();

    let skybox = engine.resource_manager.create_cubemap(&[
        "data/skybox/px.png",
        "data/skybox/nx.png",
        "data/skybox/py.png",
        "data/skybox/ny.png",
        "data/skybox/pz.png",
        "data/skybox/nz.png",
    ]);
    world.add_resource(Skybox::new(skybox)).unwrap();

    // fox — stays Rust because Controllable, Collider, RigidBody, Animated
    // are all engine-side components that Lua can't fully construct yet
    let fox = world.spawn("fox");
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
    world
        .add(
            fox,
            Controllable {
                speed: 7.0,
                sprint_speed: 20.0,
            },
        )
        .unwrap();
    world
        .add(
            fox,
            Collider::Obb(Obb::new(Vec3::new(0.8, 3.0, 4.5), Vec3::new(0.0, 1.5, 0.0))),
        )
        .unwrap();
    world.add(fox, RigidBody::new()).unwrap();
    let (fox_model, fox_animations) = engine
        .resource_manager
        .load_gltf_asset("data/models/animated_fox.glb");
    let mut fox_animations = fox_animations.unwrap();
    fox_animations.mode = PlaybackMode::Loop(fox_animations.available_animations[0]);
    // fox_animations.current_playing = None;
    world.add(fox, fox_model).unwrap();
    world.add(fox, fox_animations).unwrap();
    let fox_albedo = engine.resource_manager.create_texture(
        "data/models/textures/animated_fox_texture.png",
        resources::TextureFormat::Srgba,
    );
    world.add(fox, Material::create(fox_albedo, None)).unwrap();

    let ground = world.spawn("ground");
    let ground_tex = engine.resource_manager.create_texture(
        "data/models/textures/ground_roots.png",
        resources::TextureFormat::Srgba,
    );
    world
        .add(ground, Material::create(ground_tex, None))
        .unwrap();
    world
        .add(
            ground,
            Transform::from(None, None, Some(Vec3::new(1000.0, 1.0, 1000.0))),
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
            Collider::Obb(Obb::new(
                Vec3::new(2.0, 8.0, 2.0),
                Vec3::new(0.0, -4.0, 0.0),
            )),
        )
        .unwrap();

    // lights — could move to Lua but no reason to
    let blue_light = world.spawn("blue_light");
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

    let warm_light = world.spawn("warm_light");
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

    let tree = world.spawn("tree");
    world
        .add(
            tree,
            Transform::from(Some(Vec3::new(0.0, 150.0, 0.0)), None, None),
        )
        .unwrap();
    world
        .add(
            tree,
            engine
                .resource_manager
                .load_gltf_asset("data/models/maple_tree.glb")
                .0,
        )
        .unwrap();
    world
        .add(
            tree,
            Material {
                albedo: Texture::default(),
                alpha_clip: Some(0.5),
            },
        )
        .unwrap();
    world.add_resource(TracingConfig::default()).unwrap();
}

fn game_update(world: &mut World, _engine: &mut Engine) {
    let delta_time = world.get_resource::<Time>().unwrap().delta_time;
    player_update(world, delta_time);
    physics_update(world, delta_time);
    let camera_offset = world.get_resource::<CameraOffset>().unwrap().offset;
    camera_update(world, delta_time, camera_offset);
}

#[derive(LuaVal, Default, Clone, Debug)]
pub struct Slider {
    value: f32,
    min: f32,
    max: f32,
    label: Option<String>,
}

#[derive(LuaVal, Default, Clone, Debug)]
pub struct Button {
    clicked: bool,
    label: Option<String>,
}

#[derive(LuaVal, Default, Clone, Debug)]
pub struct Label {
    text: String,
}
#[derive(LuaVal, Default, Clone, Debug)]
pub struct TextInput {
    value: String,
    label: Option<String>,
}
#[derive(LuaVal, Default, Clone, Debug)]
pub struct NumberInput {
    value: f32,
    min: f32,
    max: f32,
    label: Option<String>,
}

#[derive(LuaRef, Clone, Default, Debug)]
#[lua(no_default)]
struct UIStuff {
    // path to widget
    widgets: HashMap<String, Widget>,
    schema: UISchema,
    show_settings: bool,
}

#[derive(LuaVal, Default, Clone, Debug)]
struct UISchema {
    windows: Vec<WindowSchema>,
}

#[derive(LuaVal, Default, Clone, Debug)]
struct WindowSchema {
    name: String,
    fields: Vec<LayoutItem>,
}
#[derive(LuaUnion, Clone, Hash, PartialEq, Eq, Debug)]
#[lua(no_default)]
pub enum LayoutItem {
    Field(String),
    Row(Row),
    Column(Collumn),
    Scroll(ScrollArea),
}
#[derive(LuaVal, Debug, Clone, Hash, PartialEq, Eq)]
#[lua(no_default)]
pub struct Row {
    items: Vec<LayoutItem>,
}
#[derive(LuaVal, Clone, Hash, PartialEq, Eq, Debug)]
#[lua(no_default)]
pub struct Collumn {
    items: Vec<LayoutItem>,
}
#[derive(LuaVal, Clone, Hash, PartialEq, Eq, Debug)]
#[lua(no_default)]
pub struct ScrollArea {
    items: Vec<LayoutItem>,
    visible_lines: usize,
}

#[derive(LuaUnion, Clone, Debug)]
#[lua(no_default)]
pub enum Widget {
    Slider(Slider),
    Button(Button),
    Label(Label),
    TextInput(TextInput),
    NumberInput(NumberInput),
    Separator,
}
impl Widget {
    fn label(&self) -> Option<&str> {
        match self {
            Widget::Slider(slider) => slider.label.as_deref(),
            Widget::Button(button) => button.label.as_deref(),
            Widget::Label(label) => Some(label.text.as_ref()),
            Widget::TextInput(text_input) => text_input.label.as_deref(),
            Widget::NumberInput(number_input) => number_input.label.as_deref(),
            Widget::Separator => None,
        }
    }
}

fn default_label(path: &str) -> &str {
    path.split(".").last().unwrap()
}

fn draw_widget(ui: &mut egui::Ui, path: &str, widget: &mut Widget) {
    let label = widget
        .label()
        // im not sure if i should filter out empty lables or leave that as an option
        //.filter(|&x| x.is_empty())
        .unwrap_or_else(|| default_label(path))
        .to_owned();

    match widget {
        Widget::Slider(s) => {
            ui.add(egui::Slider::new(&mut s.value, s.min..=s.max).text(label));
        }

        Widget::Button(b) => {
            if ui.button(label).clicked() {
                b.clicked = true;
            }
        }

        Widget::Label(v) => {
            ui.label(&v.text);
        }

        Widget::TextInput(t) => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.text_edit_singleline(&mut t.value);
            });
        }

        Widget::NumberInput(n) => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut n.value).range(n.min..=n.max));
            });
        }
        Widget::Separator => {
            ui.separator();
        }
    }
}
fn draw_item(ui: &mut egui::Ui, item: &LayoutItem, ui_stuff: &mut UIStuff) {
    match item {
        LayoutItem::Field(name) => {
            if let Some(widget) = ui_stuff.widgets.get_mut(name) {
                draw_widget(ui, name, widget);
            } else {
                warn!(name, "missing widget")
            }
        }
        LayoutItem::Row(row) => {
            ui.horizontal(|ui| {
                for item in &row.items {
                    draw_item(ui, item, ui_stuff);
                }
            });
        }

        LayoutItem::Column(col) => {
            ui.vertical(|ui| {
                for item in &col.items {
                    draw_item(ui, item, ui_stuff);
                }
            });
        }
        LayoutItem::Scroll(scroll) => {
            let area = egui::ScrollArea::both().max_height(
                ui.text_style_height(&egui::TextStyle::Body) * scroll.visible_lines as f32,
            );
            area.show(ui, |ui| {
                for item in &scroll.items {
                    draw_item(ui, item, ui_stuff);
                }
            });
        }
    }
}

fn ui(world: &mut World, context: &egui::Context) {
    if let Some(ui_stuff) = world.get_mut_resource::<UIStuff>() {
        if ui_stuff.show_settings {
            print!("{}", ui_stuff.show_settings);
            egui::Window::new("Settings").show(context, |ui| {
                context.settings_ui(ui);
            });
        };
        for item in ui_stuff.widgets.values_mut() {
            if let Widget::Button(button) = item {
                button.clicked = false;
            }
        }

        let windows = ui_stuff.schema.windows.clone();

        for window_schema in &windows {
            let window_name = &window_schema.name;

            egui::Window::new(window_name).show(context, |ui| {
                for item in &window_schema.fields {
                    draw_item(ui, item, ui_stuff);
                }
            });
        }
    }
    if let Some(cfg) = world.get_mut_resource::<TracingConfig>() {
        egui::Window::new("Tracing").show(context, |ui| {
            ui.heading("Global log level");
            ui.horizontal(|ui| {
                for level in ["trace", "debug", "info", "warn"] {
                    ui.selectable_value(&mut cfg.level, level.to_string(), level);
                }
            });

            ui.separator();
            ui.label("Subsystem overrides");

            let level = cfg.level.clone();
            for (label, field) in [
                ("player", &mut cfg.player),
                ("physics", &mut cfg.physics),
                ("pipeline", &mut cfg.pipeline),
                ("ui", &mut cfg.ui),
                ("gpu", &mut cfg.gpu),
            ] {
                ui.horizontal(|ui| {
                    let mut enabled = field.is_some();
                    if ui.checkbox(&mut enabled, label).changed() {
                        *field = enabled.then(|| level.clone());
                    }
                    if let Some(subsystem_level) = field {
                        for lvl in ["trace", "debug", "info", "warn", "off"] {
                            ui.selectable_value(subsystem_level, lvl.to_string(), lvl);
                        }
                    }
                });
            }
        });
    }
    if let Some(cfg) = world.get_resource::<TracingConfig>() {
        let overrides = [
            ("player", &cfg.player),
            ("physics", &cfg.physics),
            ("pipeline", &cfg.pipeline),
            ("ui", &cfg.ui),
            ("gpu", &cfg.gpu),
        ];

        let mut parts = vec![cfg.level.clone()];
        for (name, level) in overrides {
            if let Some(l) = level {
                parts.push(format!("{}={}", name, l));
            }
        }

        if let Some(handle) = FILTER_HANDLE.get() {
            let _ = handle.reload(EnvFilter::new(parts.join(",")));
        }
    }
}

// this on down is human written code though

fn physics_update(world: &mut World, delta_time: f32) {
    let config = *world.get_resource::<Config>().unwrap();
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
                let decel = direction * frame_decel;
                rigidbody.velocity.x -= decel.x;
                rigidbody.velocity.z -= decel.z;
            }
        }
        // NOTE: APPLIES GRAVITY, GROUNDED FLAGS DONT MATTER BECAUSE WILL BE CORRECTED FOR IN
        // RESOLUTION FOR POSITION AND THEN VELOCITY
        rigidbody.velocity.y -= config.gravity_strength * delta_time;

        let velocity = rigidbody.velocity;
        transform.position += velocity * delta_time;
    }

    let world_ptr = world as *mut World;
    let mut collidable: Vec<_> = world
        .query_mut::<(ReqM<Transform>, ReqM<Collider>, OptM<RigidBody>)>()
        .collect();
    let mut collision_penetrations = Vec::new();

    //NOTE: COLLISION RESOLUTION
    for i in 0..collidable.len() {
        for j in i + 1..collidable.len() {
            if let Some(pen_vec) = Collider::penetration_vector(
                collidable[i].1.1,
                collidable[j].1.1,
                collidable[i].1.0,
                collidable[j].1.0,
            ) {
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
        }
    }
    let terrain = unsafe { &mut *(world_ptr) }
        .get_resource::<TerrainMap>()
        .unwrap();
    for collider in collidable {
        let distance = collider.1.0.position.y
            - terrain.get_height_at(collider.1.0.position.x, collider.1.0.position.z);
        if distance < 0.0 && collider.1.2.is_some() {
            collider.1.0.position.y -= distance;
            collision_penetrations.push((collider.0, Vec3::new(0.0, distance, 0.0)));
        }
    }

    //NOTE: VELOCITY UPDATE
    for (entity, pen_vec) in collision_penetrations {
        let rigidbody = world
            .get_mut::<(RigidBody,)>(entity)
            .expect("thing with collision regesterd should have rigidbody");
        let mut restitution = 0.3;
        if rigidbody.velocity.mag_sq() < 0.3 {
            rigidbody.velocity = Vec3::zero();
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
    velocity - (1.0 + restitution) * projection
}
fn camera_update(world: &mut World, _delta_time: f32, offset: Vec3) {
    let mouse = *world.get_resource::<MouseState>().unwrap();
    let player = world.query::<(Controllable,)>().next().unwrap().0;
    let player_transform = *world.get::<(Transform,)>(player).unwrap();
    let camera = *world.get_mut_resource::<Camera>().unwrap();
    let height = world
        .get_resource::<TerrainMap>()
        .unwrap()
        .get_height_at(camera.position.x, camera.position.z);
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
    if (camera.position.y - height) < 0.5 {
        camera.position.y = height + 0.5;
    }
    world.get_mut_resource::<MouseState>().unwrap().x = 0.0;
    world.get_mut_resource::<MouseState>().unwrap().y = 0.0;
}

fn player_update(world: &mut World, delta_time: f32) {
    // first pass: get player position
    let player_pos = {
        let mut binding = world.query_mut::<(ReqM<Controllable>, ReqM<Transform>)>();
        let (_, (_, transform)) = binding.next().unwrap();
        transform.position
    };

    // sample terrain while we have no other borrows
    let (terrain_normal, terrain_height) = {
        let terrainmap = world.get_resource::<TerrainMap>().unwrap();
        (
            terrainmap.get_normal_at(player_pos.x, player_pos.z),
            terrainmap.get_height_at(player_pos.x, player_pos.z),
        )
    };

    let keyboard = world
        .get_resource::<Keyboard>()
        .expect("keyboard should have been added during resumed")
        .clone();

    let mut delta_v = Vec3::zero();
    let mut jump = 0.0;
    let camera_rotation = world.get_resource::<Camera>().unwrap().rotation;
    let mut binding = world.query_mut::<(ReqM<Controllable>, ReqM<Transform>, ReqM<RigidBody>)>();
    let (_, (control, transform, rigidbody)) = binding.next().expect("player should exist!");
    let mut max_speed = control.speed;

    if keyboard.is_down(KeyCode::ShiftLeft) {
        max_speed = control.sprint_speed;
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

    // BUG: allows the player to jump at the apex of their jump, but the period of time which the
    // bug is viable to exploit is almost zero so not going to fix
    if keyboard.is_down(KeyCode::Space) && rigidbody.velocity.y.abs() < 0.00001 {
        jump = 8.0
    }
    if !(delta_v == Vec3::zero()) {
        delta_v = camera_rotation * delta_v;
        delta_v.normalize();
        delta_v = Vec3 {
            x: delta_v.x,
            y: 0.0,
            z: delta_v.z,
        };
        delta_v.normalize();

        let on_ground = (terrain_height - player_pos.y).abs() < 0.6;
        let n = terrain_normal.normalized();
        trace!(?n);
        // rotor 1: tilts from flat ground to terrain slope — only pitch/roll, no yaw
        let tilt = if n.dot(Vec3::unit_y()) < -1.0 + 1e-6 {
            // terrain is exactly upside down — rotate 180° around any horizontal axis
            Rotor3::from_rotation_between(Vec3::unit_y(), Vec3::unit_x())
                * Rotor3::from_rotation_between(Vec3::unit_x(), n)
        } else {
            Rotor3::from_rotation_between(Vec3::unit_y(), n)
        };
        let face_direction = {
            // rotor 2: spins to face movement direction — only yaw, no pitch/roll
            let forward = Vec3::new(0.0, 0.0, -1.0);
            let yaw = if delta_v.dot(forward) < -1.0 + 1e-6 {
                // moving exactly backwards — rotate 180° around Y
                Rotor3::from_angle_plane(PI, Bivec3::from_normalized_axis(Vec3::unit_y()))
            } else {
                Rotor3::from_rotation_between(forward, delta_v)
            };

            // compose: apply yaw first, then tilt on top
            (tilt.scaled_by(if on_ground { 1.0 } else { 0.0 }) * yaw).normalized()
        };

        let horizontal_velocity = rigidbody.velocity - n * n.dot(rigidbody.velocity);
        let speed_remaining = (max_speed - horizontal_velocity.mag()).clamp(0.0, max_speed);
        let acceleration = 20.0;

        trace!(?horizontal_velocity, speed_remaining, ?delta_v);
        delta_v = if on_ground { tilt * delta_v } else { delta_v };
        delta_v *= speed_remaining * delta_time * acceleration;
        trace!(delta_v_final = ?delta_v);
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
