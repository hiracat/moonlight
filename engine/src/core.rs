use std::time::Instant;

use crate::{
    ecs::World,
    renderer::{
        draw::{UIRenderer, WorldRenderer},
        resources::ResourceManager,
    },
};

trait Game {
    fn on_start(&mut self, engine: &mut Engine);
    fn on_update(&mut self, engine: &mut Engine, delta_time: f32);
    fn on_close(&mut self, engine: &mut Engine);
}

struct Engine {
    renderer: WorldRenderer,
    ui: UIRenderer,
    resource_manager: ResourceManager,
    world: World,
    prev_frame_end: Instant,
    delta_time: f32,
    current_frame: u64,
}
