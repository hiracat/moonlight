use std::time::Instant;

use winit::event_loop::ActiveEventLoop;

use crate::{
    ecs::World,
    renderer::{
        draw::{UIRenderer, WorldRenderer},
        resources::ResourceManager,
    },
    vulkan::VulkanContext,
};

trait Game {
    fn on_start(&mut self, engine: &mut Engine);
    fn on_update(&mut self, engine: &mut Engine, delta_time: f32);
    fn on_close(&mut self, engine: &mut Engine);
}

struct Engine {
    vulkan_context: VulkanContext,
    world_renderer: WorldRenderer,
    ui_renderer: UIRenderer,
    resource_manager: ResourceManager,
    world: World,
    prev_frame_end: Instant,
    delta_time: f32,
    current_frame: u64,
}

impl Engine {
    fn init(event_loop: &ActiveEventLoop) -> Self {
        let context = VulkanContext::init(event_loop);
        let world_renderer = WorldRenderer::init(event_loop, &context.window);
        let resource_manager = ResourceManager::init(&context);
        let ui_renderer = UIRenderer::init(&context);
        Self {
            vulkan_context: context,
            world_renderer: world_renderer,
            ui_renderer: ui_renderer,
            resource_manager: resource_manager,
            world: World::init(),
            prev_frame_end: Instant::now(),
            delta_time: 0.0,
            current_frame: 0,
        }
    }
    fn run(&mut self, game: Box<dyn Game>) {
        todo!()
    }
}
