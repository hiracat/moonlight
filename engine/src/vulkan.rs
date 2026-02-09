use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use winit::window::Window;

pub type SharedAllocator = Arc<Mutex<Allocator>>;
pub type QueueFamilyIndex = u32;

pub struct VulkanContext {
    // needs to be kept alive, dont forget is very important
    // anything that starts with ash:: and not vk:: impliments drop
    _entry: ash::Entry,
    instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    pub debug_utils_loader: ash::ext::debug_utils::Instance,
    pub allocator: SharedAllocator,
    pub queue: vk::Queue,
    pub queue_family_index: QueueFamilyIndex,
}

pub struct SurfaceContext {
    pub window: Arc<Window>,
    surface: vk::SurfaceKHR,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub surface_loader: ash::khr::surface::Instance,
    pub framebuffer_resized: bool,
}
