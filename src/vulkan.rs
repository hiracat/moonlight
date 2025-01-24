use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    swapchain::Surface,
    VulkanLibrary,
};
use winit::event_loop::ActiveEventLoop;

pub fn create_instance(event_loop: &ActiveEventLoop) -> Arc<Instance> {
    let library =
        VulkanLibrary::new().expect("failed to load library, please install vulkan drivers");
    let required_extensions = Surface::required_extensions(event_loop);

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance")
}
pub fn create_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    required_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|physical_device| {
            physical_device
                .supported_extensions()
                .contains(required_extensions)
        })
        .filter_map(|physical_device| {
            physical_device
                .queue_family_properties()
                .iter() // iterate over all available queues for each device
                .enumerate() // returns an iterator of type (physicaldevice, u32)
                .position(|(index, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS)
                        && physical_device
                            .surface_support(index as u32, surface)
                            .unwrap()
                })
                .map(|index| (physical_device, index as u32))
        })
        .min_by_key(
            |(physical_device, _)| match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            },
        )
        .expect("no qualified gpu")
}
pub fn create_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    required_extensions: &DeviceExtensions,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_features: Features {
                separate_depth_stencil_layouts: true, // MUST ENABLE THIS
                ..Default::default()
            },
            enabled_extensions: *required_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    (device, queue)
}
