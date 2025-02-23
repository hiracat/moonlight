use std::{fs::File, io::BufReader, sync::Arc, time::Instant};

use obj::{load_obj, FromRawVertex, Obj};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    swapchain::Surface,
    VulkanLibrary,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::{
    model::Model,
    resources::{self, DummyVertex},
    FRAMES_IN_FLIGHT,
};

pub struct Context {
    pub window: Arc<Window>,
    pub surface: Arc<Surface>,
    pub aspect_ratio: f32,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub dummy_verts: Subbuffer<[resources::DummyVertex]>,
    pub vertex_buffer: Subbuffer<[resources::Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub start_time: Instant,
}

impl Context {
    pub fn init(event_loop: &ActiveEventLoop) -> Self {
        let start_time = Instant::now();
        let instance = create_instance(event_loop);
        let window = create_window(event_loop);
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, &surface, &device_extensions);
        let (device, queue) =
            create_device(physical_device, queue_family_index, &device_extensions);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: FRAMES_IN_FLIGHT,
                ..Default::default()
            },
        ));
        let window_size = window.inner_size();
        let aspect_ratio = {
            let windowsize: [f32; 2] = window_size.into();
            windowsize[0] / windowsize[1]
        };

        let input = BufReader::new(File::open("data/models/low poly fox.obj").unwrap());
        let model = load_obj::<obj::Vertex, _, u32>(input).unwrap();
        let vertices: Vec<resources::Vertex> = model
            .vertices
            .iter()
            .map(|v| resources::Vertex {
                position: v.position,
                normal: v.normal,
                color: [1.0, 1.0, 1.0], // map other fields as needed
            })
            .collect();
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        let indices: Vec<u32> = model.indices.iter().map(|i| *i as u32).collect();
        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let dummy_verts = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DummyVertex::list(),
        )
        .unwrap();

        Context {
            surface,
            command_buffer_allocator,
            dummy_verts,
            device,
            vertex_buffer,
            queue,
            index_buffer,
            start_time,
            window: window.clone(),
            aspect_ratio,
        }
    }
}

fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
}

fn create_instance(event_loop: &ActiveEventLoop) -> Arc<Instance> {
    let library =
        VulkanLibrary::new().expect("failed to load library, please install vulkan drivers");
    let required_extensions = Surface::required_extensions(event_loop);

    let validation_layer = "VK_LAYER_KHRONOS_validation";

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enabled_layers: vec![validation_layer.to_string()],
            ..Default::default()
        },
    )
    .expect("failed to create instance")
}
fn create_physical_device(
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
fn create_device(
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
