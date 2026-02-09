use std::{
    ffi::CStr,
    marker::PhantomData,
    ptr,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
#[allow(deprecated)]
use winit::{
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

pub type SharedAllocator = Arc<Mutex<Allocator>>;
pub type QueueFamilyIndex = u32;
pub const VALIDATION_ENABLE: bool = true;

pub struct VulkanContext {
    // needs to be kept alive, dont forget is very important
    // anything that starts with ash:: and not vk:: impliments drop
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    debug_utils_loader: ash::ext::debug_utils::Instance,

    pub allocator: SharedAllocator,
    pub queue: vk::Queue,
    pub queue_family_index: QueueFamilyIndex,

    pub window: Arc<Window>,
    pub surface: vk::SurfaceKHR,
    pub framebuffer_resized: bool,
}

impl VulkanContext {
    pub fn init(event_loop: &ActiveEventLoop) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let window = create_window(event_loop);
        let instance = create_instance(&entry, event_loop);
        let surface = unsafe {
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                // idk what to do about this, i need the raw handle
                #[allow(deprecated)]
                event_loop.raw_display_handle().unwrap(),
                #[allow(deprecated)]
                window.raw_window_handle().unwrap(),
                None,
            )
            .unwrap();
            surface
        };

        //need to make this optional to put stuff inside
        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let mut debug_messenger = vk::DebugUtilsMessengerEXT::null();
        if VALIDATION_ENABLE {
            debug_messenger = setup_debug_utils(&debug_utils_loader);
            eprintln!("set up debug utility");
        }

        let required_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME,
            // ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME,
        ];
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, surface, &required_extensions);
        let (device, queue) = create_device(
            &instance,
            physical_device,
            queue_family_index,
            &required_extensions,
        );

        let memory_allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })
        .unwrap();
        Self {
            _entry: entry,
            instance: instance,
            device: device,
            physical_device: physical_device,
            queue: queue,
            queue_family_index: queue_family_index,
            allocator: Arc::new(Mutex::new(memory_allocator)),
            debug_messenger: debug_messenger,
            debug_utils_loader: debug_utils_loader,
            window: window,
            surface: surface,
            framebuffer_resized: false,
        }
    }
}
impl Drop for VulkanContext {
    fn drop(&mut self) {
        todo!()
    }
}
pub fn create_instance(entry: &ash::Entry, event_loop: &ActiveEventLoop) -> ash::Instance {
    let engine_name = CStr::from_bytes_with_nul(b"moonlight\0").unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        p_engine_name: engine_name.as_ptr(),
        ..Default::default()
    };

    #[allow(deprecated)]
    let required_instance_extensions =
        ash_window::enumerate_required_extensions(event_loop.raw_display_handle().unwrap())
            .unwrap();

    let mut instance_extensions: Vec<_> = required_instance_extensions.iter().copied().collect();
    instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());

    let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
    let validation_features = [
        vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
        // vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
        // vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
        vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
        // vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
    ];
    let validation_features_info = vk::ValidationFeaturesEXT {
        enabled_validation_feature_count: validation_features.len() as u32,
        p_enabled_validation_features: validation_features.as_ptr(),
        disabled_validation_feature_count: 0,
        p_disabled_validation_features: std::ptr::null(),
        ..Default::default()
    };

    let validation_layers = if VALIDATION_ENABLE {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    let create_info = vk::InstanceCreateInfo {
        p_next: &validation_features_info as *const _ as *const std::ffi::c_void,
        p_application_info: &app_info,

        enabled_layer_count: validation_layers.len() as u32,
        pp_enabled_layer_names: validation_layers.as_ptr(),

        pp_enabled_extension_names: instance_extensions.as_ptr(),
        enabled_extension_count: instance_extensions.len() as u32,
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
    instance
}
pub fn setup_debug_utils(
    debug_utils_loader: &ash::ext::debug_utils::Instance,
) -> vk::DebugUtilsMessengerEXT {
    let messenger_ci = populate_debug_messenger_create_info();

    let utils_messenger = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&messenger_ci, None)
            .expect("Debug Utils Callback")
    };

    utils_messenger
}
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    println!("{}{}{:?}", severity, types, message);

    vk::FALSE
}
fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,

        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
        _marker: PhantomData,
    }
}
pub fn create_physical_device(
    instance: &ash::Instance,
    // TODO: should use this to test surface support but dont feel like it
    _surface: vk::SurfaceKHR,
    required_extensions: &[&CStr],
) -> (vk::PhysicalDevice, QueueFamilyIndex) {
    let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };

    physical_devices
        .iter()
        .copied()
        // this is desctructuring, think of it as &physical_device is like a (&,physcal_device) tuple,
        // and its doing let (&,physical_device) = physical_device_reference
        // the reference matches the reference, and the name matches the underlying data
        .filter(|&physical_device| {
            let properties = unsafe {
                instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap()
            };

            required_extensions.iter().all(|&required_name| {
                properties.iter().any(|extension_properties| {
                    *required_name == *extension_properties.extension_name_as_c_str().unwrap()
                })
            })
        })
        .filter_map(|physical_device| {
            let queue_family_properties = unsafe {
                let len =
                    instance.get_physical_device_queue_family_properties2_len(physical_device);
                let mut out = vec![vk::QueueFamilyProperties2::default(); len];

                instance.get_physical_device_queue_family_properties2(physical_device, &mut out);
                out
            };

            queue_family_properties
                .iter() // iterate over all available queues for each device
                .enumerate() // returns iterator of (index, queue_family_properties)
                .position(|(_, queue_family_properties)| {
                    queue_family_properties
                        .queue_family_properties
                        .queue_flags
                        .intersects(vk::QueueFlags::GRAPHICS)
                })
                .map(|index| (physical_device, index as u32))
        })
        .min_by_key(|(physical_device, _)| {
            let mut properties = vk::PhysicalDeviceProperties2::default();
            unsafe {
                instance.get_physical_device_properties2(*physical_device, &mut properties);
            };
            match properties.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                vk::PhysicalDeviceType::OTHER => 4,
                _ => 5,
            }
        })
        .expect("no qualified gpu")
}
pub fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: QueueFamilyIndex,
    required_extensions: &[&CStr],
) -> (ash::Device, vk::Queue) {
    let enabled_features = unsafe { instance.get_physical_device_features(physical_device) };
    let queue_create_infos = [vk::DeviceQueueCreateInfo {
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: [1.0].as_ptr(),
        ..Default::default()
    }];
    let required_extensions: Vec<*const i8> = required_extensions
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect();

    let create_info = vk::DeviceCreateInfo {
        p_enabled_features: &enabled_features,
        pp_enabled_extension_names: required_extensions.as_ptr(),
        enabled_extension_count: required_extensions.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        queue_create_info_count: queue_create_infos.len() as u32,

        ..Default::default()
    };

    let device = unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .unwrap()
    };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    (device, queue)
}

pub fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
}
