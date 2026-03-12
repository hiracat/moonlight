use crate::{renderers::world::swapchain::framebuffers::Image, vulkan::VulkanContext};
use ash::vk;
pub mod framebuffers;
pub mod renderpass;

pub struct SwapchainResources {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_image_format: vk::SurfaceFormatKHR,
    pub surface_loader: ash::khr::surface::Instance,
    pub swapchain_loader: ash::khr::swapchain::Device,
    // the images that are managed by the swapchain, indexed by swapchain_image_index
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,

    pub image_size: vk::Extent2D,
}

impl SwapchainResources {
    pub fn create(context: &VulkanContext, old_swapchain: Option<vk::SwapchainKHR>) -> Self {
        dbg!("creating swapchain");
        dbg!(old_swapchain);
        let surface_loader = ash::khr::surface::Instance::new(&context.entry, &context.instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(&context.instance, &context.device);
        let window_size = context.window.inner_size();
        let window_size = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };

        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(context.physical_device, context.surface)
        }
        .unwrap();

        let image_count = if capabilities.min_image_count == capabilities.max_image_count {
            capabilities.max_image_count
        } else {
            capabilities.min_image_count + 1
        };
        let color_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            base_mip_level: 0,
            base_array_layer: 0,
        };
        let swapchain_image_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(context.physical_device, context.surface)
                .unwrap()
        };
        let swapchain_image_format =
            SwapchainResources::choose_swapchain_format(&swapchain_image_formats).unwrap();

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            surface: context.surface,
            min_image_count: image_count,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            image_color_space: swapchain_image_format.color_space,
            image_format: swapchain_image_format.format,
            image_extent: window_size,
            present_mode: vk::PresentModeKHR::FIFO,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            old_swapchain: old_swapchain.unwrap_or(vk::SwapchainKHR::null()),
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            clipped: vk::TRUE,
            image_array_layers: 1,
            p_queue_family_indices: &context.queue_family_index,
            queue_family_index_count: 1,

            ..Default::default()
        };

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };
        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get Swapchain Images.")
        };

        let mut swapchain_image_views = Vec::new();
        for image in &swapchain_images {
            let swapchain_image_view = unsafe {
                context.device.create_image_view(
                    &vk::ImageViewCreateInfo {
                        format: swapchain_image_format.format,
                        image: *image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        subresource_range: color_subresource_range,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap();
            swapchain_image_views.push(swapchain_image_view);
        }

        Self {
            image_size: window_size,
            swapchain,
            swapchain_image_format: swapchain_image_format,
            surface_loader: surface_loader,
            swapchain_loader: swapchain_loader,
            swapchain_images: swapchain_images,
            swapchain_image_views: swapchain_image_views,
        }
    }

    pub fn choose_swapchain_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> Option<vk::SurfaceFormatKHR> {
        let preferred_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let preferred_formats = vec![
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
        ];

        for available in formats {
            for format in &preferred_formats {
                if available.format == *format && available.color_space == preferred_color_space {
                    return Some(*available);
                }
            }
        }
        return formats.first().copied();
    }
}

pub fn create_semaphores(device: &ash::Device, count: usize) -> Vec<vk::Semaphore> {
    let mut semaphores = Vec::with_capacity(count);
    for _ in 0..count {
        semaphores.push(unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        });
    }
    semaphores
}

/// ## Args
/// - **descriptor_pool**: Pool for descriptor set allocation
/// - **attachment_layout**: Descriptor set layout defining binding structure
/// - **binding_index**: Binding indices for each attachment type (length must be
/// attachments.len())

pub fn create_attachment_descriptor_sets(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    attachment_layout: vk::DescriptorSetLayout,
    attachments: &[&[Image]],
    binding_indices: &[u32],
) -> Vec<vk::DescriptorSet> {
    debug_assert!(
        attachments.len() == binding_indices.len(),
        "Must have equal number of attachment types {} as bindings",
        binding_indices.len()
    );

    let frame_count = attachments[0].len();
    let set_layouts = vec![attachment_layout; frame_count];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                p_set_layouts: set_layouts.as_ptr(),
                descriptor_set_count: set_layouts.len() as u32,
                ..Default::default()
            })
            .unwrap()
    };

    update_attachment_descriptor_sets(device, &descriptor_sets, attachments, binding_indices);

    descriptor_sets
}

pub fn update_attachment_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    attachments: &[&[Image]],
    binding_indices: &[u32],
) {
    let frame_count = attachments[0].len();
    let mut descriptor_writes = Vec::new();
    let mut all_image_infos: Vec<Vec<vk::DescriptorImageInfo>> =
        Vec::with_capacity(attachments.len());

    for binding_index in 0..attachments.len() {
        let mut image_infos = Vec::with_capacity(frame_count);
        for frame_index in 0..attachments[binding_index].len() {
            image_infos.push(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: attachments[binding_index][frame_index].view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });

            descriptor_writes.push(vk::WriteDescriptorSet {
                dst_set: descriptor_sets[frame_index],
                p_image_info: &image_infos[frame_index],
                descriptor_count: 1,
                dst_binding: binding_indices[binding_index],
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                dst_array_element: 0,
                ..Default::default()
            })
        }
        all_image_infos.push(image_infos);
    }

    unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }
}
