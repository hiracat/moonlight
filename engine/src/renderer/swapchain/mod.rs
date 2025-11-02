use ash::vk;
use winit::dpi::PhysicalSize;

use crate::renderer::{
    draw::{QueueFamilyIndex, SharedAllocator},
    swapchain::framebuffers::{Framebuffers, Image, create_framebuffers},
};
pub mod framebuffers;
pub mod renderpass;

pub(crate) struct SwapchainResources {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_image_format: vk::SurfaceFormatKHR,

    // wait before presenting
    pub render_finished: Vec<vk::Semaphore>,

    pub(crate) framebuffers: Framebuffers,
    // one per swapchain image
    pub(crate) per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,

    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    render_pass: vk::RenderPass,
    surface: vk::SurfaceKHR,
    per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    queue_family_indices: Vec<QueueFamilyIndex>,
    allocator: SharedAllocator,
}
impl SwapchainResources {
    pub(crate) fn recreate(&mut self, size: PhysicalSize<u32>) {
        *self = SwapchainResources::create(
            &self.surface_loader,
            &self.swapchain_loader,
            &self.device,
            self.physical_device,
            self.render_pass,
            Some(self.swapchain),
            self.swapchain_image_format,
            size,
            self.allocator.clone(),
            self.surface,
            self.per_swapchain_image_set_layout,
            &self.queue_family_indices,
        );
    }
    pub(crate) fn create(
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        render_pass: vk::RenderPass,
        previous_swapchain: Option<vk::SwapchainKHR>,
        swapchain_image_format: vk::SurfaceFormatKHR,
        window_size: PhysicalSize<u32>,
        allocator: SharedAllocator,
        surface: vk::SurfaceKHR,
        per_swapchain_image_set_layout: vk::DescriptorSetLayout,
        queue_family_indices: &[QueueFamilyIndex],
    ) -> Self {
        let window_size = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };

        let image_format = swapchain_image_format;

        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let image_count = if capabilities.min_image_count == capabilities.max_image_count {
            capabilities.max_image_count
        } else {
            capabilities.min_image_count + 1
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            surface,
            min_image_count: image_count,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            image_color_space: image_format.color_space,
            image_format: image_format.format,
            image_extent: window_size,
            present_mode: vk::PresentModeKHR::FIFO,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            old_swapchain: previous_swapchain.unwrap_or(vk::SwapchainKHR::null()),
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            clipped: vk::TRUE,
            image_array_layers: 1,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count: queue_family_indices.len() as u32,

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

        let framebuffers = create_framebuffers(
            device,
            &swapchain_images,
            render_pass,
            allocator.clone(),
            window_size,
            image_format.format,
        );

        // one set per swapchain image, * 4 for color normal position and final_color * 3 for
        // safety
        let required_sets: u32 = (swapchain_images.len() * 4 * 3) as u32;

        let pool_sizes = [vk::DescriptorPoolSize {
            // per uniform buffer, so six descriptors per
            // set(2x what is needed for safety
            descriptor_count: 6 * required_sets,
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
        }];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: required_sets,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        pool_size_count: pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        let descriptor_sets = create_attachment_descriptor_sets(
            device,
            descriptor_pool,
            per_swapchain_image_set_layout,
            &[
                framebuffers.color_images.as_slice(),
                framebuffers.normal_images.as_slice(),
                framebuffers.position_images.as_slice(),
            ],
            &[0, 1, 2],
        );

        let render_finished = create_semaphores(device, image_count as usize);

        eprintln!("swapchain created successfully");

        Self {
            swapchain,
            render_finished,
            swapchain_image_format: image_format,
            framebuffers,
            allocator,
            per_swapchain_image_descriptor_sets: descriptor_sets,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            device: device.clone(),
            physical_device,
            render_pass,
            surface,
            per_swapchain_image_set_layout,
            queue_family_indices: queue_family_indices.to_vec(),
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

fn create_semaphores(device: &ash::Device, count: usize) -> Vec<vk::Semaphore> {
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

pub(crate) fn create_attachment_descriptor_sets(
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

    descriptor_sets
}
