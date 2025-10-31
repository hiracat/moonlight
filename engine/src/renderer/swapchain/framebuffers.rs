use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::renderer::draw::SharedAllocator;

pub struct Framebuffers {
    pub framebuffers: Vec<vk::Framebuffer>,
    pub final_color_views: Vec<vk::ImageView>,

    pub color_images: Vec<Image>,
    pub normal_images: Vec<Image>,
    pub position_images: Vec<Image>,
}
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: Allocation,
}

pub fn create_framebuffers(
    device: &ash::Device,
    swapchain_images: &[vk::Image],
    render_pass: vk::RenderPass,
    allocator: SharedAllocator,
    swapchain_image_extent: vk::Extent2D,
    swapchain_image_format: vk::Format,
    // HACK: the vec of allocations is to keep the allocatiosn from dropping and freeing underlying
    // memory, this should be replaced with some kind of wrapper or just one allocation per
    // framebuffer
) -> Framebuffers {
    let extent = vk::Extent3D {
        width: swapchain_image_extent.width,
        height: swapchain_image_extent.height,
        depth: 1,
    };
    let color_format = vk::Format::A2B10G10R10_UNORM_PACK32;
    let depth_format = vk::Format::D32_SFLOAT;
    let normal_format = vk::Format::R16G16B16A16_SFLOAT;
    let position_format = vk::Format::R32G32B32A32_SFLOAT;

    let mut framebuffers = Vec::new();
    let mut final_colors = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();
    let mut positions = Vec::new();

    for i in 0..swapchain_images.len() {
        let color_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            base_mip_level: 0,
            base_array_layer: 0,
        };
        let depth_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            level_count: 1,
            layer_count: 1,
            base_mip_level: 0,
            base_array_layer: 0,
        };

        let depth = create_image(
            device,
            &allocator,
            "depth image",
            extent,
            depth_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth_subresource_range,
        );

        let color = create_image(
            device,
            &allocator,
            "color image",
            extent,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            color_subresource_range,
        );
        let normal = create_image(
            device,
            &allocator,
            "color image",
            extent,
            normal_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            color_subresource_range,
        );
        // HACK: needs to be removed, you can reconstruct the worldspace coordinates from the depth
        // buffer and the world matrix and screen coords, this is unnecessary
        let position = create_image(
            device,
            &allocator,
            "color image",
            extent,
            position_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            color_subresource_range,
        );

        let final_color_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    format: swapchain_image_format,
                    image: swapchain_images[i],
                    view_type: vk::ImageViewType::TYPE_2D,
                    subresource_range: color_subresource_range,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();

        let attachments = vec![
            final_color_view,
            color.view,
            normal.view,
            depth.view,
            position.view,
        ];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo {
                        render_pass,
                        p_attachments: attachments.as_ptr(),
                        attachment_count: attachments.len() as u32,
                        width: swapchain_image_extent.width,
                        height: swapchain_image_extent.height,
                        layers: 1,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        framebuffers.push(framebuffer);
        final_colors.push(final_color_view);
        colors.push(color);
        normals.push(normal);
        positions.push(position);
    }
    Framebuffers {
        framebuffers,
        final_color_views: final_colors,
        position_images: positions,
        normal_images: normals,
        color_images: colors,
    }
}

fn create_image(
    device: &ash::Device,
    allocator: &SharedAllocator,
    name: &str,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    subresource_range: vk::ImageSubresourceRange,
) -> Image {
    let create_info = vk::ImageCreateInfo {
        extent,
        format,
        usage,
        image_type: vk::ImageType::TYPE_2D,
        samples: vk::SampleCountFlags::TYPE_1,
        mip_levels: 1,
        array_layers: 1,
        ..Default::default()
    };

    let image = unsafe { device.create_image(&create_info, None).unwrap() };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            requirements,
            name,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .unwrap();
    }
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo {
                subresource_range: subresource_range,
                format: format,
                image: image,
                view_type: vk::ImageViewType::TYPE_2D,
                ..Default::default()
            },
            None,
        )
    }
    .unwrap();

    Image {
        image,
        view,
        memory: allocation,
    }
}
