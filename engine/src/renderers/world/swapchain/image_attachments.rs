use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::vulkan::SharedAllocator;

pub struct GBufferResources {
    pub color_images: Vec<Image>,
    pub normal_images: Vec<Image>,
    pub position_images: Vec<Image>,

    pub depth_images: Vec<Image>,
}

pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    // always exists, but required to satisfy the borrow checker
    pub memory: Option<Allocation>,

    device: Arc<ash::Device>,
    allocator: SharedAllocator,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
        }
        self.allocator
            .lock()
            .unwrap()
            .free(
                self.memory
                    .take()
                    .expect("memory should always be some til drop"),
            )
            .unwrap();
    }
}

pub fn create_gbuffer_resources(
    device: &Arc<ash::Device>,
    swapchain_images: &[vk::Image],
    allocator: SharedAllocator,
    swapchain_image_extent: vk::Extent2D,
) -> GBufferResources {
    let extent = vk::Extent3D {
        width: swapchain_image_extent.width,
        height: swapchain_image_extent.height,
        depth: 1,
    };
    let color_format = vk::Format::A2B10G10R10_UNORM_PACK32;
    let depth_format = vk::Format::D32_SFLOAT;
    let normal_format = vk::Format::R16G16B16A16_SFLOAT;
    let position_format = vk::Format::R32G32B32A32_SFLOAT;

    let mut colors = Vec::new();
    let mut normals = Vec::new();
    let mut positions = Vec::new();
    let mut depths = Vec::new();

    for _ in 0..swapchain_images.len() {
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
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            color_subresource_range,
        );
        let normal = create_image(
            device,
            &allocator,
            "color image",
            extent,
            normal_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
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
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            color_subresource_range,
        );

        colors.push(color);
        normals.push(normal);
        positions.push(position);
        depths.push(depth);
    }
    GBufferResources {
        color_images: colors,
        normal_images: normals,
        position_images: positions,
        depth_images: depths,
    }
}

fn create_image(
    device: &Arc<ash::Device>,
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
        memory: Some(allocation),
        allocator: allocator.clone(),
        device: device.clone(),
    }
}
