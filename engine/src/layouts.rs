use ash::vk;

pub struct DescriptorLayouts {
    pub geometry_per_model_layout_0: vk::DescriptorSetLayout,
    pub geometry_per_frame_layout_1: vk::DescriptorSetLayout,
    pub lighting_per_light_layout_2: vk::DescriptorSetLayout,
    pub lighting_per_frame_layout_1: vk::DescriptorSetLayout,
    pub lighting_per_swapchain_image_attachment_0: vk::DescriptorSetLayout,
    pub geometry_static_texture_layout_2: vk::DescriptorSetLayout,
}
pub(crate) struct Bindings {
    pub camera: u32,
    pub ambient: u32,
    pub directional: u32,
    pub model: u32,
    pub point: u32,
}

pub(crate) static BINDINGS: Bindings = Bindings {
    model: 0,
    ambient: 0,
    directional: 1,
    camera: 0,
    point: 0,
};

impl DescriptorLayouts {
    pub fn init(device: &ash::Device) -> Self {
        Self {
            geometry_per_model_layout_0: Self::create_geometry_per_model_layout_0(device),
            geometry_per_frame_layout_1: Self::create_geometry_per_frame_layout_1(device),
            lighting_per_light_layout_2: Self::create_lighting_per_light_layout_2(device),
            lighting_per_frame_layout_1: Self::create_lighting_per_frame_layout_1(device),
            geometry_static_texture_layout_2: Self::create_geometry_static_texture_layout_2(device),
            lighting_per_swapchain_image_attachment_0:
                Self::create_lighting_per_swapchain_image_layout_0(device),
        }
    }
    fn create_geometry_static_texture_layout_2(device: &ash::Device) -> vk::DescriptorSetLayout {
        let geometry_bindings = vec![
            // texture
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let geometry_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: geometry_bindings.as_ptr(),
            binding_count: geometry_bindings.len() as u32,
            ..Default::default()
        };
        let geometry = unsafe {
            device
                .create_descriptor_set_layout(&geometry_layout, None)
                .unwrap()
        };

        geometry
    }

    fn create_geometry_per_model_layout_0(device: &ash::Device) -> vk::DescriptorSetLayout {
        let geometry_bindings = vec![
            // model
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
        ];
        let geometry_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: geometry_bindings.as_ptr(),
            binding_count: geometry_bindings.len() as u32,
            ..Default::default()
        };
        let geometry = unsafe {
            device
                .create_descriptor_set_layout(&geometry_layout, None)
                .unwrap()
        };

        geometry
    }
    fn create_geometry_per_frame_layout_1(device: &ash::Device) -> vk::DescriptorSetLayout {
        let geometry_bindings = vec![
            // camera
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
        ];
        let geometry_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: geometry_bindings.as_ptr(),
            binding_count: geometry_bindings.len() as u32,
            ..Default::default()
        };
        let geometry = unsafe {
            device
                .create_descriptor_set_layout(&geometry_layout, None)
                .unwrap()
        };

        geometry
    }
    fn create_lighting_per_frame_layout_1(device: &ash::Device) -> vk::DescriptorSetLayout {
        let lighting_bindings = vec![
            //ambient
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            //directional/sun
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let lighting_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: lighting_bindings.as_ptr(),
            binding_count: lighting_bindings.len() as u32,
            ..Default::default()
        };
        let lighting = unsafe {
            device
                .create_descriptor_set_layout(&lighting_layout, None)
                .unwrap()
        };

        lighting
    }
    fn create_lighting_per_swapchain_image_layout_0(
        device: &ash::Device,
    ) -> vk::DescriptorSetLayout {
        let lighting_bindings = vec![
            //color attachment
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // normals
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // position
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let lighting_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: lighting_bindings.as_ptr(),
            binding_count: lighting_bindings.len() as u32,
            ..Default::default()
        };
        let lighting = unsafe {
            device
                .create_descriptor_set_layout(&lighting_layout, None)
                .unwrap()
        };

        lighting
    }
    fn create_lighting_per_light_layout_2(device: &ash::Device) -> vk::DescriptorSetLayout {
        let lighting_bindings = vec![
            //point light data
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let lighting_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: lighting_bindings.as_ptr(),
            binding_count: lighting_bindings.len() as u32,
            ..Default::default()
        };
        let lighting = unsafe {
            device
                .create_descriptor_set_layout(&lighting_layout, None)
                .unwrap()
        };

        lighting
    }
}
