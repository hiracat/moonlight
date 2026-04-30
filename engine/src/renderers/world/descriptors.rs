use ash::vk;

use crate::resources::{Material, ResourceManager};

/// helper for making descriptor writes
/// besides being extremely boilerplate heavy, making descriptor writes requires managing stable
/// memory addresses for the bufferinfo sub structs, so this abstracts that away
pub struct DescriptorWriteBuilder<'a> {
    // i need a stable memory address for the buffer info, so box them, then store those
    #[allow(clippy::vec_box)]
    buffer_infos: Vec<Box<vk::DescriptorBufferInfo>>,
    #[allow(clippy::vec_box)]
    image_infos: Vec<Box<vk::DescriptorImageInfo>>,
    writes: Vec<vk::WriteDescriptorSet<'a>>,
}

impl DescriptorWriteBuilder<'_> {
    pub fn new() -> Self {
        Self {
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub fn add_uniform_buffer<T: bytemuck::Pod>(
        &mut self,
        resource_manager: &mut ResourceManager,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binding: u32,
        data: &T,
    ) -> vk::DescriptorSet {
        let size = size_of::<T>();
        let (buffer, offset) = resource_manager.ubo_ring_buffer.allocate(size);
        resource_manager.ubo_ring_buffer.write(data, offset);

        let descriptor_set =
            resource_manager.allocate_temp_descriptor_set(descriptor_set_layout, descriptor_pool);

        // Store buffer info (we need stable addresses)
        self.buffer_infos.push(Box::new(vk::DescriptorBufferInfo {
            buffer,
            offset,
            range: size as u64,
        }));

        // Record the write
        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_buffer_info: self.buffer_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    pub fn add_texture(
        &mut self,
        resource_manager: &mut ResourceManager,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binding: u32,
        texture: Option<Material>,
    ) -> vk::DescriptorSet {
        let descriptor_set =
            resource_manager.allocate_temp_descriptor_set(descriptor_set_layout, descriptor_pool);

        let image = match texture {
            Some(x) => resource_manager
                .get_texture(x.albedo)
                .expect("texture should exist"),
            None => resource_manager.default_texture(),
        };

        self.image_infos.push(Box::new(vk::DescriptorImageInfo {
            sampler: image.sampler,
            image_view: image.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }));

        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_image_info: self.image_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    pub fn add_ssbo(
        &mut self,
        resource_manager: &mut ResourceManager,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
        offset: u64,
    ) -> vk::DescriptorSet {
        let descriptor_set =
            resource_manager.allocate_temp_descriptor_set(descriptor_set_layout, descriptor_pool);

        self.buffer_infos.push(Box::new(vk::DescriptorBufferInfo {
            buffer,
            range,
            offset,
        }));

        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_buffer_info: self.buffer_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    pub fn add_ssbo_binding(
        &mut self,
        descriptor_set: vk::DescriptorSet,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
        offset: u64,
    ) -> vk::DescriptorSet {
        self.buffer_infos.push(Box::new(vk::DescriptorBufferInfo {
            buffer,
            range,
            offset,
        }));

        self.writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            p_buffer_info: self.buffer_infos.last().unwrap().as_ref(),
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            dst_array_element: 0,
            ..Default::default()
        });

        descriptor_set
    }

    pub fn submit(self, device: &ash::Device) {
        unsafe {
            device.update_descriptor_sets(&self.writes, &[]);
        }
    }
}
