use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use ash::vk;
use educe::Educe;
use rspirv_reflect as rr;

use crate::{
    renderers::world::{
        pipelines::PipelineHandle,
        rendergraph::{CompiledRenderGraph, ImageId},
    },
    resources::{Material, ResourceManager, SsboHandle, UniformRingBuffer},
};
use crate::{resources::Texture, vulkan::SharedAllocator};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SetKey {
    bindings: Vec<MergedBinding>,
}

#[derive(Debug, Copy, Clone)]
pub struct BindingHandle {
    pipeline_index: usize,
    set_index: u32,
    binding_index: u32,
}

// copied from rspirv_reflect, because it doesnt impl hash
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum BindingCount {
    One,
    StaticSized(usize),
    Unbounded,
}

#[derive(Educe)]
#[educe(Debug)]
pub struct DescriptorManager {
    // indexed by pipeline_handle.arr_index, but since we dont know the order of addition, we just
    // need to try our best, initializing the blank space with none
    pipeline_resources: Vec<Option<PipelineResources>>,
    // for deduplication
    set_layout_cache: HashMap<SetKey, vk::DescriptorSetLayout>,
    //PERF: this needs some help, like a smarter way of reusing descriptor sets, but i cant think of one, and its not an issue right now
    descriptor_pool: vk::DescriptorPool,
    uniform_memory: UniformRingBuffer,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct MergedBinding {
    binding: u32,
    ty: vk::DescriptorType,
    count: BindingCount,
    stages: vk::ShaderStageFlags,
}

impl DescriptorManager {
    pub fn new(device: Arc<ash::Device>, allocator: SharedAllocator) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 256,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 256,
            },
        ];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: 256,
                        pool_size_count: pool_sizes.len() as u32,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        Self {
            pipeline_resources: Vec::new(),
            set_layout_cache: HashMap::new(),
            descriptor_pool,
            uniform_memory: UniformRingBuffer::create(allocator, &device, 1024 * 1024),
            descriptor_sets: Vec::new(),
            device,
        }
    }

    /// request bind is very cheap to call
    pub fn request_bind(
        &mut self,
        pipeline: PipelineHandle,
        set_index: u32,
        binding_index: u32,
        data: BindingData,
    ) -> BindingHandle {
        let pipeline_resources = self.pipeline_resources[pipeline.arr_index]
            .as_mut()
            .unwrap();
        pipeline_resources
            .requested_bind_cmds
            .entry(set_index)
            // fill with an empty btree map if none are there
            .or_default()
            .insert(binding_index, data);
        BindingHandle {
            pipeline_index: pipeline.arr_index,
            set_index,
            binding_index,
        }
    }
    /// bind takes all the handles and actually takes gpu state allocates buffers and writes data, so its much more
    /// expensive to call then request_bind
    pub fn bind(
        &mut self,
        resource_manager: &mut ResourceManager,
        graph: &CompiledRenderGraph,
        pipeline: PipelineHandle,
        handles: &[BindingHandle],
    ) -> (vk::PipelineLayout, Vec<vk::DescriptorSet>) {
        let pipeline_resources = self.pipeline_resources[pipeline.arr_index]
            .as_ref()
            .unwrap();

        handles.iter().for_each(|x| {
            if x.pipeline_index != pipeline.arr_index {
                panic!("cannot bind handles in a pipeline they were not created for")
            }
        });

        let mut set_indices: Vec<u32> = handles.iter().map(|h| h.set_index).collect();
        set_indices.sort_unstable();
        set_indices.dedup();
        assert!(
            set_indices
                .iter()
                .enumerate()
                .all(|(i, &si)| si == i as u32),
            "descriptor set indices must be contiguous starting from 0, got {:?}",
            set_indices
        );

        let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = set_indices
            .iter()
            .map(|set_index| pipeline_resources.set_layouts[set_index])
            .collect();

        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.descriptor_pool,
                    descriptor_set_count: descriptor_set_layouts.len() as u32,
                    p_set_layouts: descriptor_set_layouts.as_ptr(),
                    ..Default::default()
                })
                .unwrap()
        };

        let set_index_to_descriptor_set: HashMap<u32, vk::DescriptorSet> = set_indices
            .iter()
            .copied()
            .zip(descriptor_sets.iter().copied())
            .collect();

        // Build write descriptors for each handle
        // We need to keep the backing buffers alive until after update_descriptor_sets
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

        for handle in handles {
            let binding_data =
                &pipeline_resources.requested_bind_cmds[&handle.set_index][&handle.binding_index];

            match binding_data {
                BindingData::Uniform { data } => {
                    let (buffer, offset) = self.uniform_memory.push_raw(data);

                    buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: data.len() as u64,
                    });
                }
                BindingData::Texture { material } => {
                    let texture = resource_manager.get_texture(material.albedo).unwrap();
                    image_infos.push(vk::DescriptorImageInfo {
                        sampler: texture.sampler,
                        image_view: texture.image_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    });
                }
                BindingData::Ssbo { buffer } => {
                    let binding = resource_manager.ssbo_registry.get_ssbo_binding(*buffer);
                    buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: binding.buffer,
                        offset: binding.offset,
                        range: binding.size,
                    });
                }
                BindingData::RenderGraphImage { id } => {
                    let view = graph.get_view_from_id(id);
                    image_infos.push(vk::DescriptorImageInfo {
                        sampler: graph.sampler, // needs a sampler somehow
                        image_view: view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    });
                }
            }
        }

        // Second pass: build writes now that buffer_infos won't move
        let mut buffer_info_idx = 0;
        let mut image_info_idx = 0;
        for handle in handles {
            let binding_data =
                &pipeline_resources.requested_bind_cmds[&handle.set_index][&handle.binding_index];

            match binding_data {
                BindingData::Uniform { .. } => {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: set_index_to_descriptor_set[&handle.set_index],
                        dst_binding: handle.binding_index,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_infos[buffer_info_idx],
                        ..Default::default()
                    });
                    buffer_info_idx += 1;
                }
                BindingData::Texture { .. } => {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: set_index_to_descriptor_set[&handle.set_index],
                        dst_binding: handle.binding_index,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &image_infos[image_info_idx],
                        ..Default::default()
                    });
                    image_info_idx += 1;
                }
                BindingData::Ssbo { .. } => {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: set_index_to_descriptor_set[&handle.set_index],
                        dst_binding: handle.binding_index,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &buffer_infos[buffer_info_idx],
                        ..Default::default()
                    });
                    buffer_info_idx += 1;
                }
                BindingData::RenderGraphImage { .. } => {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: set_index_to_descriptor_set[&handle.set_index],
                        dst_binding: handle.binding_index,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &image_infos[image_info_idx],
                        ..Default::default()
                    });
                    image_info_idx += 1;
                }
            }
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }

        (pipeline_resources.pipeline_layout, descriptor_sets)
    }
    pub fn begin_frame(&mut self) {
        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }

        for resources in self.pipeline_resources.iter_mut().flatten() {
            resources.requested_bind_cmds.clear();
        }
    }
    pub fn add_pipeline(
        &mut self,
        pipeline_handle: PipelineHandle,
        vert: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
        frag: BTreeMap<u32, BTreeMap<u32, rr::DescriptorInfo>>,
    ) -> vk::PipelineLayout {
        // set, binding, cannot remove the binding index even though it is unused because there are
        // multiple bindings with the same set
        let mut merged_bindings: BTreeMap<(u32, u32), MergedBinding> = BTreeMap::new();
        for (set_index, set) in vert.iter().chain(frag.iter()) {
            for (binding_index, binding) in set {
                let entry = merged_bindings
                    .entry((*set_index, *binding_index))
                    .or_insert(MergedBinding {
                        binding: *binding_index,
                        ty: vk::DescriptorType::from_raw(binding.ty.0 as i32),
                        count: (&binding.binding_count).into(),
                        stages: vk::ShaderStageFlags::empty(),
                    });

                // if vert has the binding
                if vert
                    .get(set_index)
                    .and_then(|s| s.get(binding_index))
                    .is_some()
                {
                    entry.stages |= vk::ShaderStageFlags::VERTEX;
                }

                // if frag has the binding
                if frag
                    .get(set_index)
                    .and_then(|s| s.get(binding_index))
                    .is_some()
                {
                    entry.stages |= vk::ShaderStageFlags::FRAGMENT;
                }
            }
        }
        let mut merged_sets: BTreeMap<u32, Vec<MergedBinding>> = BTreeMap::new();
        for ((set_index, _binding_index), binding) in merged_bindings {
            let set = merged_sets.entry(set_index).or_default();
            set.push(binding);
        }
        let mut set_layouts = HashMap::new();

        for (set_index, bindings) in &merged_sets {
            let vk_bindings: Vec<_> = bindings
                .iter()
                .map(|x| vk::DescriptorSetLayoutBinding {
                    binding: x.binding,
                    descriptor_type: x.ty,
                    descriptor_count: match x.count {
                        BindingCount::One => 1,
                        BindingCount::StaticSized(n) => n as u32,
                        BindingCount::Unbounded => {
                            todo!("bindless descriptors not yet implimented")
                        }
                    },
                    stage_flags: x.stages,
                    ..Default::default()
                })
                .collect();
            let set_key = SetKey {
                bindings: bindings.clone(),
            };
            let &mut layout = self
                .set_layout_cache
                .entry(set_key)
                .or_insert_with(|| unsafe {
                    self.device
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo {
                                binding_count: vk_bindings.len() as u32,
                                p_bindings: vk_bindings.as_ptr(),
                                ..Default::default()
                            },
                            None,
                        )
                        .unwrap()
                });
            set_layouts.insert(*set_index, layout);
        }

        let ordered_layouts: Vec<vk::DescriptorSetLayout> =
            (0..=*merged_sets.keys().max().unwrap())
                .map(|i| set_layouts[&i])
                .collect();

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        set_layout_count: ordered_layouts.len() as u32,
                        p_set_layouts: ordered_layouts.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        while self.pipeline_resources.len() <= pipeline_handle.arr_index {
            self.pipeline_resources.push(None);
        }
        self.pipeline_resources[pipeline_handle.arr_index] = Some(PipelineResources {
            requested_bind_cmds: BTreeMap::new(),
            set_layouts,
            pipeline_layout,
        });
        pipeline_layout
    }
}

#[derive(Debug)]
struct PipelineResources {
    // indexed by set, binding
    requested_bind_cmds: BTreeMap<u32, BTreeMap<u32, BindingData>>,
    set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
}

impl From<&rr::BindingCount> for BindingCount {
    fn from(value: &rr::BindingCount) -> Self {
        match value {
            rspirv_reflect::BindingCount::One => Self::One,
            rspirv_reflect::BindingCount::StaticSized(x) => Self::StaticSized(*x),
            rspirv_reflect::BindingCount::Unbounded => Self::Unbounded,
        }
    }
}

#[derive(Debug)]
pub enum BindingData {
    Uniform { data: Vec<u8> },
    Texture { material: Material },
    Ssbo { buffer: SsboHandle },
    RenderGraphImage { id: ImageId },
}

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
            None => resource_manager.get_texture(Texture::default()).unwrap(),
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

impl Default for DescriptorWriteBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}
