use std::{
    collections::{BTreeMap, HashMap},
    fmt,
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
    resources::{LinearAllocator, Material, ResourceManager, SsboHandle},
};
use crate::{resources::Texture, vulkan::SharedAllocator};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SetKey {
    bindings: Vec<MergedBinding>,
}

#[derive(Debug, Copy, Clone)]
pub struct BindingHandle {
    pipeline: PipelineHandle,
    set_index: u32,
    binding_index: u32,
    data_index: usize,
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
    pipeline_resources: HashMap<PipelineHandle, PipelineResources>,
    // for deduplication
    set_layout_cache: HashMap<SetKey, vk::DescriptorSetLayout>,
    //PERF: this needs some help, like a smarter way of reusing descriptor sets, but i cant think of one, and its not an issue right now
    descriptor_pool: vk::DescriptorPool,
    uniform_memory: LinearAllocator,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for DescriptorManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for (_, layout) in self.set_layout_cache.iter() {
                self.device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
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
                descriptor_count: 4196,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1024,
            },
        ];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: 1024,
                        pool_size_count: pool_sizes.len() as u32,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        Self {
            pipeline_resources: HashMap::new(),
            set_layout_cache: HashMap::new(),
            descriptor_pool,
            // give it 64 kb
            uniform_memory: LinearAllocator::create(allocator, &device, 1024 * 64),
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
        let pipeline_resources = &mut self.pipeline_resources.get_mut(&pipeline).unwrap();
        let vec = pipeline_resources
            .requested_bind_cmds
            .entry(set_index)
            .or_default()
            .entry(binding_index)
            .or_default();
        let data_index = vec.len();
        vec.push(data);

        BindingHandle {
            pipeline,
            set_index,
            binding_index,
            data_index,
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
        let pipeline_resources = &self.pipeline_resources[&pipeline];

        handles.iter().for_each(|x| {
            if x.pipeline != pipeline {
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
            .map(|set_index| {
                *pipeline_resources
                    .set_layouts
                    .get(set_index)
                    .unwrap_or_else(|| {
                        panic!(
                            "pipeline {:?} has no layout for set index {}, available sets: {:?}",
                            pipeline,
                            set_index,
                            pipeline_resources.set_layouts.keys().collect::<Vec<_>>()
                        )
                    })
            })
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
            let binding_data = &pipeline_resources.requested_bind_cmds[&handle.set_index]
                [&handle.binding_index][handle.data_index];

            match binding_data {
                BindingData::Uniform { data } => {
                    let (buffer, offset) = self.uniform_memory.push_raw(data);

                    buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: data.len() as u64,
                    });
                }
                BindingData::Texture { texture } => {
                    let texture = resource_manager.get_texture(*texture).unwrap();
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
                BindingData::RawSsbo {
                    buffer,
                    size,
                    offset,
                } => {
                    buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: *buffer,
                        offset: *offset,
                        range: *size,
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
                BindingData::StorageImage { id } => {
                    let view = graph.get_view_from_id(id);
                    image_infos.push(vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(), // needs a sampler somehow
                        image_view: view,
                        image_layout: vk::ImageLayout::GENERAL,
                    });
                }
            }
        }

        // Second pass: build writes now that buffer_infos won't move
        let mut buffer_info_idx = 0;
        let mut image_info_idx = 0;
        for handle in handles {
            let binding_data = &pipeline_resources.requested_bind_cmds[&handle.set_index]
                [&handle.binding_index][handle.data_index];

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
                BindingData::RawSsbo { .. } => {
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
                BindingData::StorageImage { .. } => {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: set_index_to_descriptor_set[&handle.set_index],
                        dst_binding: handle.binding_index,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
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
        self.uniform_memory.reset();
        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }
        for (_handle, resources) in self.pipeline_resources.iter_mut() {
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

        self.pipeline_resources.insert(
            pipeline_handle,
            PipelineResources {
                requested_bind_cmds: BTreeMap::new(),
                set_layouts,
                pipeline_layout,
                device: self.device.clone(),
            },
        );
        pipeline_layout
    }

    pub fn add_compute(
        &mut self,
        pipeline_handle: PipelineHandle,
        comp: BTreeMap<u32, BTreeMap<u32, rspirv_reflect::DescriptorInfo>>,
    ) -> vk::PipelineLayout {
        // set, binding, cannot remove the binding index even though it is unused because there are
        // multiple bindings with the same set
        let mut merged_bindings: BTreeMap<(u32, u32), MergedBinding> = BTreeMap::new();
        for (set_index, set) in comp.iter() {
            for (binding_index, binding) in set {
                let entry = merged_bindings
                    .entry((*set_index, *binding_index))
                    .or_insert(MergedBinding {
                        binding: *binding_index,
                        ty: vk::DescriptorType::from_raw(binding.ty.0 as i32),
                        count: (&binding.binding_count).into(),
                        stages: vk::ShaderStageFlags::empty(),
                    });

                entry.stages |= vk::ShaderStageFlags::COMPUTE;
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

        self.pipeline_resources.insert(
            pipeline_handle,
            PipelineResources {
                requested_bind_cmds: BTreeMap::new(),
                set_layouts,
                pipeline_layout,
                device: self.device.clone(),
            },
        );
        pipeline_layout
    }
}

#[derive(Educe)]
#[educe(Debug)]
struct PipelineResources {
    // indexed by set, binding
    requested_bind_cmds: BTreeMap<u32, BTreeMap<u32, Vec<BindingData>>>,
    set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,
}

impl Drop for PipelineResources {
    fn drop(&mut self) {
        // set layouts are copied here, so dont need to be destroyed
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
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

pub enum BindingData {
    Uniform {
        data: Vec<u8>,
    },
    Texture {
        texture: Texture,
    },
    Ssbo {
        buffer: SsboHandle,
    },
    RenderGraphImage {
        id: ImageId,
    },
    StorageImage {
        id: ImageId,
    },
    RawSsbo {
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
    },
}
impl fmt::Debug for BindingData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform { data } => {
                let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
                f.debug_struct("Uniform")
                    .field("data", &format_args!("0x{hex}"))
                    .finish()
            }
            Self::Texture { texture } => {
                f.debug_struct("Texture").field("texture", texture).finish()
            }
            Self::Ssbo { buffer } => f.debug_struct("Ssbo").field("buffer", buffer).finish(),
            Self::RenderGraphImage { id } => {
                f.debug_struct("RenderGraphImage").field("id", id).finish()
            }
            Self::StorageImage { id } => f.debug_struct("StorageImage").field("id", id).finish(),
            Self::RawSsbo {
                buffer,
                size,
                offset,
            } => f
                .debug_struct("RawBuffer")
                .field("buffer", buffer)
                .field("size", size)
                .field("offset", offset)
                .finish(),
        }
    }
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
