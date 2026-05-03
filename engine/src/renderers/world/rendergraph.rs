use std::{collections::HashMap, sync::Arc};

use ash::vk;
use educe::Educe;

use crate::{
    renderers::world::{
        pipelines::PipelineHandle,
        swapchain::image_attachments::{Image, create_image},
    },
    vulkan::SharedAllocator,
};

#[derive(Debug)]
pub struct RenderGraph {
    pipelines: Vec<PipelineNode>,
    images: Vec<ImageDesc>,
    depth: Option<ImageId>,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum ImageDesc {
    Managed {
        name: &'static str,
        format: vk::Format,
    },
    Imported {
        name: &'static str,
        format: vk::Format,
    },
}
impl ImageDesc {
    fn get_name(&self) -> &'static str {
        match self {
            ImageDesc::Managed { name, .. } | ImageDesc::Imported { name, .. } => name,
        }
    }
    fn get_format(&self) -> vk::Format {
        match self {
            ImageDesc::Managed { format, .. } | ImageDesc::Imported { format, .. } => *format,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct PipelineNode {
    name: String,
    pipeline: PipelineHandle,
    reads: Vec<ImageVersion>,
    writes: Vec<ImageVersion>,
}

pub struct PipelineBuilder {
    render_graph: RenderGraph,
    name: String,
    pipeline: Option<PipelineHandle>,
    reads: Vec<ImageVersion>,
    writes: Vec<ImageVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ImageVersion {
    pub id: ImageId,
    pub name: &'static str,
    desc: ImageDesc,
    // starts at 0, increment every time the image is written to to prevent ambiguous if there are
    // multiple writes and multiple reads
    version: usize,
}

impl ImageVersion {
    // checks if the ids are the same, and
    pub fn depends_on(&self, read: &ImageVersion) -> bool {
        self.id == read.id && self.version >= read.version
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ImageId {
    arr_idx: usize,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            pipelines: Vec::new(),
            images: Vec::new(),
            depth: None,
        }
    }
    pub fn compile(
        &self,
        device: Arc<ash::Device>,
        allocator: SharedAllocator,
        imported_images: &HashMap<ImageId, ImportedImageBinding>,
        image_width: u32,
        image_height: u32,
    ) -> CompiledRenderGraph {
        CompiledRenderGraph::compile(
            self,
            device,
            allocator,
            imported_images,
            image_width,
            image_height,
        )
    }
    pub fn add_pipeline(self, name: &str) -> PipelineBuilder {
        PipelineBuilder {
            render_graph: self,
            pipeline: None,
            name: name.to_string(),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }
    pub fn add_image(&mut self, image_desc: ImageDesc) -> ImageVersion {
        self.images.push(image_desc);

        ImageVersion {
            id: ImageId {
                arr_idx: self.images.len() - 1,
            },
            name: image_desc.get_name(),
            desc: image_desc,
            version: 0,
        }
    }
}
impl PipelineBuilder {
    pub fn pipeline(mut self, pipeline: PipelineHandle) -> PipelineBuilder {
        self.pipeline = Some(pipeline);
        self
    }
    // reads dont change image version
    pub fn reads(mut self, image_handle: &ImageVersion) -> PipelineBuilder {
        self.reads.push(image_handle.clone());
        self
    }
    // writes change image version
    pub fn writes(mut self, image_handle: &mut ImageVersion) -> PipelineBuilder {
        // a read version of 0 means its not written by any pipeline in this graph
        image_handle.version += 1;
        self.writes.push(image_handle.clone());
        self
    }
    pub fn reads_depth(mut self, image_handle: &ImageVersion) -> PipelineBuilder {
        self.reads.push(image_handle.clone());

        match self.render_graph.depth {
            None => self.render_graph.depth = Some(image_handle.id),
            Some(existing) => assert!(
                existing == image_handle.id,
                "cannot read multiple different depth buffers"
            ),
        }
        self
    }
    pub fn writes_depth(mut self, image_handle: &mut ImageVersion) -> PipelineBuilder {
        image_handle.version += 1;
        self.writes.push(image_handle.clone());

        match self.render_graph.depth {
            None => self.render_graph.depth = Some(image_handle.id),
            Some(existing) => assert!(
                existing == image_handle.id,
                "cannot write multiple different depth buffers"
            ),
        }
        self
    }

    pub fn build(mut self) -> RenderGraph {
        self.render_graph.pipelines.push(PipelineNode {
            name: self.name,
            reads: self.reads,
            writes: self.writes,
            pipeline: self
                .pipeline
                .unwrap_or_else(|| panic!("no pipeline added to pipeline")),
        });
        self.render_graph
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct CompiledRenderGraph {
    pub sampler: vk::Sampler,
    pub commands: Vec<GraphCommand>,
    image_size: vk::Extent2D,
    execution_order: Vec<usize>,
    image_states: Vec<TrackedImageState>,
    #[educe(Debug(ignore))]
    device: Arc<ash::Device>,

    depth: Option<ImageId>,
}

impl Drop for CompiledRenderGraph {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

#[derive(Educe)]
#[educe(Debug)]
struct TrackedImageState {
    current_layout: vk::ImageLayout,
    initialized: bool,
    last_access: vk::AccessFlags2,
    last_stage: vk::PipelineStageFlags2,
    format: vk::Format,
    name: &'static str,

    image: TrackedImageStorage,
}

enum TrackedImageStorage {
    Owned(Image),
    Imported {
        image: vk::Image,
        view: vk::ImageView,
    },
}
impl std::fmt::Debug for TrackedImageStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackedImageStorage::Owned(_) => write!(f, "Owned(..)"),
            TrackedImageStorage::Imported { image, view } => f
                .debug_struct("Imported")
                .field("image", image)
                .field("view", view)
                .finish(),
        }
    }
}
impl TrackedImageStorage {
    fn get_view(&self) -> vk::ImageView {
        match self {
            TrackedImageStorage::Owned(image) => image.view,
            TrackedImageStorage::Imported { view, .. } => *view,
        }
    }
}

pub struct ImportedImageBinding {
    pub pre_initialized: bool,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub current_layout: vk::ImageLayout,
    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
}

impl CompiledRenderGraph {
    pub fn get_image_info(&self, id: &ImageId) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: self.get_view_from_id(id),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
    pub fn compile(
        graph: &RenderGraph,
        device: Arc<ash::Device>,
        allocator: SharedAllocator,
        // key is the array index of the images array in rendergraph
        imported_images: &HashMap<ImageId, ImportedImageBinding>,
        image_width: u32,
        image_height: u32,
    ) -> CompiledRenderGraph {
        // TOPOLOGICAL SORT
        // map from pipeline index to degrees
        // NOTE: degrees is currently the number of total images that depend on this pipeline, not  the
        // number of pipelines that depend on this pipeline, thats fine for now
        let mut indegrees: HashMap<usize, usize> = HashMap::new();
        for (idx, _pipeline) in graph.pipelines.iter().enumerate() {
            indegrees.insert(idx, 0);
        }

        // reader dependancies
        for (writer_idx, writer_node) in graph.pipelines.iter().enumerate() {
            for write in &writer_node.writes {
                for (reader_idx, reader_node) in graph.pipelines.iter().enumerate() {
                    // if they are not the same and any reader image depends on the write, increment the indegree of the reader
                    if writer_idx != reader_idx
                        && reader_node
                            .reads
                            .iter()
                            .any(|image| image.depends_on(write))
                    {
                        *indegrees.get_mut(&reader_idx).unwrap() += 1;
                    }
                }
            }
        }

        let mut execution_order: Vec<usize> = Vec::new();
        let mut zero_nodes: Vec<usize> = indegrees
            .iter()
            .filter(|&(&_idx, &deg)| deg == 0)
            .map(|(&idx, _)| idx)
            .collect();

        while let Some(pipeline_index) = zero_nodes.pop() {
            indegrees.remove(&pipeline_index);
            execution_order.push(pipeline_index);

            for write in &graph.pipelines[pipeline_index].writes {
                for (idx, pipeline) in graph.pipelines.iter().enumerate() {
                    if pipeline.reads.iter().any(|read| read.depends_on(write)) {
                        let degree = indegrees.get_mut(&idx).unwrap();
                        *degree -= 1;
                        if *degree == 0 {
                            zero_nodes.push(idx);
                        }
                    }
                }
            }
        }
        // a cycle will always have something pointing to it, so it wont ever be added to zero
        // nodes, so never removed from indegrees
        if !indegrees.is_empty() {
            panic!("cycle detected in render graph");
        }

        let mut image_usages = HashMap::new();
        for pipeline in &graph.pipelines {
            for read in &pipeline.reads {
                *image_usages
                    .entry(read.id)
                    .or_insert(vk::ImageUsageFlags::empty()) |= vk::ImageUsageFlags::SAMPLED;
            }

            for write in &pipeline.writes {
                let flags = match &graph.images[write.id.arr_idx] {
                    ImageDesc::Managed { format, .. } | ImageDesc::Imported { format, .. } => {
                        if is_depth_format(*format) {
                            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        } else {
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                        }
                    }
                };
                *image_usages
                    .entry(write.id)
                    .or_insert(vk::ImageUsageFlags::empty()) |= flags;
            }
        }

        let mut tracked_image_states = Vec::new();
        for (idx, image_desc) in graph.images.iter().enumerate() {
            tracked_image_states.push(match image_desc {
                ImageDesc::Managed { name, format } => {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: if is_depth_format(*format) {
                            vk::ImageAspectFlags::DEPTH
                        } else {
                            vk::ImageAspectFlags::COLOR
                        },
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };

                    let image = create_image(
                        &device,
                        &allocator,
                        name,
                        vk::Extent3D {
                            width: image_width,
                            height: image_height,
                            depth: 1,
                        },
                        *format,
                        *image_usages
                            .get(&ImageId { arr_idx: idx })
                            .expect("all images should have usage flags"),
                        subresource_range,
                    );
                    TrackedImageState {
                        initialized: false,
                        format: *format,
                        name,
                        current_layout: vk::ImageLayout::UNDEFINED,
                        last_access: vk::AccessFlags2::empty(),
                        last_stage: vk::PipelineStageFlags2::empty(),
                        image: TrackedImageStorage::Owned(image),
                    }
                }
                ImageDesc::Imported { format, name } => {
                    let imported_image = imported_images
                        .get(&ImageId { arr_idx: idx })
                        .expect("all imported images should have a binding");
                    TrackedImageState {
                        initialized: imported_image.pre_initialized,
                        format: *format,
                        name,
                        current_layout: imported_image.current_layout,
                        last_access: imported_image.last_access,
                        last_stage: imported_image.last_stage,
                        image: TrackedImageStorage::Imported {
                            image: imported_image.image,
                            view: imported_image.view,
                        },
                    }
                }
            });
        }

        let mut commands = Vec::new();

        //NOTE:  Emit Commands
        for &pipeline_index in &execution_order {
            let pipeline = &graph.pipelines[pipeline_index];

            let mut color_attachments = Vec::new();
            let mut depth_attachments = Option::None;
            //BUG: if a pipeline reads and writes the same thing, this will break, but cycle
            //detection will catch that, so until thats needed its fine
            for write in pipeline.writes.iter() {
                let state = &mut tracked_image_states[write.id.arr_idx];
                //PERF: this should come from either reflection or from user declaration on the
                //.write, but im too lazy
                let (dst_stage, dst_access) = if is_depth_format(write.desc.get_format()) {
                    (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                    )
                } else {
                    (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                    )
                };

                let dst_layout = if is_depth_format(state.format) {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                };
                let aspect_mask = if is_depth_format(state.format) {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                if state.current_layout != dst_layout {
                    commands.push(GraphCommand::ImageBarrier {
                        image_id: write.id,
                        src_layout: state.current_layout,
                        dst_layout,
                        dst_stage,
                        dst_access,
                        src_stage: state.last_stage,
                        src_access: state.last_access,
                        aspect_mask,
                    });
                }
                state.current_layout = dst_layout;
                state.last_stage = dst_stage;
                state.last_access = dst_access;

                let load_op = if state.initialized {
                    vk::AttachmentLoadOp::LOAD
                } else {
                    state.initialized = true;
                    vk::AttachmentLoadOp::CLEAR
                };
                if is_depth_format(state.format) {
                    if depth_attachments.is_some() {
                        panic!("only one depth attachment allowed");
                    }
                    depth_attachments = Some(RenderingAttachmentInfo {
                        view: state.image.get_view(),
                        initial_layout: state.current_layout,
                        load_op,
                        store_op: vk::AttachmentStoreOp::STORE,
                    });
                } else {
                    color_attachments.push(RenderingAttachmentInfo {
                        view: state.image.get_view(),
                        initial_layout: state.current_layout,
                        load_op,
                        store_op: vk::AttachmentStoreOp::STORE,
                    });
                }
            }

            for read in &pipeline.reads {
                let state = &mut tracked_image_states[read.id.arr_idx];
                let (dst_layout, dst_stage, dst_access) = if is_depth_format(state.format) {
                    (
                        vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                    )
                } else {
                    (
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::ALL_GRAPHICS,
                        vk::AccessFlags2::SHADER_READ,
                    )
                };

                let aspect_mask = if is_depth_format(state.format) {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                if state.current_layout != dst_layout {
                    commands.push(GraphCommand::ImageBarrier {
                        image_id: read.id,
                        src_layout: state.current_layout,
                        dst_layout,
                        dst_access,
                        dst_stage,
                        src_access: state.last_access,
                        src_stage: state.last_stage,
                        aspect_mask,
                    });
                }
                state.current_layout = dst_layout;
                state.last_access = dst_access;
                state.last_stage = dst_stage;
                if is_depth_format(state.format) {
                    depth_attachments = Some(RenderingAttachmentInfo {
                        view: state.image.get_view(),
                        initial_layout: dst_layout,
                        load_op: vk::AttachmentLoadOp::LOAD,
                        store_op: vk::AttachmentStoreOp::NONE,
                    });
                }
            }

            // PERF: because this is simple thing now, a really important optomization later is
            // detect compatable layouts and group them without emiting a begin rendering
            commands.push(GraphCommand::BeginRendering {
                color_attachments,
                depth: depth_attachments,
            });
            commands.push(GraphCommand::BindPipeline(
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            ));
            commands.push(GraphCommand::EndRendering);
        }
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        CompiledRenderGraph {
            sampler,
            image_size: vk::Extent2D {
                width: image_width,
                height: image_height,
            },
            commands,
            execution_order,
            image_states: tracked_image_states,
            device,
            depth: graph.depth,
        }
    }
    pub fn get_image_from_id(&self, id: &ImageId) -> vk::Image {
        match &self.image_states[id.arr_idx].image {
            TrackedImageStorage::Owned(image) => image.image,
            TrackedImageStorage::Imported { image, .. } => *image,
        }
    }
    pub fn get_view_from_id(&self, id: &ImageId) -> vk::ImageView {
        match &self.image_states[id.arr_idx].image {
            TrackedImageStorage::Owned(image) => image.view,
            TrackedImageStorage::Imported { view, .. } => *view,
        }
    }
}

fn is_depth_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D16_UNORM
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT
            | vk::Format::D32_SFLOAT_S8_UINT,
    )
}

#[derive(Debug, Clone)]
pub struct RenderingAttachmentInfo {
    view: vk::ImageView,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    initial_layout: vk::ImageLayout,
}

impl RenderingAttachmentInfo {
    pub fn to_vulkan<'a>(&'a self) -> vk::RenderingAttachmentInfo<'a> {
        vk::RenderingAttachmentInfo {
            image_view: self.view,
            image_layout: self.initial_layout,
            load_op: self.load_op,
            store_op: self.store_op,
            ..Default::default()
        }
    }
}
#[derive(Debug, Clone)]
pub enum GraphCommand {
    BeginRendering {
        color_attachments: Vec<RenderingAttachmentInfo>,
        depth: Option<RenderingAttachmentInfo>,
    },
    EndRendering,
    ImageBarrier {
        image_id: ImageId,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        dst_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
        aspect_mask: vk::ImageAspectFlags,
    },
    BindPipeline(vk::PipelineBindPoint, PipelineHandle),
}

#[test]
fn build_test() {
    let mut graph = RenderGraph::new();
    let mut albedo = graph.add_image(ImageDesc::Managed {
        name: "albedo",
        format: ash::vk::Format::R8G8B8A8_SRGB,
    });
    let mut depth = graph.add_image(ImageDesc::Managed {
        name: "depth",
        format: ash::vk::Format::D32_SFLOAT,
    });
    let mut final_color = graph.add_image(ImageDesc::Imported {
        name: "final_color",
        format: vk::Format::R8G8B8A8_SRGB,
    });
    let graph = graph
        .add_pipeline("geometry_pass")
        .writes(&mut albedo)
        .writes(&mut depth)
        .build();

    let graph = graph
        .add_pipeline("lighting_pass")
        .reads(&albedo)
        .reads(&depth)
        .writes(&mut final_color)
        .build();
}
