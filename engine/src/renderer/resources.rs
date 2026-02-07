/// This is for engine side components, anything that interacts with the engine in some special way
/// is kept here, such as lights, meshes, textures and animations,
use std::{collections::HashMap, io::Write, path::Path};

use ash::vk;
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use image::{DynamicImage, EncodableLayout, GenericImage, ImageReader, Rgba};
use ultraviolet as uv;

use crate::renderer::draw::{
    alloc_buffers, instant_submit_command_buffer, QueueFamilyIndex, Renderer, SharedAllocator,
};

const MAX_SCENE_BONES: usize = 2048;

pub struct ResourceManager {
    device: ash::Device,
    allocator: SharedAllocator,

    meshes: Vec<Option<GpuMesh>>,
    textures: Vec<Option<GpuTexture>>,

    pub(crate) animation_resources: AnimationResources,

    default_texture: GpuTexture,

    // is a ring buffer for all
    pub(crate) ubo_ring_buffer: UniformRingBuffer,

    //NOTE: unowned resources, just handles for reference/creating without requiring renderer to be
    //passed in
    queue_family_index: QueueFamilyIndex,
    queue: vk::Queue,
    one_time_submit_buffer: vk::CommandBuffer,
    one_time_submit_pool: vk::CommandPool,
}

#[derive(Debug)]
pub struct Skeleton {
    // TODO:just the index into the skeletons array(need to find a way to remove, but might just have to
    // waste some memory, or use option and waste some indices
    pub(crate) id: usize,
}
#[derive(Debug, Clone)]
pub struct Animation {
    // the index of the inner vec, so accessing an animation requires a skeleton and an animation
    pub(crate) id: usize,
    name: String,
}
pub(crate) struct AnimationResources {
    // store all skeletons and bones, then flatten to upload to the gpu.
    pub(crate) skeletons: Vec<SkeletonImpl>,
    // use the same indices as the skeletons, with multiple animations per skeleton possible
    pub(crate) animations: Vec<Vec<AnimationImpl>>,

    pub(crate) skeleton_transform_buffer: vk::Buffer,
    pub(crate) skeleton_transform_allocation: Allocation,

    pub(crate) skeleton_normal_buffer: vk::Buffer,
    pub(crate) skeleton_normal_allocation: Allocation,
}
impl AnimationResources {
    pub(crate) fn write_bones(&mut self) {
        let transform_matrices: Vec<uv::Mat4> = self
            .skeletons
            .iter()
            .flat_map(|x| &x.joints)
            .map(|x| x.global_transform * x.inverse_bind_matrix)
            .collect();

        let normal_matrices: Vec<uv::Mat3> = transform_matrices
            .iter()
            .map(|x| {
                uv::Mat3::new(x.cols[0].xyz(), x.cols[1].xyz(), x.cols[2].xyz())
                    .inversed()
                    .transposed()
            })
            .collect();

        write(
            &transform_matrices,
            self.skeleton_transform_allocation
                .mapped_slice_mut()
                .unwrap(),
            0,
        );

        write(
            &normal_matrices,
            self.skeleton_normal_allocation.mapped_slice_mut().unwrap(),
            0,
        );
    }
    fn create(device: &ash::Device, memory_allocator: SharedAllocator) -> Self {
        let transform_size = (size_of::<uv::Mat4>() * MAX_SCENE_BONES) as u64;
        let normal_size = (size_of::<uv::Mat3>() * MAX_SCENE_BONES) as u64;
        let (mut transform_buffer, mut transform_allocation) = alloc_buffers(
            memory_allocator.clone(),
            1,
            transform_size,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            // leave data uninitialized, we update it later
            cast_slice(&[uv::Mat4::identity()]),
            "transform memory",
        );
        let (mut normal_buffer, mut normal_allocation) = alloc_buffers(
            memory_allocator.clone(),
            1,
            normal_size,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            // leave data uninitialized, we update it later
            cast_slice(&[uv::Mat3::identity()]),
            "normal memory",
        );

        Self {
            skeleton_transform_buffer: transform_buffer.pop().unwrap(),
            skeleton_transform_allocation: transform_allocation.pop().unwrap(),
            skeleton_normal_buffer: normal_buffer.pop().unwrap(),
            skeleton_normal_allocation: normal_allocation.pop().unwrap(),
            skeletons: Vec::new(),
            animations: Vec::new(),
        }
    }
}

// marks a entity as having animations, and stores references to the animations(should be
// reconstructable from a file path, but thats deferred til serialization)
// referenced https://www.youtube.com/watch?v=da6d28IylL8 to make this, and https://whoisryosuke.com/blog/2022/importing-gltf-with-wgpu-and-rust
#[derive(Debug)]
pub struct Animated {
    //PERF: this could possible be put into the resource manager since it can be large, but not
    //necesary for now
    pub time: f32,
    pub animations: Vec<Animation>,
    pub current_playing: Option<Animation>,
    pub skeleton: Skeleton,
}

#[derive(Debug)]
pub(crate) struct SkeletonImpl {
    pub(crate) joints: Vec<Joint>,
}

impl SkeletonImpl {
    pub(crate) fn update_global_transforms(&mut self) {
        // joints is built so that parents are always before children
        for current_index in 0..self.joints.len() {
            if self.joints[current_index].parent_index == usize::MAX {
                self.joints[current_index].global_transform =
                    self.joints[current_index].get_current_local_transform();
            } else {
                self.joints[current_index].global_transform =
                    self.joints[self.joints[current_index].parent_index].global_transform
                        * self.joints[current_index].get_current_local_transform();
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct Joint {
    // set to usize::MAX for the root, but is really undefined
    parent_index: usize,
    children_indices: Vec<usize>,

    // converts from world/model space to bone/joint space
    pub(crate) inverse_bind_matrix: uv::Mat4,

    // the position relative to the parent, edited by the animation channels
    pub(crate) position: uv::Vec3,
    pub(crate) rotation: uv::Rotor3,
    pub(crate) scale: uv::Vec3,

    // the transform of this joint in world space, accounting for animations
    pub(crate) global_transform: uv::Mat4,
}

impl Joint {
    fn get_current_local_transform(&self) -> uv::Mat4 {
        uv::Mat4::from_translation(self.position)
            * self.rotation.into_matrix().into_homogeneous()
            * uv::Mat4::from_nonuniform_scale(self.scale)
    }
}

#[derive(Debug)]
pub(crate) enum Keyframes {
    Translation(Vec<uv::Vec3>),
    Rotation(Vec<uv::Rotor3>),
    Scale(Vec<uv::Vec3>),
}

#[derive(Debug)]
pub(crate) struct AnimationImpl {
    name: String,

    pub(crate) channels: Vec<AnimationChannel>,
}

#[derive(Debug)]
pub(crate) struct AnimationChannel {
    pub(crate) keyframes: Keyframes,
    pub(crate) timestamps: Vec<f32>,
    pub(crate) target_joint_index: usize,
}

fn load_joints(
    skin: &gltf::Skin,
    buffers: &Vec<gltf::buffer::Data>,
) -> (Vec<Joint>, HashMap<usize, usize>, HashMap<usize, usize>) {
    let inverse_bind_accessor = skin.inverse_bind_matrices().unwrap();
    let mut gltf_node_to_engine = HashMap::new();
    let mut gltf_joint_to_engine = HashMap::new();

    let mut gltf_node_to_gltf_joint: HashMap<usize, usize> = HashMap::new();

    for (idx, joint) in skin.joints().enumerate() {
        gltf_node_to_gltf_joint.insert(joint.index(), idx);
    }

    let inverse_bind_matrices: Vec<uv::Mat4> = if let Some(iter) =
        gltf::accessor::Iter::<[[f32; 4]; 4]>::new(inverse_bind_accessor, |buffer| {
            // map here transforms the option if it exists, basically just data.0
            // without unwrapping
            buffers.get(buffer.index()).map(|data| &data.0[..])
        }) {
        iter.map(|matrix_array| uv::Mat4::from(matrix_array))
            .collect()
    } else {
        panic!("Failed to read inverse bind matrices");
    };

    let mut joint_hierarchy = Vec::new();

    if let Some(root) = skin.skeleton() {
        build_hierarchy(
            &root,
            usize::MAX,
            &inverse_bind_matrices,
            &mut joint_hierarchy,
            &mut gltf_node_to_engine,
            &mut gltf_joint_to_engine,
            &gltf_node_to_gltf_joint,
        );
    } else {
        let mut child_indices = std::collections::HashSet::new();
        for joint in skin.joints() {
            for child in joint.children() {
                if gltf_node_to_gltf_joint.contains_key(&child.index()) {
                    child_indices.insert(child.index());
                }
            }
        }

        let mut only_one_root = true;
        for joint in skin.joints() {
            if !child_indices.contains(&joint.index()) {
                if only_one_root {
                    build_hierarchy(
                        &joint,
                        usize::MAX,
                        &inverse_bind_matrices,
                        &mut joint_hierarchy,
                        &mut gltf_node_to_engine,
                        &mut gltf_joint_to_engine,
                        &gltf_node_to_gltf_joint,
                    );
                    only_one_root = false;
                } else {
                    panic!("skeleton should only have one root");
                }
            }
        }
    };

    (joint_hierarchy, gltf_node_to_engine, gltf_joint_to_engine)
}

fn build_hierarchy(
    node: &gltf::Node,
    parent_index: usize,
    inverse_bind_matrices: &Vec<uv::Mat4>,
    joints: &mut Vec<Joint>,
    gltf_node_to_engine: &mut HashMap<usize, usize>,
    gltf_joint_to_engine: &mut HashMap<usize, usize>,
    gltf_node_to_gltf_joint: &HashMap<usize, usize>,
) {
    let current_joint_index = joints.len();

    // Add parent's child index
    if parent_index != usize::MAX {
        joints[parent_index]
            .children_indices
            .push(current_joint_index);
    }
    gltf_node_to_engine.insert(node.index(), current_joint_index);

    gltf_joint_to_engine.insert(
        *gltf_node_to_gltf_joint
            .get(&node.index())
            .expect("should have all nodes"),
        current_joint_index,
    );

    let (position, rotation, scale) = node.transform().decomposed();
    joints.push(Joint {
        position: position.into(),
        rotation: uv::Rotor3::from_quaternion_array(rotation),
        scale: scale.into(),
        // needs to be filled later
        children_indices: Vec::new(),
        inverse_bind_matrix: inverse_bind_matrices[*gltf_node_to_gltf_joint
            .get(&node.index())
            .expect("all joints should have an entry")],
        // needs to be set later
        parent_index: parent_index,
        // this will be updated by update_global_transforms
        global_transform: uv::Mat4::identity(),
    });

    for child in node.children() {
        build_hierarchy(
            &child,
            current_joint_index,
            inverse_bind_matrices,
            joints,
            gltf_node_to_engine,
            gltf_joint_to_engine,
            gltf_node_to_gltf_joint,
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub albedo: Texture,
}

#[derive(Debug, Clone, Copy)]
pub struct Texture {
    id: usize,
    path: Option<&'static str>,
}
#[derive(Debug, Clone, Copy)]
pub struct Skybox {
    pub(crate) material: Material,
}
impl Skybox {
    pub fn new(cubemap: Texture) -> Self {
        Self {
            material: Material { albedo: cubemap },
        }
    }
}
impl Material {
    pub fn create(albedo: Texture) -> Self {
        Self { albedo }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mesh {
    id: usize,
    pub(crate) animated: bool,
    path: Option<&'static str>,
}

#[derive(Debug)]
pub enum VertexType {
    Static,
    Animated,
}

pub trait IsVertex {
    fn get_type() -> VertexType;
    fn get_vertex_attributes() -> Vec<vk::VertexInputAttributeDescription>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct Vertex {
    pub position: uv::Vec3,
    pub normal: uv::Vec3,
    pub uv: uv::Vec2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct AnimatedVertex {
    pub position: uv::Vec3,
    pub normal: uv::Vec3,
    pub uv: uv::Vec2,
    pub bone_indices: [u32; 4],
    pub bone_weights: uv::Vec4,
}

impl IsVertex for AnimatedVertex {
    fn get_type() -> VertexType {
        VertexType::Animated
    }
    fn get_vertex_attributes() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            // position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0x0,
            },
            // normal
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0xC,
            },
            // uv
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0x18,
            },
            // indices
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_UINT,
                offset: 0x20,
            },
            // weights
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0x30,
            },
        ]
    }
}

impl IsVertex for Vertex {
    fn get_vertex_attributes() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            // position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0x0,
            },
            // normal
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0xC,
            },
            // uv
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0x18,
            },
        ]
    }
    fn get_type() -> VertexType {
        VertexType::Static
    }
}

pub(crate) struct GpuTexture {
    pub(crate) image: vk::Image,
    pub(crate) image_view: vk::ImageView,
    pub(crate) sampler: vk::Sampler,
    pub(crate) memory: Allocation,
}
#[derive(Debug)]
pub(crate) struct GpuMesh {
    pub(crate) vertex_buffer: vk::Buffer,
    pub(crate) index_buffer: vk::Buffer,
    pub(crate) index_alloc: Allocation,
    pub(crate) vertex_alloc: Allocation,
    pub(crate) index_count: u32,
    pub(crate) vertex_type: VertexType,
}

pub struct UniformRingBuffer {
    buffer: vk::Buffer,
    memory: Allocation,
    size: u64,
    alignment: usize,
    current: u64,
}
impl UniformRingBuffer {
    fn create(allocator: SharedAllocator, device: &ash::Device, size: u64) -> Self {
        let sharing_mode = vk::SharingMode::EXCLUSIVE;
        let mut buffers = alloc_buffers(
            allocator,
            1,
            size,
            device,
            sharing_mode,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            &vec![0u8; size as usize],
            "ring buffer",
        );
        let buffer = buffers.0.remove(0);
        let allocation = buffers.1.remove(0);
        let alignment = 64;

        Self {
            buffer: buffer,
            memory: allocation,
            size: size,
            alignment: alignment,
            current: 0,
        }
    }
    pub fn allocate(&mut self, size: usize) -> (vk::Buffer, u64) {
        let overflow = size % self.alignment;
        let padding = self.alignment - overflow;
        let aligned_size = size + padding;
        if self.current + aligned_size as u64 > self.size {
            eprintln!(
                "Ring buffer wrapping! current={}, size={}, buffer_size={}",
                self.current, aligned_size, self.size
            );
            self.current = 0;
        }

        let offset = self.current;
        self.current += aligned_size as u64;

        assert_eq!(self.current as u64 % self.alignment as u64, 0);
        assert_eq!(aligned_size as u64 % self.alignment as u64, 0);
        (self.buffer, offset)
    }
    pub fn write<T: Pod + Zeroable>(&mut self, data: &T, offset: u64) {
        let offset = offset as usize;
        let data = bytes_of(data);
        let size = size_of::<T>();

        let memory = self
            .memory
            .mapped_slice_mut()
            .expect("memory should be host visable");
        let data_segment = &mut memory[offset..offset + size];
        data_segment.copy_from_slice(data);
    }
}

/// offset is in number of T
pub fn write<T: Pod + Zeroable>(from: &[T], to: &mut [u8], offset: usize) {
    let offset_bytes = offset * size_of::<T>();
    let data: &[u8] = cast_slice(from);

    assert!(
        offset_bytes + data.len() <= to.len(),
        "write failed because of a buffer overflow"
    );

    let target_slice = &mut to[offset_bytes..offset_bytes + data.len()];
    target_slice.copy_from_slice(data);
}

impl AnimatedVertex {
    pub fn new(
        postion: uv::Vec3,
        normal: uv::Vec3,
        uv: uv::Vec2,
        bone_weights: uv::Vec4,
        bone_indices: [u32; 4],
    ) -> Self {
        Self {
            position: postion,
            normal: normal,
            uv: uv,
            bone_weights: bone_weights,
            bone_indices: bone_indices,
        }
    }
}
impl Vertex {
    pub fn new(postion: uv::Vec3, normal: uv::Vec3, uv: uv::Vec2) -> Self {
        Self {
            position: postion,
            normal: normal,
            uv: uv,
        }
    }
}

impl ResourceManager {
    pub(crate) fn init(
        device: &ash::Device,
        allocator: SharedAllocator,
        one_time_submit_pool: vk::CommandPool,
        one_time_submit_buffer: vk::CommandBuffer,
        queue: vk::Queue,
        queue_family_index: QueueFamilyIndex,
    ) -> Self {
        let ring_buffer = UniformRingBuffer::create(allocator.clone(), device, 0xFFFFF);
        let pixel = Rgba::from([255, 255, 255, 0]);
        let mut image = DynamicImage::new_rgb8(1, 1);
        image.put_pixel(0, 0, pixel);
        let default = GpuTexture::create_2d(
            device,
            queue_family_index,
            allocator.clone(),
            &image,
            queue,
            one_time_submit_buffer,
            one_time_submit_pool,
        );

        ResourceManager {
            ubo_ring_buffer: ring_buffer,
            default_texture: default,
            device: device.clone(),
            allocator: allocator.clone(),
            meshes: Vec::new(),
            textures: Vec::new(),
            animation_resources: AnimationResources::create(device, allocator),
            one_time_submit_buffer,
            queue,
            queue_family_index,
            one_time_submit_pool,
        }
    }
    pub fn load_animations(
        &mut self,
        document: &gltf::Document,
        buffers: &Vec<gltf::buffer::Data>,
    ) -> (Animated, HashMap<usize, usize>) {
        assert_eq!(document.skins().len(), 1);

        let skin = document.skins().next().unwrap();
        let (joints, gltf_node_to_engine, gltf_joint_to_engine) = load_joints(&skin, &buffers);

        let mut skeleton = SkeletonImpl { joints };
        skeleton.update_global_transforms();

        let mut animation_clips = Vec::new();
        for animation in document.animations() {
            let mut animation_channels = Vec::new();
            for channel in animation.channels() {
                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
                let timestamps = if let Some(inputs) = reader.read_inputs() {
                    match inputs {
                        gltf::accessor::Iter::Standard(times) => {
                            let times: Vec<f32> = times.collect();
                            times
                        }
                        gltf::accessor::Iter::Sparse(times) => {
                            let times: Vec<f32> = times.collect();
                            times
                        }
                    }
                } else {
                    panic!("animation has no keyframes");
                };

                let keyframes = if let Some(outputs) = reader.read_outputs() {
                    match outputs {
                        gltf::animation::util::ReadOutputs::Translations(translations) => {
                            let translations_vec: Vec<_> =
                                translations.map(|x| uv::Vec3::from(x)).collect();
                            assert_eq!(translations_vec.len(), timestamps.len());
                            Keyframes::Translation(translations_vec)
                        }
                        gltf::animation::util::ReadOutputs::Rotations(rotations) => {
                            let rotations_vec = rotations
                                .into_f32()
                                .map(|x| uv::Rotor3::from_quaternion_array(x))
                                .collect();
                            Keyframes::Rotation(rotations_vec)
                        }
                        gltf::animation::util::ReadOutputs::Scales(scales) => {
                            let scales_vec = scales.map(|x| uv::Vec3::from(x)).collect();
                            Keyframes::Scale(scales_vec)
                        }
                        gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => {
                            unimplemented!()
                        }
                    }
                } else {
                    panic!("unexpected reader output")
                };

                animation_channels.push(AnimationChannel {
                    target_joint_index: *gltf_node_to_engine
                        .get(&channel.target().node().index())
                        .expect("should have value for all nodes accessed"),
                    keyframes: keyframes,
                    timestamps: timestamps,
                })
            }
            animation_clips.push(AnimationImpl {
                name: animation.name().unwrap_or("no name").to_string(),
                channels: animation_channels,
            });
        }

        let skeleton_id = Skeleton {
            id: self.animation_resources.skeletons.len(),
        };

        let animation_ids = animation_clips
            .iter()
            .enumerate()
            .map(|(idx, animation)| Animation {
                name: animation.name.clone(),
                id: idx,
            })
            .collect();

        self.animation_resources.skeletons.push(skeleton);
        self.animation_resources.animations.push(animation_clips);

        let transform_matrices: Vec<uv::Mat4> = self
            .animation_resources
            .skeletons
            .iter()
            .flat_map(|x| &x.joints)
            .map(|x| x.global_transform * x.inverse_bind_matrix)
            .collect();

        let normal_matrices: Vec<uv::Mat3> = transform_matrices
            .iter()
            .map(|x| {
                uv::Mat3::new(x.cols[0].xyz(), x.cols[1].xyz(), x.cols[2].xyz())
                    .inversed()
                    .transposed()
            })
            .collect();

        write(
            &transform_matrices,
            self.animation_resources
                .skeleton_transform_allocation
                .mapped_slice_mut()
                .unwrap(),
            0,
        );

        write(
            &normal_matrices,
            self.animation_resources
                .skeleton_normal_allocation
                .mapped_slice_mut()
                .unwrap(),
            0,
        );

        (
            Animated {
                time: 0.0,
                animations: animation_ids,
                current_playing: None,
                skeleton: skeleton_id,
            },
            gltf_joint_to_engine,
        )
    }

    pub(crate) fn default_texture(&mut self) -> &GpuTexture {
        &self.default_texture
    }
    pub fn create_texture(&mut self, path: &'static str) -> Texture {
        let image = ImageReader::open(Path::new(path))
            .unwrap()
            .decode()
            .unwrap();
        self.textures.push(Some(GpuTexture::create_2d(
            &self.device,
            self.queue_family_index,
            self.allocator.clone(),
            &image,
            self.queue,
            self.one_time_submit_buffer,
            self.one_time_submit_pool,
        )));
        Texture {
            id: self.textures.len() - 1,
            path: Some(path),
        }
    }
    pub fn create_cubemap(&mut self, paths: [&'static str; 6]) -> Texture {
        let images: [DynamicImage; 6] = paths.map(|path| {
            ImageReader::open(Path::new(path))
                .unwrap()
                .decode()
                .unwrap()
        });
        self.textures.push(Some(GpuTexture::create_cubemap(
            &self.device,
            self.queue_family_index,
            self.allocator.clone(),
            images.into(),
            self.queue,
            self.one_time_submit_buffer,
            self.one_time_submit_pool,
        )));
        Texture {
            id: self.textures.len() - 1,
            //HACK: need to either seperate different texture types out(probably smartest and
            //easiest, or figure out a way to do multiple path types per texture(maybe enum)
            path: None,
        }
    }
    pub fn load_gltf_asset(&mut self, path: &'static str) -> (Mesh, Option<Animated>) {
        let (document, buffers, _images) = gltf::import(path).unwrap();
        assert_eq!(document.meshes().len(), 1);
        let mesh = &document.meshes().next().unwrap();
        let primative: &gltf::Primitive = &mesh.primitives().next().unwrap();

        let gpu_mesh;
        let animated;
        let is_animated;
        if document.skins().len() > 0 {
            let offset = self
                .animation_resources
                .skeletons
                .iter()
                .map(|x| x.joints.len())
                .sum();
            // this adds to self.animation_resources.skeletons, so do it after computing offset
            let (animations, gltf_joints_to_engine) = self.load_animations(&document, &buffers);

            gpu_mesh = Self::load_animated_mesh(
                primative,
                &buffers,
                self.allocator.clone(),
                &self.device,
                &gltf_joints_to_engine,
                offset,
            );
            animated = Some(Animated {
                current_playing: None,
                animations: animations.animations,
                skeleton: animations.skeleton,
                time: 0.0,
            });
            is_animated = true;
        } else {
            gpu_mesh =
                Self::load_static_mesh(primative, &buffers, self.allocator.clone(), &self.device);
            animated = None;
            is_animated = false;
        }
        self.meshes.push(Some(gpu_mesh));

        (
            Mesh {
                id: self.meshes.len() - 1,
                path: Some(path),
                animated: is_animated,
            },
            animated,
        )
    }

    fn load_static_mesh(
        primative: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>,
        allocator: SharedAllocator,
        device: &ash::Device,
    ) -> GpuMesh {
        let reader = primative.reader(|buffer| Some(&buffers[buffer.index()]));

        let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();
        let positions = reader.read_positions().unwrap();
        let normals = reader.read_normals().unwrap();
        let tex_coords = reader.read_tex_coords(0).unwrap().into_f32();

        let vertices: Vec<_> = positions
            .zip(normals)
            .zip(tex_coords)
            .map(|((position, normal), uv)| Vertex {
                position: position.into(),
                normal: normal.into(),
                uv: uv.into(),
            })
            .collect();

        let (vertex_buffer, mut vertex_alloc) = alloc_buffers(
            allocator.clone(),
            1,
            (vertices.len() * size_of::<Vertex>()) as u64,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(vertices.as_ref()),
            "vertex buffer",
        );

        let (index_buffer, mut index_alloc) = alloc_buffers(
            allocator.clone(),
            1,
            indices.len() as u64 * 4,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(indices.as_ref()),
            "index buffer",
        );

        let mesh = GpuMesh {
            index_alloc: index_alloc.remove(0),
            vertex_alloc: vertex_alloc.remove(0),
            index_count: indices.len() as u32,
            index_buffer: index_buffer[0],
            vertex_buffer: vertex_buffer[0],
            vertex_type: VertexType::Static,
        };
        mesh
    }

    fn load_animated_mesh(
        primative: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>,
        allocator: SharedAllocator,
        device: &ash::Device,
        gltf_joint_to_engine: &HashMap<usize, usize>,
        offset_for_prev_arrays: usize,
    ) -> GpuMesh {
        let reader = primative.reader(|buffer| Some(&buffers[buffer.index()]));

        let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();
        let positions = reader.read_positions().unwrap();
        let normals = reader.read_normals().unwrap();
        let tex_coords = reader.read_tex_coords(0).unwrap().into_f32();

        let bone_weights = reader.read_weights(0).unwrap().into_f32();
        let raw_bone_indices = reader.read_joints(0).unwrap().into_u16();

        dbg!(gltf_joint_to_engine);
        let translated_bone_indices: Vec<[u32; 4]> = raw_bone_indices
            .map(|x| {
                x.map(|x| {
                    *gltf_joint_to_engine.get(&(x as usize)).unwrap() as u32
                        + offset_for_prev_arrays as u32
                })
            })
            .collect();

        let vertices: Vec<_> = positions
            .zip(normals)
            .zip(tex_coords)
            .zip(bone_weights)
            .zip(translated_bone_indices)
            .map(
                |((((position, normal), uv), bone_weights), bone_indices)| AnimatedVertex {
                    position: position.into(),
                    normal: normal.into(),
                    uv: uv.into(),
                    bone_weights: bone_weights.into(),
                    bone_indices: bone_indices,
                },
            )
            .collect();

        let (vertex_buffer, mut vertex_alloc) = alloc_buffers(
            allocator.clone(),
            1,
            (vertices.len() * size_of::<AnimatedVertex>()) as u64,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(vertices.as_ref()),
            "vertex buffer",
        );

        let (index_buffer, mut index_alloc) = alloc_buffers(
            allocator.clone(),
            1,
            indices.len() as u64 * 4,
            &device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(indices.as_ref()),
            "index buffer",
        );

        let mesh = GpuMesh {
            index_alloc: index_alloc.remove(0),
            vertex_alloc: vertex_alloc.remove(0),
            index_count: indices.len() as u32,
            index_buffer: index_buffer[0],
            vertex_buffer: vertex_buffer[0],
            vertex_type: VertexType::Animated,
        };
        mesh
    }

    pub(crate) fn get_mesh(&mut self, mesh: Mesh) -> Option<&GpuMesh> {
        self.meshes.get(mesh.id).and_then(|m| m.as_ref())
    }
    pub(crate) fn get_texture(&mut self, tex: Texture) -> Option<&GpuTexture> {
        self.textures.get(tex.id).and_then(|m| m.as_ref())
    }
    pub(crate) fn allocate_temp_descriptor_set(
        &mut self,
        set_layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
    ) -> vk::DescriptorSet {
        let allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            p_set_layouts: &set_layout,
            descriptor_set_count: 1,
            ..Default::default()
        };
        unsafe {
            self.device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()[0]
        }
    }
}

impl GpuTexture {
    pub fn update_subimage(
        &mut self,
        patch: DynamicImage,
        offset_x: usize,
        offset_y: usize,
        device: &ash::Device,
        allocator: SharedAllocator,
        queue_family_index: QueueFamilyIndex,
        queue: vk::Queue,
        one_time_submit_buffer: vk::CommandBuffer,
        one_time_submit_pool: vk::CommandPool,
    ) {
        let image = patch.to_rgba8();
        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            size: (image.width() * image.height() * 4) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image patch staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        staging_mem
            .mapped_slice_mut()
            .unwrap()
            .write(image.as_bytes())
            .unwrap();

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        instant_submit_command_buffer(
            &device,
            one_time_submit_buffer,
            one_time_submit_pool,
            queue,
            |command_buffer| {
                let to_writable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: self.image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_writable],
                    )
                };

                let regions = [vk::BufferImageCopy {
                    image_offset: vk::Offset3D {
                        x: offset_x as i32,
                        y: offset_y as i32,
                        z: 0,
                    },
                    image_subresource: vk::ImageSubresourceLayers {
                        mip_level: 0,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        base_array_layer: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: patch.width(),
                        height: patch.height(),
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                }];

                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer,
                        self.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    )
                };
                let to_readable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    image: self.image,
                    subresource_range,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                };
                //barrier the image into the shader readable layout
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_readable],
                    )
                };
            },
        );

        unsafe { device.destroy_buffer(staging_buffer, None) };
        allocator.lock().unwrap().free(staging_mem).unwrap();
    }
    pub fn create_cubemap(
        device: &ash::Device,
        queue_family_index: QueueFamilyIndex,
        allocator: SharedAllocator,
        mut images: [DynamicImage; 6],
        queue: vk::Queue,
        one_time_submit_buffer: vk::CommandBuffer,
        one_time_submit_pool: vk::CommandPool,
    ) -> Self {
        let image_bytes: Vec<_> = images
            .iter_mut()
            .map(|x| x.to_rgba8().into_raw())
            .flatten()
            .collect();

        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            // 4 bits per pixel * 6 images
            size: (images[0].width() * images[0].height() * 4 * 6) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        let image_format = vk::Format::R8G8B8A8_SRGB;

        let create_info = vk::ImageCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            extent: vk::Extent3D {
                width: images[0].width(),
                height: images[0].height(),
                depth: 1,
            },
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format: image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 6,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let final_image = unsafe { device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_image_memory_requirements(final_image) };

        let final_image_mem_desc = AllocationCreateDesc {
            name: "image memory",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let image_mem = allocator
            .lock()
            .unwrap()
            .allocate(&final_image_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_image_memory(final_image, image_mem.memory(), image_mem.offset())
                .unwrap()
        }

        staging_mem
            .mapped_slice_mut()
            .unwrap()
            .write(&image_bytes)
            .unwrap();

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        };
        instant_submit_command_buffer(
            &device,
            one_time_submit_buffer,
            one_time_submit_pool,
            queue,
            |command_buffer| {
                let to_writable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: final_image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_writable],
                    )
                };

                let regions = [vk::BufferImageCopy {
                    image_offset: vk::Offset3D::default(),
                    image_subresource: vk::ImageSubresourceLayers {
                        mip_level: 0,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 6,
                        base_array_layer: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: images[0].width(),
                        height: images[0].height(),
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                }];

                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer,
                        final_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    )
                };
                let to_readable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    image: final_image,
                    subresource_range,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                };
                //barrier the image into the shader readable layout
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_readable],
                    )
                };
            },
        );

        unsafe { device.destroy_buffer(staging_buffer, None) };
        allocator.lock().unwrap().free(staging_mem).unwrap();

        let sampler_create_info = vk::SamplerCreateInfo {
            flags: vk::SamplerCreateFlags::empty(),

            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,

            mipmap_mode: vk::SamplerMipmapMode::LINEAR,

            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,

            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,

            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,

            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE, // or the max mip level of your texture

            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,

            ..Default::default()
        };

        let sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: final_image,
            format: image_format,
            view_type: vk::ImageViewType::CUBE,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range,
            ..Default::default()
        };
        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        Self {
            image_view,
            image: final_image,
            memory: image_mem,
            sampler,
        }
    }
    pub fn create_2d(
        device: &ash::Device,
        queue_family_index: QueueFamilyIndex,
        allocator: SharedAllocator,
        image: &DynamicImage,
        queue: vk::Queue,
        one_time_submit_buffer: vk::CommandBuffer,
        one_time_submit_pool: vk::CommandPool,
    ) -> Self {
        let image = image.to_rgba8();
        let create_info = vk::BufferCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            size: (image.width() * image.height() * 4) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            flags: vk::BufferCreateFlags::empty(),
            ..Default::default()
        };
        let staging_buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_mem_desc = AllocationCreateDesc {
            name: "image staging buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut staging_mem = allocator
            .lock()
            .unwrap()
            .allocate(&staging_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_mem.memory(), staging_mem.offset())
                .unwrap()
        }

        let image_format = vk::Format::R8G8B8A8_SRGB;

        let create_info = vk::ImageCreateInfo {
            p_queue_family_indices: &queue_family_index,
            queue_family_index_count: 1,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            extent: vk::Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1,
            },
            flags: vk::ImageCreateFlags::empty(),
            format: image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let final_image = unsafe { device.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.get_image_memory_requirements(final_image) };

        let final_image_mem_desc = AllocationCreateDesc {
            name: "image memory",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let image_mem = allocator
            .lock()
            .unwrap()
            .allocate(&final_image_mem_desc)
            .unwrap();

        unsafe {
            device
                .bind_image_memory(final_image, image_mem.memory(), image_mem.offset())
                .unwrap()
        }

        staging_mem
            .mapped_slice_mut()
            .unwrap()
            .write(image.as_bytes())
            .unwrap();

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        instant_submit_command_buffer(
            &device,
            one_time_submit_buffer,
            one_time_submit_pool,
            queue,
            |command_buffer| {
                let to_writable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: final_image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_writable],
                    )
                };

                let regions = [vk::BufferImageCopy {
                    image_offset: vk::Offset3D::default(),
                    image_subresource: vk::ImageSubresourceLayers {
                        mip_level: 0,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        base_array_layer: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: image.width(),
                        height: image.height(),
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                }];

                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer,
                        final_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    )
                };
                let to_readable = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    image: final_image,
                    subresource_range,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                };
                //barrier the image into the shader readable layout
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_readable],
                    )
                };
            },
        );

        unsafe { device.destroy_buffer(staging_buffer, None) };
        allocator.lock().unwrap().free(staging_mem).unwrap();

        let sampler_create_info = vk::SamplerCreateInfo {
            flags: vk::SamplerCreateFlags::empty(),

            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,

            mipmap_mode: vk::SamplerMipmapMode::LINEAR,

            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,

            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,

            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,

            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE, // or the max mip level of your texture

            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,

            ..Default::default()
        };

        let sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: final_image,
            format: image_format,
            view_type: vk::ImageViewType::TYPE_2D,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range,
            ..Default::default()
        };
        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        Self {
            image_view,
            image: final_image,
            memory: image_mem,
            sampler,
        }
    }
}

impl GpuMesh {
    pub fn create<V: IsVertex + Pod>(
        renderer: &mut Renderer,
        vertices: &[V],
        indices: &[u32],
    ) -> Self {
        let (mut vertex_buffers, mut vertex_allocs) = alloc_buffers(
            renderer.allocator.clone(),
            1,
            vertices.len() as u64 * size_of::<Vertex>() as u64,
            &renderer.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(vertices.as_ref()),
            "vertex buffer",
        );

        let (mut index_buffers, mut index_allocs) = alloc_buffers(
            renderer.allocator.clone(),
            1,
            indices.len() as u64 * 4,
            &renderer.device,
            vk::SharingMode::EXCLUSIVE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            bytemuck::cast_slice(indices.as_ref()),
            "index buffer",
        );

        Self {
            index_count: indices.len() as u32,
            index_alloc: index_allocs.pop().unwrap(),
            index_buffer: index_buffers.pop().unwrap(),
            vertex_alloc: vertex_allocs.pop().unwrap(),
            vertex_buffer: vertex_buffers.pop().unwrap(),
            vertex_type: V::get_type(),
        }
    }
}
