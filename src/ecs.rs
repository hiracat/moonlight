use std::{
    any::{Any, TypeId},
    collections::{HashMap, HashSet},
    sync::Arc,
};

use thiserror::Error;
use ultraviolet::{projection, Mat4, Rotor3, Vec3, Vec4};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::PersistentDescriptorSet,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::graphics::vertex_input,
};
use winit::window::Window;

#[derive(Debug, Eq, Hash, PartialEq, Clone, Copy)]
pub struct EntityId(u32);

pub struct World {
    entities: HashSet<EntityId>,

    component_storage: HashMap<EntityId, HashMap<TypeId, Box<dyn Any>>>,
    resource_storage: HashMap<TypeId, Box<dyn Any>>,

    next_free: u32,
    last_dead: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum WorldError {
    #[error("entity not found")]
    EntityNotFound,
    #[error("entity already has component")]
    ComponentAlreadyAdded,
    #[error("the requested entity does not have the component")]
    ComponentMissing,
}

impl World {
    // public api
    pub fn init() -> Self {
        Self {
            entities: HashSet::new(),
            component_storage: HashMap::new(),
            resource_storage: HashMap::new(),

            next_free: 0,
            last_dead: Vec::new(),
        }
    }

    pub fn entity_create(&mut self) -> EntityId {
        let free = {
            if self.last_dead.is_empty() {
                let tmp = self.next_free;
                self.next_free += 1;
                tmp
            } else {
                self.last_dead.pop().unwrap() // litterally just checked so is safe
            }
        };
        self.entities.insert(EntityId(free));
        self.component_storage
            .insert(EntityId(free), HashMap::new());

        EntityId(free)
    }
    pub fn entity_destroy(&mut self, entity: EntityId) {
        self.last_dead.push(entity.0);
        self.entities.remove(&entity);
        self.component_storage.remove(&entity);
    }

    pub fn component_add<T: 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<(), WorldError> {
        match self.component_storage.get_mut(&entity) {
            Some(components) => {
                if components.contains_key(&component.type_id()) {
                    return Err(WorldError::ComponentAlreadyAdded);
                }
                components.insert(TypeId::of::<T>(), Box::new(component));
                return Ok(());
            }
            None => Err(WorldError::EntityNotFound),
        }
    }
    pub fn component_remove<T: 'static>(&mut self, entity: EntityId) -> Result<(), WorldError> {
        self.entity_component_remove(entity, &TypeId::of::<T>())
    }
    pub fn component_get<T: 'static>(&self, entity: EntityId) -> Result<&T, WorldError> {
        self.entity_component_get(&entity, &TypeId::of::<T>())
            .and_then(|component| {
                component
                    .downcast_ref::<T>()
                    .ok_or(WorldError::ComponentMissing)
            })
    }
    pub fn component_get_mut<T: 'static>(
        &mut self,
        entity: EntityId,
    ) -> Result<&mut T, WorldError> {
        self.entity_component_get_mut(&entity, &TypeId::of::<T>())
            .and_then(|component| {
                component
                    .downcast_mut::<T>()
                    .ok_or(WorldError::ComponentMissing)
            })
    }
    pub fn resource_add<T: 'static>(&mut self, resource: T) {
        self.resource_storage
            .insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn resource_get<T: 'static>(&self) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.resource_storage
            .get(&type_id)
            .and_then(|res| res.downcast_ref::<T>())
    }

    pub fn query<T: 'static>(&self) -> Vec<(EntityId, &T)> {
        let type_id = TypeId::of::<T>();
        self.entities
            .iter()
            .filter_map(|entity| {
                self.component_storage
                    .get(entity)
                    .and_then(|components| components.get(&type_id))
                    .and_then(|component| {
                        component.downcast_ref::<T>().map(|c| (entity.clone(), c))
                    })
            })
            .collect()
    }

    // impl details
    fn entity_component_remove(
        &mut self,
        entity: EntityId,
        component_type: &TypeId,
    ) -> Result<(), WorldError> {
        match self.component_storage.get_mut(&entity) {
            Some(components) => match components.remove(component_type) {
                Some(_) => Ok(()),
                None => Err(WorldError::ComponentMissing),
            },
            None => Err(WorldError::EntityNotFound),
        }
    }
    fn entity_component_get(
        &self,
        entity: &EntityId,
        component_type: &TypeId,
    ) -> Result<&dyn Any, WorldError> {
        match self.component_storage.get(&entity) {
            Some(components) => match components.get(&component_type) {
                Some(c) => return Ok(c.as_ref()),
                None => return Err(WorldError::ComponentMissing),
            },
            None => return Err(WorldError::EntityNotFound),
        }
    }
    fn entity_component_get_mut(
        &mut self,
        entity: &EntityId,
        component_type: &TypeId,
    ) -> Result<&mut dyn Any, WorldError> {
        match self.component_storage.get_mut(&entity) {
            Some(components) => match components.get_mut(&component_type) {
                Some(c) => return Ok(c.as_mut()),
                None => return Err(WorldError::ComponentMissing),
            },
            None => return Err(WorldError::EntityNotFound),
        }
    }
}

// marker

struct Transform {
    position: Vec3,
    orientation: Rotor3,
    scale: Vec3,
}
#[derive(Default)]
pub struct Camera {
    view: ultraviolet::Mat4,
    proj: ultraviolet::Mat4,

    fov_radians: f32,
    near: f32,
    far: f32,

    u_buffer: Option<Vec<Subbuffer<CameraUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}
impl Camera {
    pub fn new(
        position: Vec3,
        look_at: Vec3,
        fov: f32,
        near: f32,
        far: f32,
        window: &Arc<Window>,
    ) -> Self {
        let fov_radians = fov * (std::f32::consts::PI / 180.0);
        let size: [f32; 2] = window.inner_size().into();
        let ratio = size[0] / size[1];
        Self {
            view: ultraviolet::Mat4::look_at(
                position,
                look_at,
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            fov_radians,
            near,
            far,
            proj: projection::perspective_vk(fov_radians, ratio, near, far),
            u_buffer: None,
            descriptor_set: None,
        }
    }

    pub fn recreate(&mut self, window: &Arc<Window>, memory_allocator: &Arc<dyn MemoryAllocator>) {
        let size: [f32; 2] = window.inner_size().into();
        let ratio = size[0] / size[1];
        self.proj = projection::perspective_vk(self.fov_radians, ratio, self.near, self.far);

        for buffer in self.u_buffer.as_mut().unwrap() {
            *buffer = Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                CameraUBO {
                    proj: self.proj,
                    view: self.view,
                },
            )
            .unwrap();
        }
    }
}
#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
struct CameraUBO {
    view: ultraviolet::Mat4,
    proj: ultraviolet::Mat4,
}

#[derive(Default)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,

    u_buffer: Option<Vec<Subbuffer<AmbientLightUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
struct AmbientLightUBO {
    color: [f32; 3],
    intensity: f32,
}
impl AmbientLight {
    pub fn new(color: [f32; 3], intensity: f32) -> Self {
        Self {
            color,
            intensity,
            u_buffer: None,
            descriptor_set: None,
        }
    }
}

pub struct PointLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
    pub brightness: f32,
    pub linear: f32,
    pub quadratic: f32,

    u_buffer: Option<Vec<Subbuffer<PointLightUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
struct PointLightUBO {
    position: [f32; 4],
    color: [f32; 3],
    brightness: f32,
    linear: f32,
    quadratic: f32,
}

impl PointLight {
    pub fn new(
        position: [f32; 4],
        color: [f32; 3],
        brightness: Option<f32>,
        linear: Option<f32>,
        quadratic: Option<f32>,
    ) -> Self {
        Self {
            brightness: brightness.unwrap_or(10.0),
            linear: linear.unwrap_or(0.7),
            quadratic: quadratic.unwrap_or(1.2),
            position,
            color,
            descriptor_set: None,
            u_buffer: None,
        }
    }
}

pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],

    u_buffer: Option<Vec<Subbuffer<DirectionalLightUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
struct DirectionalLightUBO {
    position: [f32; 4],
    color: [f32; 3],
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> Self {
        Self {
            position,
            color,
            descriptor_set: None,
            u_buffer: None,
        }
    }
}

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub requires_update: bool,
    pub position: Vec4,
    pub rotation: Rotor3,

    matrix: Mat4,
    u_buffer: Option<Vec<Subbuffer<ModelUBO>>>,
    descriptor_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
    vertex_buffer: Option<Subbuffer<[Vertex]>>,
    index_buffer: Option<Subbuffer<[u32]>>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
#[derive(Debug)]
struct ModelUBO {
    model: Mat4,
    normal: Mat4,
}
impl Model {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, position: Vec4) -> Self {
        Model {
            vertices,
            indices,
            matrix: Mat4::identity(),
            rotation: Rotor3::identity(),
            requires_update: true,
            position,

            index_buffer: None,
            vertex_buffer: None,

            u_buffer: None,
            descriptor_set: None,
        }
    }
}
#[derive(vulkano::buffer::BufferContents, vertex_input::Vertex)]
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}
