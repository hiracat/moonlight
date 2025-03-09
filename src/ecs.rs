#![allow(dead_code)]
use std::{
    any::{Any, TypeId},
    collections::{HashMap, HashSet},
};

use thiserror::Error;

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
                Ok(())
            }
            None => Err(WorldError::EntityNotFound),
        }
    }
    pub fn component_remove<T: 'static>(&mut self, entity: EntityId) -> Result<(), WorldError> {
        self.entity_component_remove(entity, &TypeId::of::<T>())
    }
    pub fn component_get<T: 'static>(&self, entity: EntityId) -> Result<&T, WorldError> {
        self.entity_component_get(entity, &TypeId::of::<T>())
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
        self.entity_component_get_mut(entity, &TypeId::of::<T>())
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

    pub fn resource_get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.resource_storage
            .get_mut(&type_id)
            .and_then(|res| res.downcast_mut::<T>())
    }

    pub fn query_entities<T: 'static>(&self) -> Vec<EntityId> {
        let type_id = TypeId::of::<T>();
        self.entities
            .iter()
            .filter(|entity| {
                self.component_storage
                    .get(entity)
                    .is_some_and(|components| components.contains_key(&type_id))
            })
            .copied()
            .collect()
    }
    pub fn query<T: 'static>(&self) -> Vec<(EntityId, &T)> {
        let type_id = TypeId::of::<T>();
        self.entities
            .iter()
            .filter_map(|entity| {
                self.component_storage
                    .get(entity)
                    .and_then(|components| components.get(&type_id))
                    .and_then(|component| component.downcast_ref::<T>().map(|c| (*entity, c)))
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
        entity: EntityId,
        component_type: &TypeId,
    ) -> Result<&dyn Any, WorldError> {
        match self.component_storage.get(&entity) {
            Some(components) => match components.get(component_type) {
                Some(c) => Ok(c.as_ref()),
                None => Err(WorldError::ComponentMissing),
            },
            None => Err(WorldError::EntityNotFound),
        }
    }
    fn entity_component_get_mut(
        &mut self,
        entity: EntityId,
        component_type: &TypeId,
    ) -> Result<&mut dyn Any, WorldError> {
        match self.component_storage.get_mut(&entity) {
            Some(components) => match components.get_mut(component_type) {
                Some(c) => Ok(c.as_mut()),
                None => Err(WorldError::ComponentMissing),
            },
            None => Err(WorldError::EntityNotFound),
        }
    }
}

// marker
