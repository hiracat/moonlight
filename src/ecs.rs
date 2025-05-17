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
    pub fn has_component<T: 'static>(&self, id: EntityId) -> bool {
        self.component_storage
            .get(&id)
            .map_or(false, |x| x.contains_key(&TypeId::of::<T>()))
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
    // is safe because entity_id is guaranteed to be unique, then we only need to index the inner
    // hashmap once
    //
    /// returns an empty array if nothing matches
    pub fn query_mut<T: 'static>(&mut self) -> Vec<(EntityId, &mut T)> {
        let type_id = TypeId::of::<T>();
        let mut result = Vec::new();
        let storage_pointer: *mut HashMap<EntityId, HashMap<TypeId, Box<dyn Any>>> =
            &raw mut self.component_storage;

        for entity in &self.entities {
            unsafe {
                let storage = &mut *storage_pointer;
                if let Some(components) = storage.get_mut(entity) {
                    if let Some(component) = components.get_mut(&type_id) {
                        if let Some(typed_component) = component.downcast_mut::<T>() {
                            result.push((*entity, typed_component));
                        }
                    }
                }
            }
        }

        result
    }

    ///Panics: panics if type a and type b are the same
    pub fn query2_mut<A: 'static, B: 'static>(&mut self) -> Vec<(EntityId, &mut A, &mut B)> {
        let type_a = TypeId::of::<A>();
        let type_b = TypeId::of::<B>();
        assert!(type_a != type_b);

        let mut result = Vec::new();
        let component_storage: *mut HashMap<EntityId, HashMap<TypeId, Box<dyn Any>>> =
            &raw mut self.component_storage;

        for entity in &self.entities {
            unsafe {
                if let Some(components) = (*component_storage).get_mut(entity) {
                    let comp_storage_pointer: *mut HashMap<TypeId, Box<dyn Any>> =
                        components as *mut _;

                    if let (Some(comp_a), Some(comp_b)) = (
                        (*comp_storage_pointer).get_mut(&type_a),
                        (*comp_storage_pointer).get_mut(&type_b),
                    ) {
                        if let (Some(a), Some(b)) =
                            (comp_a.downcast_mut::<A>(), comp_b.downcast_mut::<B>())
                        {
                            result.push((*entity, a, b));
                        }
                    }
                }
            }
        }

        result
    }

    ///Panics: panics if type a and type b are the same
    pub fn query3_mut<A: 'static, B: 'static, C: 'static>(
        &mut self,
    ) -> Vec<(EntityId, &mut A, &mut B, &mut C)> {
        let type_a = TypeId::of::<A>();
        let type_b = TypeId::of::<B>();
        let type_c = TypeId::of::<C>();
        assert!(type_a != type_b && type_a != type_c);

        let mut result = Vec::new();
        let component_storage: *mut HashMap<EntityId, HashMap<TypeId, Box<dyn Any>>> =
            &raw mut self.component_storage;

        for entity in &self.entities {
            unsafe {
                if let Some(components) = (*component_storage).get_mut(entity) {
                    let comp_storage_pointer: *mut HashMap<TypeId, Box<dyn Any>> =
                        components as *mut _;

                    if let (Some(comp_a), Some(comp_b), Some(comp_c)) = (
                        (*comp_storage_pointer).get_mut(&type_a),
                        (*comp_storage_pointer).get_mut(&type_b),
                        (*comp_storage_pointer).get_mut(&type_c),
                    ) {
                        if let (Some(a), Some(b), Some(c)) = (
                            comp_a.downcast_mut::<A>(),
                            comp_b.downcast_mut::<B>(),
                            comp_c.downcast_mut::<C>(),
                        ) {
                            result.push((*entity, a, b, c));
                        }
                    }
                }
            }
        }

        result
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
