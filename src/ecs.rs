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

    component_storage: HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>>,
    resource_storage: HashMap<TypeId, Box<dyn Any>>,

    next_free: u32,
    last_dead: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum WorldError {
    #[error("entity not found")]
    EntityNotHasComponent,
    #[error("entity already has component")]
    ComponentAlreadyAdded,
    #[error("the requested entity does not have the component")]
    ComponentNotExist,
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

        EntityId(free)
    }
    pub fn entity_destroy(&mut self, entity: EntityId) {
        self.last_dead.push(entity.0);
        self.entities.remove(&entity);
        for (_type_id, map) in &mut self.component_storage {
            map.remove(&entity);
        }
    }

    pub fn component_add<T: 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<(), WorldError> {
        if !self.entities.contains(&entity) {
            return Err(WorldError::EntityNotHasComponent);
        }
        let type_id = TypeId::of::<T>();
        // entry returns either a empty slot which can be assigned to, or a occupied marker
        let inner: &mut HashMap<EntityId, Box<dyn Any>> = self
            .component_storage
            .entry(type_id)
            .or_insert_with(HashMap::new);

        match inner.entry(entity) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(Box::new(component));
                Ok(())
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                Err(WorldError::ComponentAlreadyAdded)
            }
        }
    }

    pub fn component_remove<T: 'static>(&mut self, entity: EntityId) -> Result<(), WorldError> {
        self.entity_component_remove(entity, TypeId::of::<T>())
    }
    pub fn component_get<T: 'static>(&self, entity: EntityId) -> Result<&T, WorldError> {
        self.entity_component_get(entity, TypeId::of::<T>())
            .and_then(|component| {
                component
                    .downcast_ref::<T>()
                    .ok_or(WorldError::ComponentNotExist)
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
                    .ok_or(WorldError::ComponentNotExist)
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
            .get(&TypeId::of::<T>())
            .map_or(false, |x| x.contains_key(&id))
    }

    /// may return an empty vec
    pub fn query_entities<T: 'static>(&self) -> Vec<EntityId> {
        let type_id = TypeId::of::<T>();

        if let Some(inner_map) = self.component_storage.get(&type_id) {
            inner_map.keys().copied().collect()
        } else {
            Vec::new()
        }
    }
    pub fn query<T: 'static>(&self) -> Vec<(EntityId, &T)> {
        let type_id = TypeId::of::<T>();

        if let Some(inner_map) = self.component_storage.get(&type_id) {
            inner_map
                .iter()
                .map(|x| {
                    (
                        *x.0,
                        x.1.downcast_ref::<T>()
                            .expect("component type mismatch: storage corrupted"),
                    )
                })
                .collect()
        } else {
            Vec::new()
        }
    }
    // is safe because entity_id is guaranteed to be unique, then we only need to index the inner
    // hashmap once
    //
    /// returns an empty array if nothing matches
    pub fn query_mut<T: 'static>(&mut self) -> Vec<(EntityId, &mut T)> {
        let type_id = TypeId::of::<T>();

        if let Some(inner_map) = self.component_storage.get_mut(&type_id) {
            inner_map
                .iter_mut()
                .map(|x| {
                    (
                        *x.0,
                        x.1.downcast_mut::<T>()
                            .expect("component type mismatch: storage corrupted"),
                    )
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    ///Panics: panics if type a and type b are the same
    pub fn query2_mut<A: 'static, B: 'static>(&mut self) -> Vec<(EntityId, &mut A, &mut B)> {
        let type_a = TypeId::of::<A>();
        let type_b = TypeId::of::<B>();
        assert!(type_a != type_b);

        let component_storage: *mut HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>> =
            &raw mut self.component_storage;
        let storage_a;
        let storage_b;
        unsafe {
            storage_a = component_storage.as_mut().unwrap().get_mut(&type_a);
            storage_b = component_storage.as_mut().unwrap().get_mut(&type_b);
        }
        if storage_a.is_none() || storage_b.is_none() {
            return Vec::new();
        }
        let storage_a = storage_a.unwrap();
        let storage_b = storage_b.unwrap();

        let mut results = Vec::new();
        for (&entity, a) in storage_a.iter_mut() {
            if let Some(b) = unsafe {
                let ptr_b: *mut HashMap<EntityId, Box<dyn Any + 'static>> = storage_b as *mut _;
                (*ptr_b).get_mut(&entity)
            } {
                let a_mut = a.downcast_mut::<A>().expect("malformed component");
                let b_mut = b.downcast_mut::<B>().expect("malformed component");
                results.push((entity, a_mut, b_mut));
            }
        }
        results
    }

    ///Panics: panics if type a and type b are the same
    pub fn query3_mut<A: 'static, B: 'static, C: 'static>(
        &mut self,
    ) -> Vec<(EntityId, &mut A, &mut B, &mut C)> {
        let type_a = TypeId::of::<A>();
        let type_b = TypeId::of::<B>();
        let type_c = TypeId::of::<C>();
        assert!(type_a != type_b && type_a != type_c && type_b != type_c);

        let component_storage: *mut HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>> =
            &raw mut self.component_storage;
        let storage_a;
        let storage_b;
        let storage_c;
        unsafe {
            storage_a = component_storage.as_mut().unwrap().get_mut(&type_a);
            storage_b = component_storage.as_mut().unwrap().get_mut(&type_b);
            storage_c = component_storage.as_mut().unwrap().get_mut(&type_c);
        }

        if storage_a.is_none() || storage_b.is_none() || storage_c.is_none() {
            return Vec::new();
        }
        let storage_a = storage_a.unwrap();
        let storage_b = storage_b.unwrap();
        let storage_c = storage_c.unwrap();

        let mut results = Vec::new();
        for (&entity, a) in storage_a.iter_mut() {
            if let (Some(b), Some(c)) = unsafe {
                let ptr_b: *mut HashMap<EntityId, Box<dyn Any + 'static>> = storage_b as *mut _;
                let ptr_c: *mut HashMap<EntityId, Box<dyn Any + 'static>> = storage_c as *mut _;

                ((*ptr_b).get_mut(&entity), (*ptr_c).get_mut(&entity))
            } {
                let a_mut = a.downcast_mut::<A>().expect("malformed component");
                let b_mut = b.downcast_mut::<B>().expect("malformed component");
                let c_mut = c.downcast_mut::<C>().expect("malformed component");
                results.push((entity, a_mut, b_mut, c_mut));
            }
        }
        results
    }

    // impl details
    fn entity_component_remove(
        &mut self,
        entity: EntityId,
        component_type: TypeId,
    ) -> Result<(), WorldError> {
        match self.component_storage.get_mut(&component_type) {
            Some(components) => match components.remove(&entity) {
                Some(_) => Ok(()),
                None => Err(WorldError::EntityNotHasComponent),
            },
            None => Err(WorldError::ComponentNotExist),
        }
    }
    fn entity_component_get(
        &self,
        entity: EntityId,
        component_type: TypeId,
    ) -> Result<&dyn Any, WorldError> {
        match self.component_storage.get(&component_type) {
            Some(components) => match components.get(&entity) {
                Some(c) => Ok(c.as_ref()),
                None => Err(WorldError::EntityNotHasComponent),
            },
            None => Err(WorldError::ComponentNotExist),
        }
    }
    fn entity_component_get_mut(
        &mut self,
        entity: EntityId,
        component_type: &TypeId,
    ) -> Result<&mut dyn Any, WorldError> {
        match self.component_storage.get_mut(&component_type) {
            Some(components) => match components.get_mut(&entity) {
                Some(c) => Ok(c.as_mut()),
                None => Err(WorldError::EntityNotHasComponent),
            },
            None => Err(WorldError::ComponentNotExist),
        }
    }
}
