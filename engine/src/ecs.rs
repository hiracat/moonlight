#![allow(dead_code)]
use std::{
    any::{Any, TypeId},
    collections::{hash_map::HashMap, HashSet},
    time::Instant,
};

macro_rules! assert_unique_types {
    ($($ty:ident)+) => {
        {
        let type_ids = [$(TypeId::of::<$ty>()),+];
        let seen = HashSet::from(type_ids);
        assert_eq!(type_ids.len(), seen.len());
        }
    };
}
macro_rules! get_first {
    ($first:ident $($rest:ident)*) => {
        $first
    };
}
macro_rules! impl_join{
    ($struct:ident, $iter:ident, $(($name:ident, $peek:ident: $ty:ident)),+ [opt] $(($opt_name:ident, $peek_opt:ident: $opt_ty:ident)),*) => {
        struct $struct<$($ty),+, $($opt_ty),*>{
            $(
                $name: $ty,
            )+
            $(
                $opt_name: $opt_ty
            ),*
        }
        impl <$($ty),+, $($opt_ty),*> Joinable for $struct<$($ty),+, $($opt_ty),*>
        where
            $(
                $ty: Joinable,
            )+
            $(
                $opt_ty: Joinable,
            )*
            {
                type Ref<'a> = ($($ty::Ref<'a>,)+ $(Option<$opt_ty::Ref<'a>>,)*) where Self: 'a;
                type Component<'a> = ($($ty::Component<'a>,)+$($opt_ty::Component<'a>,)*)
                where
                    $($ty: 'a,)+
                    $($opt_ty: 'a,)*;
                type Iter<'a> = $iter<'a, $($ty,)+ $($opt_ty),*>
                where
                    $($ty: 'a,)+
                    $($opt_ty: 'a,)*;
                fn join<'a>(self) -> Self::Iter<'a> {
                    $iter::new($(self.$name.join(),)+ $(self.$opt_name.join()),*)
                }
            }

        struct $iter<'a, $($ty),+, $($opt_ty),*>
        where
        $(
        $ty: Joinable+ 'a,
        )+
        $(
        $opt_ty: Joinable+ 'a,
        )*
        {
            $(
                $name: <$ty as Joinable>::Iter<'a>,
                $peek: Option<(EntityId, <$ty as Joinable>::Ref<'a>)>,
            )+
            $(
                $opt_name: <$opt_ty as Joinable>::Iter<'a>,
                $peek_opt: Option<(EntityId, <$opt_ty as Joinable>::Ref<'a>)>,
            )*
        }
        impl<'a, $($ty),+, $($opt_ty),*> $iter<'a, $($ty),+, $($opt_ty),*>
        where
            $(
                $ty: Joinable,
            )+
            $(
                $opt_ty: Joinable,
            )*
            {
                fn new(
                $(
                   mut $name: $ty::Iter<'a>,
                )+
                $(
                   mut $opt_name: $opt_ty::Iter<'a>,
                )*
                )
                    -> Self{
                $(
                    let $peek = $name.next();
                )+
                $(
                    let $peek_opt = $opt_name.next();
                )*
                Self {
                $(
                   $peek,
                   $name,
                )+
                $(
                   $peek_opt,
                   $opt_name,
                )*
                }
                }

            }
        impl <'a, $($ty),+, $($opt_ty),*> Iterator for $iter<'a, $($ty),+, $($opt_ty),*>
        where
        $(
        $ty: Joinable,
        )+
        $(
        $opt_ty: Joinable,
        )*
        {
            type Item = (EntityId,
                (
                $(
                    $ty::Ref<'a>,
                )+
                $(
                    Option<$opt_ty::Ref<'a>>,
                )*
                ),
            );
            fn next(&mut self) -> Option<Self::Item> {
                $(
                let (mut $name, mut $peek) = Option::take(&mut self.$peek)?;
                self.$peek = self.$name.next();
                )+
                let mut max_rq_id;
                // bring anything required up to the current required index early return none if any
                // run out
                loop {
                    max_rq_id = get_first!($($name)+)$(.max($name))+;
                    let mut all_max = true;
                    $(
                    if $name < max_rq_id {
                        ($name, $peek) = Option::take(&mut self.$peek)?;
                        self.$peek = self.$name.next();
                        all_max = false;
                    }
                    )+
                    if all_max {
                        break;
                    }
                }
                // bring all optionals to at least the current place
                $(
                while let Some(($opt_name, _)) = self.$peek_opt{
                    if $opt_name < max_rq_id {
                        self.$peek_opt = self.$opt_name.next()
                    } else {
                        break;
                    }
                }
                // take a peek
                let $peek_opt = if let Some(($opt_name, $peek_opt)) = Option::take(&mut self.$peek_opt) {
                    // if peek is the one we want
                    if max_rq_id == $opt_name {
                        // use it  and advance
                        let tmp = Some($peek_opt);
                        self.$peek_opt = self.$opt_name.next();
                        tmp
                    } else {
                        // else put it back for next time because it might be useful then
                        self.$peek_opt = Some(($opt_name, $peek_opt));
                        None
                    }
                } else {
                    None
                };
                )*
                return Some((max_rq_id, ($($peek,)+ $($peek_opt),*)));
            }
        }

    };
}
macro_rules! impl_join_mut {
    ($struct:ident, $iter:ident, $(($name:ident, $peek:ident: $ty:ident)),+ [opt] $(($opt_name:ident, $peek_opt:ident: $opt_ty:ident)),*) => {
        struct $struct<$($ty),+, $($opt_ty),*>{
            $(
                $name: $ty,
            )+
            $(
                $opt_name: $opt_ty
            ),*
        }
        impl <$($ty),+, $($opt_ty),*> JoinableMut for $struct<$($ty),+, $($opt_ty),*>
        where
            $(
                $ty: JoinableMut,
            )+
            $(
                $opt_ty: JoinableMut,
            )*
            {
                type Mut<'a> = ($($ty::Mut<'a>,)+ $(Option<$opt_ty::Mut<'a>>,)*) where Self: 'a;
                type Component<'a> = ($($ty::Component<'a>,)+$($opt_ty::Component<'a>,)*)
                where
                    $($ty: 'a,)+
                    $($opt_ty: 'a,)*;
                type Iter<'a> = $iter<'a, $($ty,)+ $($opt_ty),*>
                where
                    $($ty: 'a,)+
                    $($opt_ty: 'a,)*;
                unsafe fn join<'a>(self) -> Self::Iter<'a> {
                    unsafe {$iter::new($(self.$name.join(),)+ $(self.$opt_name.join()),*)}
                }
            }

        struct $iter<'a, $($ty),+, $($opt_ty),*>
        where
        $(
        $ty: JoinableMut + 'a,
        )+
        $(
        $opt_ty: JoinableMut + 'a,
        )*
        {
            $(
                $name: <$ty as JoinableMut>::Iter<'a>,
                $peek: Option<(EntityId, <$ty as JoinableMut>::Mut<'a>)>,
            )+
            $(
                $opt_name: <$opt_ty as JoinableMut>::Iter<'a>,
                $peek_opt: Option<(EntityId, <$opt_ty as JoinableMut>::Mut<'a>)>,
            )*
        }
        impl<'a, $($ty),+, $($opt_ty),*> $iter<'a, $($ty),+, $($opt_ty),*>
        where
            $(
                $ty: JoinableMut,
            )+
            $(
                $opt_ty: JoinableMut,
            )*
            {
                unsafe fn new(
                $(
                   mut $name: $ty::Iter<'a>,
                )+
                $(
                   mut $opt_name: $opt_ty::Iter<'a>,
                )*
                )
                    -> Self{
                $(
                    let $peek = $name.next();
                )+
                $(
                    let $peek_opt = $opt_name.next();
                )*
                Self {
                $(
                   $peek,
                   $name,
                )+
                $(
                   $peek_opt,
                   $opt_name,
                )*
                }
                }

            }
        impl <'a, $($ty),+, $($opt_ty),*> Iterator for $iter<'a, $($ty),+, $($opt_ty),*>
        where
        $(
        $ty: JoinableMut,
        )+
        $(
        $opt_ty: JoinableMut,
        )*
        {
            type Item = (EntityId,
                (
                $(
                    $ty::Mut<'a>,
                )+
                $(
                    Option<$opt_ty::Mut<'a>>,
                )*
                ),
            );
            fn next(&mut self) -> Option<Self::Item> {
                $(
                let (mut $name, mut $peek) = Option::take(&mut self.$peek)?;
                self.$peek = self.$name.next();
                )+
                let mut max_rq_id;
                // bring anything required up to the current required index early return none if any
                // run out
                loop {
                    max_rq_id = get_first!($($name)+)$(.max($name))+;
                    let mut all_max = true;
                    $(
                    if $name < max_rq_id {
                        ($name, $peek) = Option::take(&mut self.$peek)?;
                        self.$peek = self.$name.next();
                        all_max = false;
                    }
                    )+
                    if all_max {
                        break;
                    }
                }
                // bring all optionals to at least the current place
                $(
                while let Some(($opt_name, _)) = self.$peek_opt{
                    if $opt_name < max_rq_id {
                        self.$peek_opt = self.$opt_name.next()
                    } else {
                        break;
                    }
                }
                // take a peek
                let $peek_opt = if let Some(($opt_name, $peek_opt)) = Option::take(&mut self.$peek_opt) {
                    // if peek is the one we want
                    if max_rq_id == $opt_name {
                        // use it  and advance
                        let tmp = Some($peek_opt);
                        self.$peek_opt = self.$opt_name.next();
                        tmp
                    } else {
                        // else put it back for next time because it might be useful then
                        self.$peek_opt = Some(($opt_name, $peek_opt));
                        None
                    }
                } else {
                    None
                };
                )*
                return Some((max_rq_id, ($($peek,)+ $($peek_opt),*)));
            }
        }
    }
}
macro_rules! impl_fetch{
    ($iter:ident, $struct:ident, $(($name:ident, $ty:ident)),+ [opt] $(($optname:ident, $opt:ident)),* ) => {
impl< $($ty: 'static,)+ $($opt: 'static),*> Fetch for ($(Req<$ty>,)+ $(Opt<$opt>),*) {
    type Iter<'w> = $iter<'w, $(&'w [(EntityId, $ty)],)+ $(&'w [(EntityId, $opt)]),*>;
    type Item<'w> = ($(&'w $ty,)+ $(Option<&'w $opt>),*);

    type OptionalItems<'w> = Result<($(&'w $ty,)+ $(Option<&'w $opt>),*), WorldError>;

    fn get<'w>(entity: EntityId, world: &'w World) -> Self::OptionalItems<'w> {
        let ptr_world: &World = world;
        assert_unique_types!($($ty)+ $($opt)*);
        $(
        let $name = {
            (*ptr_world)
                .get_storage::<$ty>()
                .ok_or(WorldError::ComponentTypeMissing)?
                .get(entity)
                .ok_or(WorldError::EntityMissingComponent)?
        };
        )+
        $(
        let $optname: Option<&$opt> = {
            (*ptr_world)
                .get_storage::<$opt>()
                .and_then(|x| x.get(entity))
        };
        )*
        Ok(($($name,)+ $($optname),*))
    }
    fn fetch<'w>(world: &'w World) -> Self::Iter<'w> {
        assert_unique_types!($($ty)+ $($opt)*);
        let ptr_world: &World = world;
        $(
        let $name: &[(EntityId, $ty)] = {
            ptr_world
                .get_storage::<$ty>()
                .map_or(&[], |x| x.data.as_slice())
        };
        )+
        $(
        let $optname: &[(EntityId, $opt)] = {
            ptr_world
                .get_storage::<$opt>()
                .map_or(&[], |x| x.data.as_slice())
        };
        )*
        let join = $struct {
            $(
            $name,
            )+
            $(
            $optname,
            )*
        };
        //Safety: Panics of any types are the same
        return  join.join() ;
    }
}
};
}
macro_rules! impl_fetch_mut{
    ($iter:ident, $struct:ident, $(($name:ident, $ty:ident)),+ [opt] $(($optname:ident, $opt:ident)),* ) => {
impl< $($ty: 'static,)+ $($opt: 'static),*> FetchMut for ($(ReqM<$ty>,)+ $(OptM<$opt>),*) {
    type Iter<'w> = $iter<'w, $(&'w mut [(EntityId, $ty)],)+ $(&'w mut [(EntityId, $opt)]),*>;
    type Item<'w> = ($(&'w mut $ty,)+ $(Option<&'w mut $opt>),*);

    type OptionalItems<'w> = Result<($(&'w mut $ty,)+ $(Option<&'w mut $opt>),*), WorldError>;

    fn get_mut<'w>(entity: EntityId, world: &'w mut World) -> Self::OptionalItems<'w> {
        let ptr_world: *mut World = world;
        assert_unique_types!($($ty)+ $($opt)*);
        $(
        let $name = unsafe {
            (*ptr_world)
                .get_storage_mut::<$ty>()
                .ok_or(WorldError::ComponentTypeMissing)?
                .get_mut(entity)
                .ok_or(WorldError::EntityMissingComponent)?
        };
        )+
        $(
        let $optname: Option<&mut $opt> = unsafe {
            (*ptr_world)
                .get_storage_mut::<$opt>()
                .and_then(|x| x.get_mut(entity))
        };
        )*
        Ok(($($name,)+ $($optname),*))
    }
    fn fetch<'w>(world: &'w mut World) -> Self::Iter<'w> {
        assert_unique_types!($($ty)+ $($opt)*);
        let ptr_world: *mut World = world;
        $(
        let $name: &mut [(EntityId, $ty)] = unsafe {
            (*ptr_world)
                .get_storage_mut::<$ty>()
                .map_or(&mut [], |x| x.data.as_mut_slice())
        };
        )+
        $(
        let $optname: &mut [(EntityId, $opt)] = unsafe {
            (*ptr_world)
                .get_storage_mut::<$opt>()
                .map_or(&mut [], |x| x.data.as_mut_slice())
        };
        )*
        let join = $struct {
            $(
            $name,
            )+
            $(
            $optname,
            )*
        };
        //Safety: Panics of any types are the same
        return unsafe { join.join() };
    }
}
};
}

// N = 2
impl_fetch_mut!(JoinMut2Opt0Iter, JoinMut2Opt0, (a, T), (b, U)[opt]);
impl_fetch_mut!(JoinMut2Opt1Iter, JoinMut2Opt1, (a, T)[opt](b, U));

// N = 3
impl_fetch_mut!(JoinMut3Opt0Iter, JoinMut3Opt0, (a, T), (b, U), (c, V)[opt]);
impl_fetch_mut!(JoinMut3Opt1Iter, JoinMut3Opt1, (a, T), (b, U)[opt](c, V));
impl_fetch_mut!(JoinMut3Opt2Iter, JoinMut3Opt2, (a, T)[opt](b, U), (c, V));

// N = 4
impl_fetch_mut!(
    JoinMut4Opt0Iter,
    JoinMut4Opt0,
    (a, T),
    (b, U),
    (c, V),
    (d, W)[opt]
);
impl_fetch_mut!(
    JoinMut4Opt1Iter,
    JoinMut4Opt1,
    (a, T),
    (b, U),
    (c, V)[opt](d, W)
);
impl_fetch_mut!(
    JoinMut4Opt2Iter,
    JoinMut4Opt2,
    (a, T),
    (b, U)[opt](c, V),
    (d, W)
);
impl_fetch_mut!(
    JoinMut4Opt3Iter,
    JoinMut4Opt3,
    (a, T)[opt](b, U),
    (c, V),
    (d, W)
);
// N = 2
impl_join_mut!(JoinMut2Opt0, JoinMut2Opt0Iter, (a, peek_a: T), (b, peek_b: U) [opt]);
impl_join_mut!(JoinMut2Opt1, JoinMut2Opt1Iter, (a, peek_a: T) [opt] (b, peek_b: U));

// N = 3
impl_join_mut!(JoinMut3Opt0, JoinMut3Opt0Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V) [opt]);
impl_join_mut!(JoinMut3Opt1, JoinMut3Opt1Iter, (a, peek_a: T), (b, peek_b: U) [opt] (c, peek_c: V));
impl_join_mut!(JoinMut3Opt2, JoinMut3Opt2Iter, (a, peek_a: T) [opt] (b, peek_b: U), (c, peek_c: V));

// N = 4
impl_join_mut!(JoinMut4Opt0, JoinMut4Opt0Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V), (d, peek_d: W) [opt]);
impl_join_mut!(JoinMut4Opt1, JoinMut4Opt1Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V) [opt] (d, peek_d: W) );
impl_join_mut!(JoinMut4Opt2, JoinMut4Opt2Iter, (a, peek_a: T), (b, peek_b: U) [opt] (c, peek_c: V), (d, peek_d: W));
impl_join_mut!(JoinMut4Opt3, JoinMut4Opt3Iter, (a, peek_a: T) [opt] (b, peek_b: U), (c, peek_c: V), (d, peek_d: W));

// N = 2
impl_fetch!(Join2Opt0Iter, Join2Opt0, (a, T), (b, U)[opt]);
impl_fetch!(Join2Opt1Iter, Join2Opt1, (a, T)[opt](b, U));

// N = 3
impl_fetch!(Join3Opt0Iter, Join3Opt0, (a, T), (b, U), (c, V)[opt]);
impl_fetch!(Join3Opt1Iter, Join3Opt1, (a, T), (b, U)[opt](c, V));
impl_fetch!(Join3Opt2Iter, Join3Opt2, (a, T)[opt](b, U), (c, V));

// N = 4
impl_fetch!(
    Join4Opt0Iter,
    Join4Opt0,
    (a, T),
    (b, U),
    (c, V),
    (d, W)[opt]
);
impl_fetch!(Join4Opt1Iter, Join4Opt1, (a, T), (b, U), (c, V)[opt](d, W));
impl_fetch!(Join4Opt2Iter, Join4Opt2, (a, T), (b, U)[opt](c, V), (d, W));
impl_fetch!(Join4Opt3Iter, Join4Opt3, (a, T)[opt](b, U), (c, V), (d, W));

// N = 2
impl_join!(Join2Opt0, Join2Opt0Iter, (a, peek_a: T), (b, peek_b: U) [opt]);
impl_join!(Join2Opt1, Join2Opt1Iter, (a, peek_a: T) [opt] (b, peek_b: U));

// N = 3
impl_join!(Join3Opt0, Join3Opt0Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V) [opt]);
impl_join!(Join3Opt1, Join3Opt1Iter, (a, peek_a: T), (b, peek_b: U) [opt] (c, peek_c: V) );
impl_join!(Join3Opt2, Join3Opt2Iter, (a, peek_a: T) [opt] (b, peek_b: U), (c, peek_c: V));

// N = 4
impl_join!(Join4Opt0, Join4Opt0Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V), (d, peek_d: W) [opt]);
impl_join!(Join4Opt1, Join4Opt1Iter, (a, peek_a: T), (b, peek_b: U), (c, peek_c: V) [opt] (d, peek_d: W));
impl_join!(Join4Opt2, Join4Opt2Iter, (a, peek_a: T), (b, peek_b: U) [opt] (c, peek_c: V), (d, peek_d: W));
impl_join!(Join4Opt3, Join4Opt3Iter, (a, peek_a: T) [opt] (b, peek_b: U), (c, peek_c: V), (d, peek_d: W));

#[derive(Debug, Ord, PartialOrd, Eq, Hash, PartialEq, Clone, Copy)]
pub struct EntityId(u32);

#[derive(Debug, thiserror::Error)]
pub enum WorldError {
    #[error("entity already has component")]
    ComponentAlreadyAdded,
    #[error("entity does not have the requested component")]
    EntityMissingComponent,
    #[error("the requested entity does not exist")]
    EntityMissing,
    #[error("the requested component does not exist")]
    ComponentTypeMissing,
    #[error("a resource of this type has already been added")]
    ResourceAlreadyAdded,
}

pub struct World {
    entities: HashSet<EntityId>,

    component_storages: HashMap<TypeId, Box<dyn ErasedComponentStorage>>,
    resource_storage: HashMap<TypeId, Box<dyn Any>>,

    next_free: u32,
    last_dead: Vec<u32>,
}

struct Query<'a, C: Fetch> {
    iter: C::Iter<'a>,
}
impl<'w, C> Iterator for Query<'w, C>
where
    C: Fetch + 'static,
    C::Iter<'w>: Iterator<Item = (EntityId, C::Item<'w>)>,
{
    type Item = (EntityId, C::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

struct QueryMut<'a, C: FetchMut> {
    iter: C::Iter<'a>,
}
impl<'w, C> Iterator for QueryMut<'w, C>
where
    C: FetchMut + 'static,
    C::Iter<'w>: Iterator<Item = (EntityId, C::Item<'w>)>,
{
    type Item = (EntityId, C::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

trait Fetch {
    // reference to component storage iterator
    type Iter<'w>: Iterator;
    type Item<'w>;
    type OptionalItems<'w>;
    fn fetch<'w>(world: &'w World) -> Self::Iter<'w>;
    fn get<'w>(entity: EntityId, world: &'w World) -> Self::OptionalItems<'w>;
}

trait FetchMut {
    // reference to component storage iterator
    type Iter<'w>: Iterator;
    type Item<'w>;
    type OptionalItems<'w>;
    fn fetch<'w>(world: &'w mut World) -> Self::Iter<'w>;
    fn get_mut<'w>(entity: EntityId, world: &'w mut World) -> Self::OptionalItems<'w>;
}
trait Joinable {
    type Ref<'a>: 'a + Copy
    where
        Self: 'a;

    type Component<'a>: 'a
    where
        Self: 'a;

    /// The iterator over `(EntityId, Ref<'a>)`
    type Iter<'a>: Iterator<Item = (EntityId, Self::Ref<'a>)>
    where
        Self: 'a;

    fn join<'a>(self) -> Self::Iter<'a>;
}

trait JoinableMut {
    type Mut<'a>: 'a
    where
        Self: 'a;

    type Component<'a>: 'a
    where
        Self: 'a;

    /// The iterator over `(EntityId, Ref<'a>)`
    type Iter<'a>: Iterator<Item = (EntityId, Self::Mut<'a>)>
    where
        Self: 'a;

    // SAFETY: must ensure that all the iterators point to distinct places
    unsafe fn join<'a>(self) -> Self::Iter<'a>;
}

impl<T: 'static> Fetch for (T,) {
    type Item<'w> = (&'w T,);
    type Iter<'w> = std::iter::Map<
        std::slice::Iter<'w, (EntityId, T)>,
        fn(&'w (EntityId, T)) -> (EntityId, (&'w T,)),
    >;
    type OptionalItems<'w> = Result<&'w T, WorldError>;

    fn fetch<'w>(world: &'w World) -> Self::Iter<'w> {
        fn map_fn<T>(x: &(EntityId, T)) -> (EntityId, (&T,)) {
            (x.0, (&x.1,))
        }
        match world.get_storage() {
            Some(x) => x.data.iter().map(map_fn),
            None => [].iter().map(map_fn),
        }
    }
    fn get<'w>(entity: EntityId, world: &'w World) -> Self::OptionalItems<'w> {
        let storage = match world.get_storage().ok_or(WorldError::ComponentTypeMissing) {
            Ok(it) => it,
            Err(err) => return Err(err),
        };
        storage
            .get(entity)
            .ok_or(WorldError::EntityMissingComponent)
    }
}
impl<T: 'static> FetchMut for (T,) {
    type Item<'w> = (&'w mut T,);
    type Iter<'w> = std::iter::Map<
        std::slice::IterMut<'w, (EntityId, T)>,
        fn(&'w mut (EntityId, T)) -> (EntityId, (&'w mut T,)),
    >;
    type OptionalItems<'w> = Result<&'w mut T, WorldError>;

    fn get_mut<'w>(entity: EntityId, world: &'w mut World) -> Self::OptionalItems<'w> {
        let storage = match world
            .get_storage_mut()
            .ok_or(WorldError::ComponentTypeMissing)
        {
            Ok(it) => it,
            Err(err) => return Err(err),
        };
        storage
            .get_mut(entity)
            .ok_or(WorldError::EntityMissingComponent)
    }

    fn fetch<'w>(world: &'w mut World) -> Self::Iter<'w> {
        fn map_fn<T>(x: &mut (EntityId, T)) -> (EntityId, (&mut T,)) {
            (x.0, (&mut x.1,))
        }
        match world.get_storage_mut() {
            Some(x) => x.data.iter_mut().map(map_fn),
            None => [].iter_mut().map(map_fn),
        }
    }
}

impl<'s, T: 'static> Joinable for &'s [(EntityId, T)] {
    type Component<'a>
        = T
    where
        's: 'a;
    type Iter<'a>
        = std::iter::Map<
        std::slice::Iter<'a, (EntityId, T)>,
        fn(&'a (EntityId, T)) -> (EntityId, &'a T),
    >
    where
        's: 'a;

    type Ref<'a>
        = &'a T
    where
        's: 'a;

    fn join<'a>(self) -> Self::Iter<'a>
    where
        's: 'a,
    {
        fn map_fn<T>(x: &(EntityId, T)) -> (EntityId, &T) {
            (x.0, &x.1)
        }
        self.iter().map(map_fn::<T>)
    }
}
impl<'s, T: 'static> JoinableMut for &'s mut [(EntityId, T)] {
    type Component<'a>
        = T
    where
        's: 'a;
    type Iter<'a>
        = std::iter::Map<
        std::slice::IterMut<'a, (EntityId, T)>,
        fn(&'a mut (EntityId, T)) -> (EntityId, &'a mut T),
    >
    where
        's: 'a;

    type Mut<'a>
        = &'a mut T
    where
        's: 'a;

    ///SAFETY: part of the trait, but this implimentation is guaranteed to be safe, it simply maps
    ///the shape from an awkward one to the one that the recursive trait expects
    unsafe fn join<'a>(self) -> Self::Iter<'a>
    where
        's: 'a,
    {
        fn map_fn<T>(x: &mut (EntityId, T)) -> (EntityId, &mut T) {
            (x.0, &mut x.1)
        }
        self.iter_mut().map(map_fn::<T>)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Opt<C>(pub C);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Req<C>(pub C);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct OptM<C>(pub C);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReqM<C>(pub C);

#[derive(Debug)]
pub struct ComponentStorage<T: 'static> {
    data: Vec<(EntityId, T)>,
}
impl<T: 'static> ComponentStorage<T> {
    pub fn new() -> ComponentStorage<T> {
        ComponentStorage::<T> { data: Vec::new() }
    }
    pub fn add(&mut self, entity: EntityId, value: T) -> Result<(), WorldError> {
        match self.data.binary_search_by_key(&entity, |x| x.0) {
            Ok(_) => return Err(WorldError::ComponentAlreadyAdded),
            Err(x) => {
                self.data.insert(x, (entity, value));
                return Ok(());
            }
        }
    }
    pub fn replace(&mut self, entity: EntityId, value: T) {
        match self.data.binary_search_by_key(&entity, |x| x.0) {
            Ok(x) => {
                self.data[x] = (entity, value);
            }
            Err(x) => {
                self.data.insert(x, (entity, value));
            }
        }
    }
    pub fn remove(&mut self, entity: EntityId) -> Result<(), WorldError> {
        match self.data.binary_search_by_key(&entity, |x| x.0) {
            Ok(x) => {
                self.data.remove(x);
                if self.data.len() < 1 {
                    self.data.shrink_to_fit();
                }
                return Ok(());
            }
            Err(_) => return Err(WorldError::EntityMissingComponent),
        }
    }
    pub fn get(&self, entity: EntityId) -> Option<&T> {
        match self.data.binary_search_by_key(&entity, |x| x.0) {
            Ok(x) => return self.data.get(x).map(|x| &x.1),
            Err(_) => return None,
        }
    }
    pub fn get_mut(&mut self, entity: EntityId) -> Option<&mut T> {
        match self.data.binary_search_by_key(&entity, |x| x.0) {
            Ok(x) => return self.data.get_mut(x).map(|x| &mut x.1),
            Err(_) => return None,
        }
    }
}

trait ErasedComponentStorage: Any {
    fn remove_entity(&mut self, entity: EntityId);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> ErasedComponentStorage for ComponentStorage<T> {
    /// returns true if remove was successful
    fn remove_entity(&mut self, entity: EntityId) {
        self.remove(entity);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl World {
    pub fn init() -> Self {
        Self {
            entities: HashSet::new(),
            component_storages: HashMap::new(),
            resource_storage: HashMap::new(),

            next_free: 0,
            last_dead: Vec::new(),
        }
    }
    pub fn spawn(&mut self) -> EntityId {
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
    pub fn destroy(&mut self, entity: EntityId) -> Result<(), WorldError> {
        //HACK: this needs to be replaced with a proper id allocator later, maybe a list of
        //preallocated ids that grows when reaches max capacity, and shrinks when < 50% is full

        match self.entities.remove(&entity) {
            true => {
                if self.last_dead.len() < 1_000 {
                    self.last_dead.push(entity.0);
                }
                for storage in &mut self.component_storages {
                    storage.1.remove_entity(entity);
                }
                Ok(())
            }
            false => Err(WorldError::EntityMissing),
        }
    }
    pub fn add<T: 'static>(&mut self, entity: EntityId, component: T) -> Result<(), WorldError> {
        if self.entities.contains(&entity) {
            self.get_or_make_storage::<T>().add(entity, component)
        } else {
            Err(WorldError::EntityMissing)
        }
    }
    pub fn replace<T: 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<(), WorldError> {
        if self.entities.contains(&entity) {
            Ok(self.get_or_make_storage::<T>().replace(entity, component))
        } else {
            Err(WorldError::EntityMissing)
        }
    }
    pub fn remove<T: 'static>(&mut self, entity: EntityId) -> Result<(), WorldError> {
        self.get_storage_mut::<T>()
            .map_or(Err(WorldError::EntityMissingComponent), |x| {
                x.remove(entity)
            })
    }

    pub fn get<T: 'static + Fetch>(&self, entity: EntityId) -> T::OptionalItems<'_> {
        T::get(entity, self)
    }
    pub fn get_mut<T: 'static + FetchMut>(&mut self, entity: EntityId) -> T::OptionalItems<'_> {
        T::get_mut(entity, self)
    }
    pub fn add_resource<T: 'static>(&mut self, resource: T) -> Result<(), WorldError> {
        let resource: Box<dyn Any> = Box::new(resource);
        match self.resource_storage.entry(TypeId::of::<T>()) {
            std::collections::hash_map::Entry::Vacant(x) => {
                x.insert(resource);
                Ok(())
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(WorldError::ResourceAlreadyAdded)
            }
        }
    }
    pub fn remove_resource<T: 'static>(&mut self) -> Option<T> {
        match self.resource_storage.entry(TypeId::of::<T>()) {
            std::collections::hash_map::Entry::Vacant(_) => None,
            std::collections::hash_map::Entry::Occupied(x) => Some(
                *x.remove()
                    .downcast::<T>()
                    .expect("storage should only contain matching typeid"),
            ),
        }
    }
    pub fn get_resource<T: 'static>(&mut self) -> Option<&T> {
        Some(
            self.resource_storage
                .get(&TypeId::of::<T>())?
                .downcast_ref::<T>()
                .expect("incorrect type in storage"),
        )
    }
    pub fn get_mut_resource<T: 'static>(&mut self) -> Option<&mut T> {
        Some(
            self.resource_storage
                .get_mut(&TypeId::of::<T>())?
                .downcast_mut::<T>()
                .expect("incorrect type in storage"),
        )
    }

    /// Panics: panics if any of the queried types are identical
    /// Usage: takes a tuple of any types and returns an iterator over each type, however, the
    /// tuple can have at most 2 elements, more than two are needed, they can be nested
    /// (T, (U, V)),
    pub fn query<'w, C>(&'w self) -> Query<'w, C>
    where
        C: Fetch,
    {
        let iterator: C::Iter<'w> = C::fetch(self);

        Query { iter: iterator }
    }

    // Panics: panics if any of the queried types are identical
    pub fn query_mut<'w, C>(&'w mut self) -> QueryMut<'w, C>
    where
        C: FetchMut,
    {
        let iterator: C::Iter<'w> = C::fetch(self);

        QueryMut { iter: iterator }
    }

    fn get_or_make_storage<'s, T: 'static>(&'s mut self) -> &'s mut ComponentStorage<T> {
        let entry = self
            .component_storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                Box::new(ComponentStorage::<T>::new()) as Box<dyn ErasedComponentStorage>
            });

        entry
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .expect("ComponentStorage<T> was just inserted, so downcast must work")
    }

    fn get_storage<'s, T: 'static>(&'s self) -> Option<&'s ComponentStorage<T>> {
        self.component_storages.get(&TypeId::of::<T>()).map(|x| {
            x.as_any()
                .downcast_ref::<ComponentStorage<T>>()
                .expect("should be able to downcast if it exists")
        })
    }
    fn get_storage_mut<'s, T: 'static>(&'s mut self) -> Option<&'s mut ComponentStorage<T>> {
        self.component_storages
            .get_mut(&TypeId::of::<T>())
            .map(|x| {
                x.as_any_mut()
                    .downcast_mut::<ComponentStorage<T>>()
                    .expect("should be able to downcast if it exists")
            })
    }
}

/// Run `f()` once to warm up, then `iters` times measuring total elapsed.
fn bench<F>(label: &str, iters: usize, mut f: F)
where
    F: FnMut(),
{
    // warm-up
    f();

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed / (iters as u32);
    println!(
        "{:<30} {:>10?}  ({:>8.3} µs/op)",
        label,
        per,
        per.as_secs_f64() * 1e6
    );
}

fn test() {
    correctness();
    benchmark();
}
fn benchmark() {
    // You can tweak these sizes
    let scales = [1_000, 10_000, 1_000_000, 10_000_000];

    for &n in &scales {
        println!("\n=== Benchmark scale: {} entities ===", n);

        // 1) Spawn + add 3 components
        bench(&format!("spawn+add (3 comps) x{}", n), 1, || {
            let mut world = World::init();
            let mut counter = 0u64;
            for _ in 0..n {
                let e = world.spawn();
                world.add(e, (3.14f64, 42u32)).unwrap(); // comp A
                world.add(e, [1.0f64; 16]).unwrap(); // comp B
                world.add(e, "hello").unwrap(); // comp C
                counter += e.0 as u64; // Force the compiler to actually process
                std::hint::black_box(counter); // Prevent optimization
            }
        });

        // Prepare a world with all entities pre-populated
        let mut world = World::init();
        for _ in 0..n {
            let e = world.spawn();
            world.add(e, (3.14f64, 1u32)).unwrap();
            world.add(e, [1.0f64; 16]).unwrap();
            world.add(e, "hello").unwrap();
            // half of them also get an extra optional
            if e.0 % 2 == 0 {
                world.add(e, 123i64).unwrap();
            }
        }
        let mut counter = 0.0;

        // 2) Read-only query of 3 required comps
        bench(
            &format!("query   (Req<A>,Req<B>,Req<C>) x{}", n),
            10,
            || {
                let query = world.query::<(Req<(f64, u32)>, Req<[f64; 16]>, Req<&str>)>();

                for item in query {
                    counter += item.1 .1[0];
                    std::hint::black_box(counter);
                }
            },
        );
        dbg!(counter);

        let mut counter = 0;
        // 3) Read-only query with one optional
        bench(
            &format!("query   (Req<A>,Req<B>,Opt<D>) x{}", n),
            10,
            || {
                let query = world.query::<(Req<(f64, u32)>, Req<[f64; 16]>, Opt<i64>)>();

                for item in query {
                    counter += item.1 .2.unwrap_or(&0);
                    std::hint::black_box(counter);
                }
            },
        );

        let mut counter = 0;
        // 4) Mutable query of 3 required comps
        bench(
            &format!("query_mut (Req<A>,Req<B>,Req<C>) x{}", n),
            10,
            || {
                for (_, (ab, arr, s)) in
                    world.query_mut::<(ReqM<(f64, u32)>, ReqM<[f64; 16]>, ReqM<&str>)>()
                {
                    arr[0] *= 2.00;
                    let _ = ab.1;
                    counter += s.len();
                    std::hint::black_box(counter);
                }
            },
        );
        dbg!(counter);

        // 5) Mutable query with one optional
        let mut counter = 0.0;
        bench(
            &format!("query_mut (Req<A>,Req<B>,Opt<D>) x{}", n),
            10,
            || {
                for (_, (_, arr, opt_i)) in
                    world.query_mut::<(ReqM<(f64, u32)>, ReqM<[f64; 16]>, OptM<i64>)>()
                {
                    if let Some(i) = opt_i {
                        *i += 100;
                        counter += *i as f64;
                    }
                    arr[1] *= 0.4;
                    counter += arr[2];
                    std::hint::black_box(counter);
                }
            },
        );
        dbg!(counter);

        // // 6) Remove a component from half the entities
        // bench(&format!("remove D from half x{}", n), 1, || {
        //     for e in 0..n {
        //         if e % 2 == 0 {
        //             let _ = world.remove::<i64>(EntityId(e as u32));
        //         }
        //     }
        // });
        //
        // // 7) Destroy all entities in reverse
        // bench(&format!("destroy all rev x{}", n), 1, || {
        //     for e in (0..n).rev() {
        //         let _ = world.destroy(EntityId(e as u32));
        //     }
        // });
    }
}

// Test what happens with non-consecutive entity IDs
fn correctness() {
    use std::collections::HashSet;

    // Test 1: Empty world operations
    {
        let mut world = World::init();

        // Querying empty world should return no results
        let count = world.query::<(String,)>().count();
        assert_eq!(count, 0, "Empty world should have no query results");

        // Mutation on empty world should also return no results
        let mut_count = world.query_mut::<(ReqM<String>,)>().count();
        assert_eq!(
            mut_count, 0,
            "Empty world should have no mutable query results"
        );

        println!("✅ Empty world test passed");
    }

    // Test 2: Single entity with all component variations
    {
        let mut world = World::init();
        let entity = world.spawn();

        // Add components of different types
        world.add(entity, String::from("test")).unwrap();
        world.add(entity, 42u32).unwrap();
        world.add(entity, 3.14f64).unwrap();
        world.add(entity, true).unwrap();

        // Test querying single entity with all required components
        let mut found = false;
        for (e, (s, n, f, b)) in world.query::<(Req<String>, Req<u32>, Req<f64>, Req<bool>)>() {
            assert_eq!(e, entity);
            assert_eq!(s, "test");
            assert_eq!(*n, 42u32);
            assert_eq!(*f, 3.14f64);
            assert_eq!(*b, true);
            found = true;
        }
        assert!(found, "Should find the single entity with all components");

        println!("✅ Single entity all components test passed");
    }

    // Test 3: Component type conflicts and overwriting
    {
        let mut world = World::init();
        let entity = world.spawn();

        // Add initial component
        world.add(entity, 100u32).unwrap();

        // Verify initial value
        for (_, (val,)) in world.query::<(u32,)>() {
            assert_eq!(*val, 100u32);
        }

        // Overwrite with new value (this tests if your ECS handles component replacement)
        world.replace(entity, 200u32).unwrap();

        // Verify overwritten value
        for (_, (val,)) in world.query::<(u32,)>() {
            assert_eq!(
                *val, 200u32,
                "Component should be overwritten, not duplicated"
            );
        }

        println!("✅ Component overwriting test passed");
    }

    // Test 4: Complex optional component patterns
    {
        let mut world = World::init();
        let mut entities_with_optional = HashSet::new();
        let mut all_entities = Vec::new();

        // Create entities with various optional component patterns
        for i in 0..20 {
            let e = world.spawn();
            all_entities.push(e);

            // All entities get these base components
            world.add(e, format!("entity-{}", i)).unwrap();
            world.add(e, i as u32).unwrap();

            // Optional components based on different patterns
            if i % 3 == 0 {
                world.add(e, i as f64).unwrap(); // Every 3rd has f64
                entities_with_optional.insert(e);
            }
            if i % 5 == 0 {
                world.add(e, i % 2 == 0).unwrap(); // Every 5th has bool
            }
            if i % 7 == 0 {
                world.add(e, "special-{}").unwrap(); // Every 7th has str
            }
        }

        // Test mixed required/optional query
        let mut count_with_f64 = 0;
        let mut count_without_f64 = 0;

        for (e, (name, id, opt_f64)) in world.query::<(Req<String>, Req<u32>, Opt<f64>)>() {
            // Verify base components are always present
            assert!(name.starts_with("entity-"));
            assert_eq!(*id, e.0); // Assuming EntityId.0 gives the numeric ID

            if entities_with_optional.contains(&e) {
                assert!(opt_f64.is_some(), "Entity {} should have f64 component", id);
                assert_eq!(*opt_f64.unwrap(), *id as f64);
                count_with_f64 += 1;
            } else {
                assert!(
                    opt_f64.is_none(),
                    "Entity {} should not have f64 component",
                    id
                );
                count_without_f64 += 1;
            }
        }

        assert_eq!(
            count_with_f64 + count_without_f64,
            20,
            "Should process all entities"
        );
        assert_eq!(count_with_f64, entities_with_optional.len());

        println!("✅ Complex optional components test passed");
    }

    // Test 5: Mutation edge cases
    {
        let mut world = World::init();
        let mut entities = Vec::new();

        // Create entities for mutation testing
        for i in 0..5 {
            let e = world.spawn();
            entities.push(e);
            world.add(e, i as u32).unwrap();
            world.add(e, format!("original-{}", i)).unwrap();
        }

        // Test partial mutation (only some components)
        for (_, (num,)) in world.query_mut::<(u32,)>() {
            *num *= 10; // Multiply all numbers by 10
        }

        // Verify partial mutation worked, strings unchanged
        for (e, (num, text)) in world.query::<(Req<u32>, Req<String>)>() {
            let expected_num = (e.0 as u32) * 10;
            assert_eq!(*num, expected_num);
            assert!(text.starts_with("original-"), "String should be unchanged");
        }

        // Test simultaneous mutation of multiple component types
        for (_, (num, text)) in world.query_mut::<(ReqM<u32>, ReqM<String>)>() {
            *num += 1;
            *text = format!("modified-{}", *num);
        }

        // Verify simultaneous mutations
        for (_, (num, text)) in world.query::<(Req<u32>, Req<String>)>() {
            let expected_text = format!("modified-{}", *num);
            assert_eq!(text, &expected_text);
        }

        println!("✅ Mutation edge cases test passed");
    }

    // Test 6: Component removal patterns
    {
        let mut world = World::init();
        let mut entities = Vec::new();

        // Create entities with multiple components
        for i in 0..10 {
            let e = world.spawn();
            entities.push(e);
            world.add(e, i as u32).unwrap();
            world.add(e, format!("entity-{}", i)).unwrap();
            world.add(e, i as f64 * 1.5).unwrap();
        }

        // Remove components from half the entities
        for (i, &e) in entities.iter().enumerate() {
            if i % 2 == 0 {
                world.remove::<f64>(e).unwrap();
            }
        }

        // Verify removal - entities without f64 should not appear in required f64 queries
        let mut count_with_f64 = 0;
        for (e, (_, _, f)) in world.query::<(Req<String>, Req<u32>, Req<f64>)>() {
            let entity_index = e.0 as usize;
            assert!(
                entity_index % 2 == 1,
                "Only odd-indexed entities should have f64"
            );
            assert_eq!(*f, entity_index as f64 * 1.5);
            count_with_f64 += 1;
        }
        assert_eq!(count_with_f64, 5, "Should find 5 entities with f64");

        // But all entities should still appear in optional f64 queries
        let mut total_count = 0;
        for (_, (_, _, _)) in world.query::<(Req<String>, Req<u32>, Opt<f64>)>() {
            total_count += 1;
            // Don't assert on opt_f value here since we know some are None
        }
        assert_eq!(
            total_count, 10,
            "All entities should appear in optional query"
        );

        println!("✅ Component removal test passed");
    }

    // Test 7: Entity destruction edge cases
    {
        let mut world = World::init();
        let mut entities = Vec::new();

        // Create a bunch of entities
        for i in 0..15 {
            let e = world.spawn();
            entities.push(e);
            world.add(e, i as u32).unwrap();
            world.add(e, format!("entity-{}", i)).unwrap();
        }

        // Destroy every 3rd entity
        let mut destroyed = HashSet::new();
        for (i, &e) in entities.iter().enumerate() {
            if i % 3 == 0 {
                world.destroy(e).unwrap();
                destroyed.insert(e);
            }
        }

        // Verify destroyed entities don't appear in queries
        for (e, _) in world.query::<(Req<u32>, Req<String>)>() {
            assert!(
                !destroyed.contains(&e),
                "Destroyed entity should not appear in queries"
            );
        }

        // Count remaining entities
        let remaining_count = world.query::<(u32,)>().count();
        assert_eq!(remaining_count, 15 - destroyed.len());

        // Try to add component to destroyed entity (should fail)
        if let Some(&destroyed_entity) = destroyed.iter().next() {
            let result = world.add(destroyed_entity, 999u32);
            // This should either fail or be handled gracefully
            // Adjust assertion based on your ECS behavior
            if result.is_ok() {
                println!("⚠️  Warning: Adding to destroyed entity succeeded (might be intended behavior)");
            }
        }

        println!("✅ Entity destruction test passed");
    }

    // Test 8: Query result consistency during iteration
    {
        let mut world = World::init();
        let mut entities = Vec::new();

        // Create entities
        for i in 0..8 {
            let e = world.spawn();
            entities.push(e);
            world.add(e, i as u32).unwrap();
        }

        // Test that we can collect query results without issues
        let results: Vec<_> = world.query::<(u32,)>().collect();
        assert_eq!(results.len(), 8);

        // Test nested queries (if your system supports it)
        let mut outer_count = 0;
        for (outer_e, (outer_val,)) in world.query::<(u32,)>() {
            outer_count += 1;

            // Inner query should still work and see all entities
            let inner_count = world.query::<(u32,)>().count();
            assert_eq!(inner_count, 8, "Inner query should see all entities");

            // Verify outer entity data is valid
            assert!(entities.contains(&outer_e));
            assert_eq!(*outer_val, outer_e.0);
        }
        assert_eq!(outer_count, 8);

        println!("✅ Query consistency test passed");
    }

    // Test 9: Zero-sized component types (if supported)
    {
        let mut world = World::init();

        // Using unit type as zero-sized component
        let e1 = world.spawn();
        let e2 = world.spawn();

        world.add(e1, ()).unwrap(); // Unit type component
        world.add(e1, 42u32).unwrap();
        world.add(e2, 84u32).unwrap(); // No unit component

        // Query for entities with unit component
        let mut count_with_unit = 0;
        for (e, (_, num)) in world.query::<(Req<()>, Req<u32>)>() {
            assert_eq!(e, e1, "Only e1 should have unit component");
            assert_eq!(*num, 42u32);
            count_with_unit += 1;
        }
        assert_eq!(count_with_unit, 1);

        // Optional unit component query
        let mut total_with_optional_unit = 0;
        for (_, (num, unit_opt)) in world.query::<(Req<u32>, Opt<()>)>() {
            total_with_optional_unit += 1;
            if *num == 42 {
                assert!(
                    unit_opt.is_some(),
                    "Entity with num=42 should have unit component"
                );
            } else {
                assert!(
                    unit_opt.is_none(),
                    "Entity with num=84 should not have unit component"
                );
            }
        }
        assert_eq!(total_with_optional_unit, 2);

        println!("✅ Zero-sized component test passed");
    }

    // Test 10: Large-scale stress test with mixed operations
    {
        let mut world = World::init();
        let mut all_entities = Vec::new();
        let entity_count = 10000;

        // Create many entities with varied component patterns
        for i in 0..entity_count {
            let e = world.spawn();
            all_entities.push(e);

            // Base components for all
            world.add(e, i as u32).unwrap();

            // Conditional components create different archetypes
            if i % 2 == 0 {
                world.add(e, format!("even-{}", i)).unwrap();
            }
            if i % 3 == 0 {
                world.add(e, i as f64 / 3.0).unwrap();
            }
            if i % 5 == 0 {
                world.add(e, i % 2 == 0).unwrap();
            }
        }

        // Test large-scale querying performance and correctness
        let start_time = std::time::Instant::now();

        // Query 1: All entities (should find all 1000)
        let all_count = world.query::<(u32,)>().count();
        assert_eq!(all_count, entity_count);

        // Query 2: Entities with string component (every 2nd = 500)
        let string_count = world.query::<(Req<u32>, Req<String>)>().count();
        assert_eq!(string_count, entity_count / 2);

        // Query 3: Complex optional query
        let mut complex_query_count = 0;
        let mut entities_with_all_optionals = 0;

        for (_, (id, s_opt, f_opt, b_opt)) in
            world.query::<(Req<u32>, Opt<String>, Opt<f64>, Opt<bool>)>()
        {
            complex_query_count += 1;
            let i = *id as usize;

            // Verify optional component presence matches creation pattern
            assert_eq!(s_opt.is_some(), i % 2 == 0);
            assert_eq!(f_opt.is_some(), i % 3 == 0);
            assert_eq!(b_opt.is_some(), i % 5 == 0);

            if s_opt.is_some() && f_opt.is_some() && b_opt.is_some() {
                entities_with_all_optionals += 1;
            }
        }

        assert_eq!(complex_query_count, entity_count);

        // Mathematical verification: entities divisible by 2*3*5= 30
        let expected_all_optionals = ((entity_count - 1) / 30) + 1;

        assert_eq!(
            entities_with_all_optionals, expected_all_optionals,
            "Entities with all optional components should be divisible by 30"
        );

        let query_time = start_time.elapsed();
        println!("✅ Large-scale test passed in {:?}", query_time);

        // Test large-scale mutations
        let mut_start = std::time::Instant::now();

        for (_, (id,)) in world.query_mut::<(u32,)>() {
            *id = id.wrapping_add(1_000_000); // Large number to test overflow handling
        }

        // Verify all mutations
        for (_, (id,)) in world.query::<(u32,)>() {
            assert!(*id >= 1_000_000, "All IDs should be incremented by 1000000");
        }

        let mut_time = mut_start.elapsed();
        println!("✅ Large-scale mutation test passed in {:?}", mut_time);
    }

    // Test 11: Error condition testing
    {
        let mut world = World::init();
        let entity = world.spawn();

        // Test removing non-existent component
        let remove_result = world.remove::<String>(entity);
        // Should either return an error or handle gracefully
        if remove_result.is_ok() {
            println!("⚠️  Removing non-existent component succeeded (might be intended)");
        }

        // Test operations on non-existent entity
        let fake_entity = EntityId(999999);
        let add_result = world.add(fake_entity, 42u32);
        if add_result.is_ok() {
            println!("⚠️  Adding to non-existent entity succeeded (check if intended)");
        }

        let remove_result = world.remove::<u32>(fake_entity);
        if remove_result.is_ok() {
            println!("⚠️  Removing from non-existent entity succeeded (check if intended)");
        }

        println!("✅ Error condition test completed");
    }

    println!("\n🎉 All comprehensive ECS tests passed!");
}
