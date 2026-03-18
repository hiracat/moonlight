use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use mlua::prelude::*;
use winit::keyboard::KeyCode;

use crate::{
    components::*,
    core::{Controllable, Engine, Keyboard, MouseState},
    ecs::{DynamicComponent, EntityId, QueryInfo, World},
    physics::{Aabb, Collider, RigidBody},
    resources::{Animated, Animation, Material, Mesh, Skeleton, Skybox, Texture},
};
pub trait PrintOnError {
    fn print_on_error(self);
}

impl<T, E: std::fmt::Debug> PrintOnError for Result<T, E> {
    fn print_on_error(self) {
        if let Err(e) = self {
            eprintln!("Error: {:?}", e);
        }
    }
}
use ultraviolet as uv;

pub struct LuaVM {
    lua: Lua,
    path: String,
    last_modified: std::time::SystemTime,
}
impl LuaVM {
    pub fn new(path: &str) -> LuaVM {
        let lua = Lua::new();
        let src = std::fs::read_to_string(path).unwrap();
        let err = lua.load(src).exec();
        err.print_on_error();
        dbg!(&path);
        Self {
            lua: lua,
            path: path.to_string(),
            last_modified: std::time::SystemTime::now(),
        }
    }

    pub fn run_script(
        &mut self,
        world: &mut World,
        engine: &mut Engine,
        entry_point: &str,
    ) -> Result<(), mlua::Error> {
        self.maybe_reload()?;
        let function: mlua::Function = self.lua.globals().get(entry_point)?;
        // make the lua user data
        let lua_world_ud = self.lua.create_userdata(LuaWorld {
            world: world as *mut World,
            registry: LuaComponentRegistry::new(),
            query_active: false,
        })?;
        let lua_engine_ud = self.lua.create_userdata(LuaEngine {
            engine: engine as *mut Engine,
        })?;
        function.call::<()>((lua_world_ud, lua_engine_ud))?;
        Ok(())
    }

    pub fn maybe_reload(&mut self) -> Result<(), mlua::Error> {
        let modified = std::fs::metadata(&self.path)?.modified()?;
        if modified > self.last_modified {
            let src = std::fs::read_to_string(&self.path)?;
            self.lua.load(src).exec()?;
            self.last_modified = modified;
        }

        Ok(())
    }
}

pub struct LuaWorld {
    pub world: *mut World,
    pub registry: LuaComponentRegistry,

    query_active: bool,
}
pub struct LuaEngine {
    pub engine: *mut Engine,
}

// ── Ref wrappers (ECS query results) ─────────────────────────────────────────

struct ColliderRef(*mut Collider);
struct TransformRef(*mut Transform);
struct RigidBodyRef(*mut RigidBody);
struct PointLightRef(*mut PointLight);
struct AmbientLightRef(*mut AmbientLight);
struct DirectionalLightRef(*mut DirectionalLight);
struct CameraRef(*mut Camera);
struct MeshRef(*mut Mesh);
struct MaterialRef(*mut Material);
struct TextureRef(*mut Texture);
struct SkyboxRef(*mut Skybox);
struct SkeletonRef(*mut Skeleton);
struct AnimationRef(*mut Animation);
struct AnimatedRef(*mut Animated);

// ── Registry ─────────────────────────────────────────────────────────────────

pub struct LuaComponentRegistry {
    pub typeids: HashMap<String, TypeId>,
    pub to_lua: HashMap<String, fn(*mut u8, &Lua) -> LuaResult<LuaAnyUserData>>,
    pub from_lua_table: HashMap<String, fn(&LuaTable) -> LuaResult<Box<dyn DynamicComponent>>>,
    pub from_userdata: HashMap<String, fn(&LuaAnyUserData) -> LuaResult<Box<dyn DynamicComponent>>>,
}

trait FromLuaTable {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self>
    where
        Self: Sized;
}

impl LuaComponentRegistry {
    pub fn new() -> Self {
        let mut typeids = HashMap::new();
        let mut to_lua: HashMap<String, fn(*mut u8, &Lua) -> LuaResult<LuaAnyUserData>> =
            HashMap::new();
        let mut from_lua: HashMap<String, fn(&LuaTable) -> LuaResult<Box<dyn DynamicComponent>>> =
            HashMap::new();
        let mut from_userdata: HashMap<
            String,
            fn(&LuaAnyUserData) -> LuaResult<Box<dyn DynamicComponent>>,
        > = HashMap::new();

        macro_rules! reg {
            // constructable from a table, just pure data
            ($name:literal, $t:ty, $ref:ident) => {
                typeids.insert($name.to_string(), TypeId::of::<$t>());
                to_lua.insert($name.to_string(), |ptr, lua| {
                    lua.create_userdata($ref(ptr as *mut $t))
                });
                from_lua.insert($name.to_string(), |table| {
                    let val = <$t>::from_lua_table(table)?;
                    Ok(Box::new(val) as Box<dyn DynamicComponent>)
                });
            };
            //ability to readback, because can be moved around/created on lua side via engine
            //functions
            (opaque, $name:literal, $t:ty, $ref:ident) => {
                typeids.insert($name.to_string(), TypeId::of::<$t>());
                to_lua.insert($name.to_string(), |ptr, lua| {
                    lua.create_userdata($ref(ptr as *mut $t))
                });
                from_userdata.insert($name.to_string(), |ud| {
                    Ok(Box::new(ud.borrow::<$t>()?.clone()) as Box<dyn DynamicComponent>)
                });
            };
            // purely read only, like keyboard and mouse state and deltatime
            (readonly, $name:literal, $t:ty) => {
                typeids.insert($name.to_string(), TypeId::of::<$t>());
                to_lua.insert($name.to_string(), |ptr, lua| {
                    lua.create_userdata(unsafe { (*(ptr as *mut $t)).clone() })
                });
            };
        }

        // components constructable from a lua table
        reg!("Transform", Transform, TransformRef);
        reg!("RigidBody", RigidBody, RigidBodyRef);
        reg!("Collider", Collider, ColliderRef);
        reg!("PointLight", PointLight, PointLightRef);
        reg!("AmbientLight", AmbientLight, AmbientLightRef);
        reg!("DirectionalLight", DirectionalLight, DirectionalLightRef);
        reg!("Camera", Camera, CameraRef);
        reg!("Material", Material, MaterialRef);

        // components with gpu state, are not constructable from a lua table
        typeids.insert("Mesh".to_string(), TypeId::of::<Mesh>());
        to_lua.insert("Mesh".to_string(), |ptr, lua| {
            lua.create_userdata(MeshRef(ptr as *mut u8 as *mut Mesh))
        });
        from_userdata.insert("Mesh".to_string(), |ud| {
            Ok(Box::new(ud.borrow::<Mesh>()?.clone()) as Box<dyn DynamicComponent>)
        });
        reg!(opaque, "Material", Material, MaterialRef);
        reg!(opaque, "Texture", Texture, TextureRef);
        reg!(opaque, "Skybox", Skybox, SkyboxRef);
        reg!(opaque, "Skeleton", Skeleton, SkeletonRef);
        reg!(opaque, "Animation", Animation, AnimationRef);
        reg!(opaque, "Animated", Animated, AnimatedRef);

        reg!(readonly, "Keyboard", Keyboard);
        reg!(readonly, "MouseState", MouseState);
        reg!(readonly, "Time", Time);
        reg!(readonly, "Controllable", Controllable);

        Self {
            typeids,
            to_lua,
            from_lua_table: from_lua,
            from_userdata: from_userdata,
        }
    }
}

// ── Ref UserData impls ────────────────────────────────────────────────────────

impl LuaUserData for TransformRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("x", |_, this| Ok(unsafe { (*this.0).position.x }));
        fields.add_field_method_set("x", |_, this, val: f32| {
            unsafe {
                (*this.0).position.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("y", |_, this| Ok(unsafe { (*this.0).position.y }));
        fields.add_field_method_set("y", |_, this, val: f32| {
            unsafe {
                (*this.0).position.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("z", |_, this| Ok(unsafe { (*this.0).position.z }));
        fields.add_field_method_set("z", |_, this, val: f32| {
            unsafe {
                (*this.0).position.z = val;
            }
            Ok(())
        });
        fields.add_field_method_get("sx", |_, this| Ok(unsafe { (*this.0).scale.x }));
        fields.add_field_method_set("sx", |_, this, val: f32| {
            unsafe {
                (*this.0).scale.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("sy", |_, this| Ok(unsafe { (*this.0).scale.y }));
        fields.add_field_method_set("sy", |_, this, val: f32| {
            unsafe {
                (*this.0).scale.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("sz", |_, this| Ok(unsafe { (*this.0).scale.z }));
        fields.add_field_method_set("sz", |_, this, val: f32| {
            unsafe {
                (*this.0).scale.z = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set_rotation_xz", |_, this, angle: f32| {
            unsafe {
                (*this.0).rotation = uv::Rotor3::from_rotation_xz(angle);
            }
            Ok(())
        });
        methods.add_method("get_forward", |_, this, ()| {
            let mut vec = uv::Vec3::new(0.0, 0.0, 1.0);
            unsafe {
                (*this.0).rotation.rotate_vec(&mut vec);
            }
            Ok((vec.x, vec.y, vec.z))
        });
    }
}

impl LuaUserData for RigidBodyRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("vx", |_, this| Ok(unsafe { (*this.0).velocity.x }));
        fields.add_field_method_set("vx", |_, this, val: f32| {
            unsafe {
                (*this.0).velocity.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("vy", |_, this| Ok(unsafe { (*this.0).velocity.y }));
        fields.add_field_method_set("vy", |_, this, val: f32| {
            unsafe {
                (*this.0).velocity.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("vz", |_, this| Ok(unsafe { (*this.0).velocity.z }));
        fields.add_field_method_set("vz", |_, this, val: f32| {
            unsafe {
                (*this.0).velocity.z = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<RigidBodyRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

impl LuaUserData for PointLightRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("r", |_, this| Ok(unsafe { (*this.0).color.x }));
        fields.add_field_method_set("r", |_, this, val: f32| {
            unsafe {
                (*this.0).color.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("g", |_, this| Ok(unsafe { (*this.0).color.y }));
        fields.add_field_method_set("g", |_, this, val: f32| {
            unsafe {
                (*this.0).color.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("b", |_, this| Ok(unsafe { (*this.0).color.z }));
        fields.add_field_method_set("b", |_, this, val: f32| {
            unsafe {
                (*this.0).color.z = val;
            }
            Ok(())
        });
        fields.add_field_method_get("brightness", |_, this| Ok(unsafe { (*this.0).brightness }));
        fields.add_field_method_set("brightness", |_, this, val: f32| {
            unsafe {
                (*this.0).brightness = val;
            }
            Ok(())
        });
        fields.add_field_method_get("linear", |_, this| Ok(unsafe { (*this.0).linear }));
        fields.add_field_method_set("linear", |_, this, val: f32| {
            unsafe {
                (*this.0).linear = val;
            }
            Ok(())
        });
        fields.add_field_method_get("quadratic", |_, this| Ok(unsafe { (*this.0).quadratic }));
        fields.add_field_method_set("quadratic", |_, this, val: f32| {
            unsafe {
                (*this.0).quadratic = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<PointLightRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

impl LuaUserData for AmbientLightRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("r", |_, this| Ok(unsafe { (*this.0).color.x }));
        fields.add_field_method_set("r", |_, this, val: f32| {
            unsafe {
                (*this.0).color.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("g", |_, this| Ok(unsafe { (*this.0).color.y }));
        fields.add_field_method_set("g", |_, this, val: f32| {
            unsafe {
                (*this.0).color.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("b", |_, this| Ok(unsafe { (*this.0).color.z }));
        fields.add_field_method_set("b", |_, this, val: f32| {
            unsafe {
                (*this.0).color.z = val;
            }
            Ok(())
        });
        fields.add_field_method_get("intensity", |_, this| Ok(unsafe { (*this.0).intensity }));
        fields.add_field_method_set("intensity", |_, this, val: f32| {
            unsafe {
                (*this.0).intensity = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<AmbientLightRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

impl LuaUserData for DirectionalLightRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("r", |_, this| Ok(unsafe { (*this.0).color.x }));
        fields.add_field_method_set("r", |_, this, val: f32| {
            unsafe {
                (*this.0).color.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("g", |_, this| Ok(unsafe { (*this.0).color.y }));
        fields.add_field_method_set("g", |_, this, val: f32| {
            unsafe {
                (*this.0).color.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("b", |_, this| Ok(unsafe { (*this.0).color.z }));
        fields.add_field_method_set("b", |_, this, val: f32| {
            unsafe {
                (*this.0).color.z = val;
            }
            Ok(())
        });
        fields.add_field_method_get("dx", |_, this| Ok(unsafe { (*this.0).from_position.x }));
        fields.add_field_method_set("dx", |_, this, val: f32| {
            unsafe {
                (*this.0).from_position.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("dy", |_, this| Ok(unsafe { (*this.0).from_position.y }));
        fields.add_field_method_set("dy", |_, this, val: f32| {
            unsafe {
                (*this.0).from_position.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("dz", |_, this| Ok(unsafe { (*this.0).from_position.z }));
        fields.add_field_method_set("dz", |_, this, val: f32| {
            unsafe {
                (*this.0).from_position.z = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<DirectionalLightRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

impl LuaUserData for CameraRef {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("x", |_, this| Ok(unsafe { (*this.0).position.x }));
        fields.add_field_method_set("x", |_, this, val: f32| {
            unsafe {
                (*this.0).position.x = val;
            }
            Ok(())
        });
        fields.add_field_method_get("y", |_, this| Ok(unsafe { (*this.0).position.y }));
        fields.add_field_method_set("y", |_, this, val: f32| {
            unsafe {
                (*this.0).position.y = val;
            }
            Ok(())
        });
        fields.add_field_method_get("z", |_, this| Ok(unsafe { (*this.0).position.z }));
        fields.add_field_method_set("z", |_, this, val: f32| {
            unsafe {
                (*this.0).position.z = val;
            }
            Ok(())
        });
        fields.add_field_method_get("pitch", |_, this| Ok(unsafe { (*this.0).pitch }));
        fields.add_field_method_set("pitch", |_, this, val: f32| {
            unsafe {
                (*this.0).pitch = val;
            }
            Ok(())
        });
        fields.add_field_method_get("yaw", |_, this| Ok(unsafe { (*this.0).yaw }));
        fields.add_field_method_set("yaw", |_, this, val: f32| {
            unsafe {
                (*this.0).yaw = val;
            }
            Ok(())
        });
        fields.add_field_method_get("fov", |_, this| Ok(unsafe { (*this.0).fov_rads }));
        fields.add_field_method_set("fov", |_, this, val: f32| {
            unsafe {
                (*this.0).fov_rads = val;
            }
            Ok(())
        });
    }
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<CameraRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

// opaque refs — set only
impl LuaUserData for MeshRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<MeshRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for MaterialRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<MaterialRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for TextureRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<TextureRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for SkyboxRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<SkyboxRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for SkeletonRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<SkeletonRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for AnimationRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<AnimationRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}
impl LuaUserData for AnimatedRef {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_, this, val: LuaAnyUserData| {
            unsafe {
                std::ptr::copy_nonoverlapping(val.borrow::<AnimatedRef>()?.0, this.0, 1);
            }
            Ok(())
        });
    }
}

// ── Owned variants (returned from ResourceManager, not ECS pointers) ─────────

impl LuaUserData for Mesh {}
impl LuaUserData for Texture {}

impl LuaUserData for Animated {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("time", |_, this| Ok(this.time));
        fields.add_field_method_set("time", |_, this, val: f32| {
            this.time = val;
            Ok(())
        });
    }
}

impl LuaUserData for LuaEngine {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("create_texture", |_, this, path: String| {
            Ok(unsafe { (*this.engine).resource_manager.create_texture(&path) })
        });
        methods.add_method_mut("create_cubemap", |_, this, paths: Vec<String>| {
            let paths_str: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
            Ok(unsafe { (*this.engine).resource_manager.create_cubemap(&paths_str) })
        });
        methods.add_method_mut("load_gltf_asset", |_, this, path: String| {
            let (mesh, animated) =
                unsafe { (*this.engine).resource_manager.load_gltf_asset(&path) };
            Ok((mesh, animated))
        });
    }
}

impl LuaUserData for Controllable {}
impl LuaUserData for EntityId {}

impl LuaUserData for Keyboard {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("is_down", |_, this: &Keyboard, key: String| {
            Ok(this.is_down(parse_keycode(&key)))
        });
    }
}

fn parse_keycode(s: &str) -> KeyCode {
    match s {
        "w" => KeyCode::KeyW,
        "a" => KeyCode::KeyA,
        "s" => KeyCode::KeyS,
        "d" => KeyCode::KeyD,
        "space" => KeyCode::Space,
        "shift" => KeyCode::ShiftLeft,
        _ => KeyCode::F35,
    }
}

impl LuaUserData for MouseState {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("x", |_, this| Ok(this.x));
        fields.add_field_method_get("y", |_, this| Ok(this.y));
        fields.add_field_method_get("locked", |_, this| Ok(this.locked));
    }
}
impl LuaUserData for Time {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("delta_time", |_, this| Ok(this.delta_time));
    }
}

impl LuaUserData for LuaWorld {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("spawn", |_, this, table: LuaTable| {
            if this.query_active {
                return Err(mlua::Error::runtime("cannot spawn with active query, maybe create a list of spawns to do after query completes"));
            }
            let mut components = Vec::new();
            for pair in table.pairs::<String, LuaValue>() {
                let (k, val) = pair?;
                match val {
                    LuaValue::Table(val) => {
                        let from_lua = this
                            .registry
                            .from_lua_table
                            .get(&k)
                            .ok_or(mlua::Error::runtime(format!("unknown component: {}", k)))?;
                        components.push(from_lua(&val)?);
                    }
                    LuaValue::UserData(op) => {
                        let from_userdata = this
                            .registry
                            .from_userdata
                            .get(&k)
                            .ok_or(mlua::Error::runtime(format!("unknown component {}", k)))?;
                        components.push(from_userdata(&op)?);
                    }
                    _ => return Err(mlua::Error::runtime(format!("invalid component {}", k))),
                }
            }
            let entity = unsafe { &mut *this.world }.spawn();
            for component in components {
                component.add_to_world(unsafe { &mut *this.world }, entity);
            }
            Ok(entity)
        });
        methods.add_method_mut("despawn", |_, this, val: LuaAnyUserData| {
            if this.query_active {
                return Err(mlua::Error::runtime("cannot despawn with active query, maybe create a list of despawns to do after query completes"));
            }
            //HACK:, should return this error somehow, i dont feel like it rn tho, need to use the
            //thiserror crate or something
            let _ = unsafe { &mut *this.world }.despawn(*val.borrow::<EntityId>()?);
            Ok(())
        });
        methods.add_method_mut("end_query", |_, this: &mut LuaWorld, _: ()| {
            this.query_active = false;
            Ok(())
        });
        methods.add_method_mut(
            "get_resource",
            |lua, this, name: String| -> LuaResult<LuaAnyUserData> {
                let typeid = this
                    .registry
                    .typeids
                    .get(&name)
                    .ok_or(mlua::Error::runtime(format!(
                        "resoource not registered {}",
                        name
                    )))?;
                let any = unsafe { &mut *this.world }
                    .get_mut_resource_dyn(typeid)
                    .ok_or(mlua::Error::runtime(format!(
                        "resource not found: {}",
                        name
                    )))?;
                let ptr = any.as_mut() as *mut dyn Any as *mut u8;

                let to_lua = this
                    .registry
                    .to_lua
                    .get(&name)
                    .ok_or(mlua::Error::runtime("component not registered"))?;

                let userdata = to_lua(ptr, lua)?;

                Ok(userdata)
            },
        );
        methods.add_method_mut(
            "get_component",
            |lua, this, (entity, name): (LuaAnyUserData, String)| -> LuaResult<LuaAnyUserData> {
                let entity = *entity.borrow::<EntityId>()?;
                let typeid = this
                    .registry
                    .typeids
                    .get(&name)
                    .ok_or(mlua::Error::runtime(format!("unknown component: {}", name)))?;
                let to_lua =
                    this.registry
                        .to_lua
                        .get(&name)
                        .ok_or(mlua::Error::runtime(format!(
                            "component not accessible from lua: {}",
                            name
                        )))?;
                let ptr = unsafe { &mut *this.world }
                    .get_component_dyn(entity, *typeid)
                    .ok_or(mlua::Error::runtime(format!(
                        "entity does not have component: {}",
                        name
                    )))?;
                to_lua(ptr, lua)
            },
        );
        methods.add_method_mut("query", |lua, this, val: LuaTable| -> LuaResult<LuaTable> {
            if this.query_active {
                return Err(mlua::Error::runtime(
                    "cannot have multiple queries active simultaniously",
                ));
            }
            this.query_active = true;
            let req = val
                .get::<LuaTable>("req")
                .map(|t| {
                    t.sequence_values::<String>()
                        .collect::<LuaResult<Vec<String>>>()
                })
                .unwrap_or(Ok(vec![]))?;
            if req.len() == 0 {
                return Err(mlua::Error::runtime(
                    "queries must include at least one req component",
                ));
            }
            let opt = val
                .get::<LuaTable>("opt")
                .map(|t| {
                    t.sequence_values::<String>()
                        .collect::<LuaResult<Vec<String>>>()
                })
                .unwrap_or(Ok(vec![]))?;
            let without = val
                .get::<LuaTable>("without")
                .map(|t| {
                    t.sequence_values::<String>()
                        .collect::<LuaResult<Vec<String>>>()
                })
                .unwrap_or(Ok(vec![]))?;

            let registry = &this.registry;
            let query_info = QueryInfo {
                req_typeids: req
                    .iter()
                    .filter_map(|n| registry.typeids.get(n).copied())
                    .collect(),
                opt_typeids: opt
                    .iter()
                    .filter_map(|n| registry.typeids.get(n).copied())
                    .collect(),
                not_typeids: without
                    .iter()
                    .filter_map(|n| registry.typeids.get(n).copied())
                    .collect(),
            };

            let raw = unsafe { &mut *this.world }.dyn_query_mut(&query_info);
            let rows = lua.create_table()?;
            for (i, (entity, req_ptrs, opt_ptrs)) in raw.into_iter().enumerate() {
                let row = lua.create_table()?;
                row.set("entity", entity)?;

                // req components — always present
                for (j, name) in req.iter().enumerate() {
                    if let Some(maker) = registry.to_lua.get(name) {
                        row.set(name.as_str(), maker(req_ptrs[j] as *mut u8, lua)?)?;
                    }
                }

                // opt components — may be None
                for (j, name) in opt.iter().enumerate() {
                    if let Some(maker) = registry.to_lua.get(name) {
                        match opt_ptrs[j] {
                            Some(ptr) => row.set(name.as_str(), maker(ptr as *mut u8, lua)?)?,
                            None => row.set(name.as_str(), LuaNil)?,
                        }
                    }
                }

                rows.set(i + 1, row)?;
            }

            Ok(rows)
        });
        methods.add_method_mut(
            "add_component",
            |_, this, (entity, name, val): (LuaAnyUserData, String, LuaValue)| {
                if this.query_active {
                    return Err(mlua::Error::runtime(
                        "cannot add component with active query",
                    ));
                }
                let entity = *entity.borrow::<EntityId>()?;
                let component: Box<dyn DynamicComponent> =
                    match val {
                        LuaValue::Table(t) => {
                            let from_lua = this.registry.from_lua_table.get(&name).ok_or(
                                mlua::Error::runtime(format!("unknown component: {}", name)),
                            )?;
                            from_lua(&t)?
                        }
                        LuaValue::UserData(ud) => {
                            let from_userdata = this.registry.from_userdata.get(&name).ok_or(
                                mlua::Error::runtime(format!("unknown component: {}", name)),
                            )?;
                            from_userdata(&ud)?
                        }
                        _ => {
                            return Err(mlua::Error::runtime(format!(
                                "invalid component value for {}",
                                name
                            )))
                        }
                    };
                component.add_to_world(unsafe { &mut *this.world }, entity);
                Ok(())
            },
        );

        methods.add_method_mut(
            "remove_component",
            |_, this, (entity, name): (LuaAnyUserData, String)| {
                if this.query_active {
                    return Err(mlua::Error::runtime(
                        "cannot remove component with active query",
                    ));
                }
                let entity = *entity.borrow::<EntityId>()?;
                let typeid = this
                    .registry
                    .typeids
                    .get(&name)
                    .ok_or(mlua::Error::runtime(format!("unknown component: {}", name)))?;
                unsafe { &mut *this.world }
                    .remove_dyn(entity, *typeid)
                    .map_err(|e| {
                        mlua::Error::runtime(format!("failed to remove component: {:?}", e))
                    })?;
                Ok(())
            },
        );
    }
}
impl FromLuaTable for Transform {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        Ok(Transform::from(
            Some(uv::Vec3::new(
                table.get("x")?,
                table.get("y")?,
                table.get("z")?,
            )),
            None,
            Some(uv::Vec3::new(
                table.get("sx").unwrap_or(1.0),
                table.get("sy").unwrap_or(1.0),
                table.get("sz").unwrap_or(1.0),
            )),
        ))
    }
}

impl FromLuaTable for RigidBody {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        let mut rb = RigidBody::new();
        rb.velocity = uv::Vec3::new(
            table.get("vx").unwrap_or(0.0),
            table.get("vy").unwrap_or(0.0),
            table.get("vz").unwrap_or(0.0),
        );
        Ok(rb)
    }
}

impl FromLuaTable for PointLight {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        Ok(PointLight::new(
            uv::Vec3::new(table.get("r")?, table.get("g")?, table.get("b")?),
            table.get("brightness")?,
            table.get("linear").ok(),
            table.get("quadratic").ok(),
        ))
    }
}

impl FromLuaTable for AmbientLight {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        Ok(AmbientLight::create(
            uv::Vec3::new(table.get("r")?, table.get("g")?, table.get("b")?),
            table.get("intensity")?,
        ))
    }
}

impl FromLuaTable for DirectionalLight {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        Ok(DirectionalLight::create(
            uv::Vec4::new(table.get("dx")?, table.get("dy")?, table.get("dz")?, 1.0),
            uv::Vec3::new(table.get("r")?, table.get("g")?, table.get("b")?),
        ))
    }
}

impl FromLuaTable for Camera {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        Ok(Camera::create(
            uv::Vec3::new(table.get("x")?, table.get("y")?, table.get("z")?),
            table.get("fov")?,
            table.get("near")?,
            table.get("far")?,
            table.get("aspect")?,
        ))
    }
}
impl FromLuaTable for Material {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        let texture = table.get::<LuaAnyUserData>("albedo")?;
        let texture = texture.borrow::<Texture>()?.clone();
        Ok(Material::create(texture))
    }
}

impl LuaUserData for ColliderRef {}

impl FromLuaTable for Collider {
    fn from_lua_table(table: &LuaTable) -> LuaResult<Self> {
        let collider_type: String = table.get("type").unwrap_or("aabb".to_string());
        match collider_type.as_str() {
            "aabb" => Ok(Collider::Aabb(Aabb::new(
                uv::Vec3::new(
                    table.get("extent_x").unwrap_or(0.5),
                    table.get("extent_y").unwrap_or(0.5),
                    table.get("extent_z").unwrap_or(0.5),
                ),
                uv::Vec3::new(
                    table.get("offset_x").unwrap_or(0.0),
                    table.get("offset_y").unwrap_or(0.0),
                    table.get("offset_z").unwrap_or(0.0),
                ),
            ))),
            // add more shapes like this in the future
            // "sphere" => Ok(Collider::Sphere(Sphere::new(
            //     table.get("radius").unwrap_or(0.5),
            //     uv::Vec3::new(
            //         table.get("ox").unwrap_or(0.0),
            //         table.get("oy").unwrap_or(0.0),
            //         table.get("oz").unwrap_or(0.0),
            //     ),
            // ))),
            // "capsule" => Ok(Collider::Capsule(Capsule::new(
            //     table.get("radius").unwrap_or(0.5),
            //     table.get("height").unwrap_or(1.0),
            //     uv::Vec3::new(
            //         table.get("ox").unwrap_or(0.0),
            //         table.get("oy").unwrap_or(0.0),
            //         table.get("oz").unwrap_or(0.0),
            //     ),
            // ))),
            _ => Err(mlua::Error::runtime(format!(
                "unknown collider type: {}",
                collider_type
            ))),
        }
    }
}
