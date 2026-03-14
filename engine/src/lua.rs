use std::any::TypeId;

use mlua::prelude::*;

struct LuaRunner {
    lua: Lua,
}
impl LuaRunner {
    fn init() {}
}

fn init_lua() {
    let lua = Lua::new();

    let map_table = lua.create_table()?;
    map_table.set(1, "one")?;
    map_table.set("two", 2)?;

    lua.globals().set("map_table", map_table)?;

    lua.load("for k,v in pairs(map_table) do print(k,v) end")
        .exec()?;

    Ok(())
}

struct LuaQuery {
    pub with: Vec<TypeId>,
    pub opt: Vec<TypeId>,
    pub not: Vec<TypeId>,
}
impl LuaQuery {
    fn get_result(&self, world: &World) -> LuaTable {}
}
