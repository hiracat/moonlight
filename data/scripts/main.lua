---@type boolean
local ui_initialized = false

---------------------------------------------------------------------
-- WIDGET ACCESSORS
---------------------------------------------------------------------

local function slider(widgets, name)
	local w = widgets[name]
	return (w and w.type == "Slider") and w.data or nil
end

local function button(widgets, name)
	local w = widgets[name]
	return (w and w.type == "Button") and w.data or nil
end

local function text_input(widgets, name)
	local w = widgets[name]
	return (w and w.type == "TextInput") and w.data or nil
end

local function number_input(widgets, name)
	local w = widgets[name]
	return (w and w.type == "NumberInput") and w.data or nil
end

local function set_label(widgets, name, text)
	local w = widgets[name]
	if w and w.type == "Label" then
		w.data.text = text
	end
end

---------------------------------------------------------------------
-- ENTITY HELPERS
---------------------------------------------------------------------

---@param world any
---@param name string?
---@return any?
local function find_entity(world, name)
	if not name or name == "" then
		return nil
	end
	return world:find(name)
end

---------------------------------------------------------------------
-- WIDGET FACTORY HELPERS
---------------------------------------------------------------------

local function Slider(label, value, min, max)
	return { type = "Slider", label = label, data = { value = value, min = min, max = max } }
end

local function Button(label)
	return { type = "Button", label = label, data = { clicked = false } }
end

local function TextInput(label, placeholder)
	return { type = "TextInput", label = label, data = { value = placeholder or "" } }
end

local function NumberInput(label, value, min, max)
	return { type = "NumberInput", label = label, data = { value = value, min = min, max = max } }
end

local function Label(label, default_text)
	return { type = "Label", label = label, data = { text = default_text or "idle" } }
end

local function Separator()
	return { type = "Separator" }
end

---------------------------------------------------------------------
-- INIT UI
---------------------------------------------------------------------

local function init_ui(ui)
	ui.widgets = {
		-- Terrain / Global Light --
		["terrain.light.r"] = Slider("Red", 1.0, 0.0, 1.0),
		["terrain.light.g"] = Slider("Green", 0.8, 0.0, 1.0),
		["terrain.light.b"] = Slider("Blue", 0.4, 0.0, 1.0),
		["terrain.light.brightness"] = Slider("Brightness", 3.0, 0.0, 20.0),
		["terrain.light.linear"] = Slider("Linear Falloff", 0.22, 0.0, 2.0),
		["terrain.light.quadratic"] = Slider("Quadratic Falloff", 0.20, 0.0, 5.0),
		["terrain.heightmap.res"] = Slider("Resolution", 1000, 5, 5000),

		-- Player / Camera --
		["player.sprint_speed"] = Slider("Sprint Speed", 20, 5, 200),
		["camera.offset.x"] = Slider("Offset X", 0, -300, 300),
		["camera.offset.y"] = Slider("Offset Y", 0, -100, 300),
		["camera.offset.z"] = Slider("Offset Z", 0, -100, 300),
		["world.gravity"] = Slider("Gravity", 9.8, -10, 30),

		-- Entity Transform --
		["entity.name"] = TextInput("Entity Name"),
		["entity.pos.x"] = Slider("Position X", 0, -10000, 10000),
		["entity.pos.y"] = Slider("Position Y", 0, -10000, 10000),
		["entity.pos.z"] = Slider("Position Z", 0, -10000, 10000),
		["entity.rot.pitch"] = Slider("Pitch", 0, -360, 360),
		["entity.rot.yaw"] = Slider("Yaw", 0, -360, 360),
		["entity.rot.roll"] = Slider("Roll", 0, -360, 360),
		["entity.scale.x"] = Slider("Scale X", 1, 0.001, 1000),
		["entity.scale.y"] = Slider("Scale Y", 1, 0.001, 1000),
		["entity.scale.z"] = Slider("Scale Z", 1, 0.001, 1000),
		["entity.teleport.x"] = NumberInput("Teleport X", 0, -10000, 10000),
		["entity.teleport.y"] = NumberInput("Teleport Y", 0, -10000, 10000),
		["entity.teleport.z"] = NumberInput("Teleport Z", 0, -10000, 10000),
		["entity.btn.read"] = Button("Read Transform"),
		["entity.btn.apply"] = Button("Apply Transform"),
		["entity.btn.teleport"] = Button("Teleport Here"),
		["entity.status"] = Label("Status", "idle"),
		["entity.list"] = Label("Active Entities", "none"),

		-- Point Light --
		["light.name"] = TextInput("Light Entity Name"),
		["light.color.r"] = Slider("Red", 1.0, 0.0, 1.0),
		["light.color.g"] = Slider("Green", 1.0, 0.0, 1.0),
		["light.color.b"] = Slider("Blue", 1.0, 0.0, 1.0),
		["light.brightness"] = Slider("Brightness", 3.0, 0.0, 20.0),
		["light.linear"] = Slider("Linear Falloff", 0.22, 0.0, 2.0),
		["light.quadratic"] = Slider("Quadratic Falloff", 0.20, 0.0, 5.0),
		["light.btn.apply"] = Button("Apply Light"),
		["light.btn.read"] = Button("Read Light"),
		["light.status"] = Label("Status", "idle"),

		["separator"] = Separator(),
	}

	ui.schema = {
		windows = {
			{
				name = "Terrain & Global Lighting",
				fields = {
					{ type = "Field", data = "terrain.light.r" },
					{ type = "Field", data = "terrain.light.g" },
					{ type = "Field", data = "terrain.light.b" },
					{ type = "Field", data = "terrain.light.brightness" },
					{ type = "Field", data = "terrain.light.linear" },
					{ type = "Field", data = "terrain.light.quadratic" },
					{ type = "Field", data = "terrain.heightmap.res" },
				},
			},

			{
				name = "Player & Camera",
				fields = {
					{ type = "Field", data = "player.sprint_speed" },
					{ type = "Field", data = "camera.offset.x" },
					{ type = "Field", data = "camera.offset.y" },
					{ type = "Field", data = "camera.offset.z" },
					{ type = "Field", data = "world.gravity" },
				},
			},

			{
				name = "Entity Control",
				fields = {
					{ type = "Field", data = "entity.list" },
					{ type = "Field", data = "entity.name" },

					{ type = "Field", data = "entity.pos.x" },
					{ type = "Field", data = "entity.pos.y" },
					{ type = "Field", data = "entity.pos.z" },

					{ type = "Field", data = "entity.rot.pitch" },
					{ type = "Field", data = "entity.rot.yaw" },
					{ type = "Field", data = "entity.rot.roll" },

					{ type = "Field", data = "entity.scale.x" },
					{ type = "Field", data = "entity.scale.y" },
					{ type = "Field", data = "entity.scale.z" },

					{ type = "Field", data = "entity.btn.read" },
					{ type = "Field", data = "entity.btn.apply" },

					{ type = "Field", data = "separator" },

					{ type = "Field", data = "entity.teleport.x" },
					{ type = "Field", data = "entity.teleport.y" },
					{ type = "Field", data = "entity.teleport.z" },
					{ type = "Field", data = "entity.btn.teleport" },

					{ type = "Field", data = "entity.status" },
				},
			},

			{
				name = "Point Light Control",
				fields = {
					{ type = "Field", data = "light.name" },
					{ type = "Field", data = "light.color.r" },
					{ type = "Field", data = "light.color.g" },
					{ type = "Field", data = "light.color.b" },
					{ type = "Field", data = "light.brightness" },
					{ type = "Field", data = "light.linear" },
					{ type = "Field", data = "light.quadratic" },
					{ type = "Field", data = "light.btn.read" },
					{ type = "Field", data = "light.btn.apply" },
					{ type = "Field", data = "light.status" },
				},
			},
		},
	}
end

---------------------------------------------------------------------
-- APPLY / READ TRANSFORM HELPERS
---------------------------------------------------------------------

local function apply_transform(world, widgets)
	local entity = find_entity(world, text_input(widgets, "entity.name").value)
	if not entity then
		set_label(widgets, "entity.status", "not found")
		return
	end

	local t = world:get_component(entity, "Transform")
	if not t then
		set_label(widgets, "entity.status", "no Transform")
		return
	end

	t.position = {
		x = slider(widgets, "entity.pos.x").value,
		y = slider(widgets, "entity.pos.y").value,
		z = slider(widgets, "entity.pos.z").value,
	}
	t.rotation = {
		pitch = slider(widgets, "entity.rot.pitch").value,
		yaw = slider(widgets, "entity.rot.yaw").value,
		roll = slider(widgets, "entity.rot.roll").value,
	}
	t.scale = {
		x = slider(widgets, "entity.scale.x").value,
		y = slider(widgets, "entity.scale.y").value,
		z = slider(widgets, "entity.scale.z").value,
	}
	set_label(widgets, "entity.status", "transform applied")
end

local function read_transform(world, widgets)
	local entity = find_entity(world, text_input(widgets, "entity.name").value)
	if not entity then
		set_label(widgets, "entity.status", "not found")
		return
	end

	local t = world:get_component(entity, "Transform")
	if not t then
		set_label(widgets, "entity.status", "no Transform")
		return
	end

	slider(widgets, "entity.pos.x").value = t.position.x
	slider(widgets, "entity.pos.y").value = t.position.y
	slider(widgets, "entity.pos.z").value = t.position.z
	slider(widgets, "entity.rot.pitch").value = t.rotation.pitch
	slider(widgets, "entity.rot.yaw").value = t.rotation.yaw
	slider(widgets, "entity.rot.roll").value = t.rotation.roll
	slider(widgets, "entity.scale.x").value = t.scale.x
	slider(widgets, "entity.scale.y").value = t.scale.y
	slider(widgets, "entity.scale.z").value = t.scale.z
	set_label(widgets, "entity.status", "transform read")
end

local function teleport_entity(world, widgets)
	local entity = find_entity(world, text_input(widgets, "entity.name").value)
	if not entity then
		set_label(widgets, "entity.status", "not found")
		return
	end

	local t = world:get_component(entity, "Transform")
	if not t then
		set_label(widgets, "entity.status", "no Transform")
		return
	end

	t.position = {
		x = number_input(widgets, "entity.teleport.x").value,
		y = number_input(widgets, "entity.teleport.y").value,
		z = number_input(widgets, "entity.teleport.z").value,
	}
	set_label(widgets, "entity.status", "teleported")
end

---------------------------------------------------------------------
-- APPLY / READ LIGHT HELPERS
---------------------------------------------------------------------

local function apply_light(world, widgets)
	local entity = find_entity(world, text_input(widgets, "light.name").value)
	if not entity then
		set_label(widgets, "light.status", "not found")
		return
	end

	local light = world:get_component(entity, "PointLight")
	if not light then
		set_label(widgets, "light.status", "no PointLight")
		return
	end

	light.color = {
		x = slider(widgets, "light.color.r").value,
		y = slider(widgets, "light.color.g").value,
		z = slider(widgets, "light.color.b").value,
	}
	light.brightness = slider(widgets, "light.brightness").value
	light.linear = slider(widgets, "light.linear").value
	light.quadratic = slider(widgets, "light.quadratic").value
	set_label(widgets, "light.status", "light applied")
end

local function read_light(world, widgets)
	local entity = find_entity(world, text_input(widgets, "light.name").value)
	if not entity then
		set_label(widgets, "light.status", "not found")
		return
	end

	local light = world:get_component(entity, "PointLight")
	if not light then
		set_label(widgets, "light.status", "no PointLight")
		return
	end

	slider(widgets, "light.color.r").value = light.color.x
	slider(widgets, "light.color.g").value = light.color.y
	slider(widgets, "light.color.b").value = light.color.z
	slider(widgets, "light.brightness").value = light.brightness
	slider(widgets, "light.linear").value = light.linear
	slider(widgets, "light.quadratic").value = light.quadratic
	set_label(widgets, "light.status", "light read")
end

---------------------------------------------------------------------
-- UPDATE
---------------------------------------------------------------------

---@param world any
---@param engine any
function Update(world, engine)
	---@type UIStuff
	local ui = world:get_resource("UIStuff")

	---@type TerrainMap
	local terrainmap = world:get_resource("TerrainMap")

	---@type CameraOffset
	local offset = world:get_resource("CameraOffset")

	---@type Config
	local config = world:get_resource("Config")

	if not ui_initialized then
		init_ui(ui)
		ui_initialized = true
	end
	local widgets = ui.widgets

	-----------------------------------------------------------------
	-- PLAYER & WORLD SETTINGS (live, every frame)
	-----------------------------------------------------------------

	for _, p in ipairs(world:query({ req = { "Controllable" } })) do
		p.Controllable.sprint_speed = slider(widgets, "player.sprint_speed").value
	end

	offset.offset = {
		x = slider(widgets, "camera.offset.x").value,
		y = slider(widgets, "camera.offset.y").value,
		z = slider(widgets, "camera.offset.z").value,
	}

	config.gravity_strength = slider(widgets, "world.gravity").value
	terrainmap.resolution = slider(widgets, "terrain.heightmap.res").value

	-----------------------------------------------------------------
	-- ACTIVE ENTITY LIST
	-----------------------------------------------------------------

	local names = {}
	for _, p in ipairs(world:query({ req = { "EntityName" } })) do
		names[#names + 1] = p.EntityName.name
	end
	set_label(widgets, "entity.list", #names > 0 and table.concat(names, "\n") or "none")

	-----------------------------------------------------------------
	-- ENTITY CONTROL BUTTONS
	-----------------------------------------------------------------

	if button(widgets, "entity.btn.read").clicked then
		read_transform(world, widgets)
	end
	if button(widgets, "entity.btn.apply").clicked then
		apply_transform(world, widgets)
	end
	if button(widgets, "entity.btn.teleport").clicked then
		teleport_entity(world, widgets)
	end

	-----------------------------------------------------------------
	-- POINT LIGHT CONTROL BUTTONS
	-----------------------------------------------------------------

	if button(widgets, "light.btn.read").clicked then
		read_light(world, widgets)
	end
	if button(widgets, "light.btn.apply").clicked then
		apply_light(world, widgets)
	end

	ui.widgets = widgets
end
