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
-- TREE SPAWNING
---------------------------------------------------------------------

---@type boolean
local trees_spawned = false

local function spawn_trees(world, widgets)
	local count = math.floor(slider(widgets, "trees.count").value)
	local spread = slider(widgets, "trees.spread").value
	local scale_min = slider(widgets, "trees.scale_min").value
	local scale_max = slider(widgets, "trees.scale_max").value

	-- Grab the mesh from the existing "tree" entity
	local tree_template = world:find("tree")
	if not tree_template then
		set_label(widgets, "trees.status", "no 'tree' entity found")
		return
	end
	local template_mesh = world:get_component(tree_template, "Mesh") ---@type Mesh
	local template_material = world:get_component(tree_template, "Material") ---@type Material
	if not template_mesh then
		set_label(widgets, "trees.status", "tree has no Mesh component")
		return
	end

	-- Despawn existing spawned trees
	local to_despawn = {}
	for _, e in ipairs(world:query({ req = { "EntityName" } })) do
		if e.EntityName.name:sub(1, 5) == "tree_" then
			to_despawn[#to_despawn + 1] = e.entity
		end
	end
	for _, entity in ipairs(to_despawn) do
		world:despawn(entity)
	end

	local seed = math.floor(slider(widgets, "trees.seed").value)
	local function lcg_next(s)
		return (s * 1664525 + 1013904223) % (2 ^ 32)
	end
	local function lcg_float(s)
		local ns = lcg_next(s)
		return ns, (ns / (2 ^ 31)) - 1.0
	end

	for i = 1, count do
		local fx, fz, fs, fy
		seed, fx = lcg_float(seed)
		seed, fz = lcg_float(seed)
		seed, fs = lcg_float(seed)
		seed, fy = lcg_float(seed)

		local x = fx * spread
		local z = fz * spread
		local y = world:get_height(x, z)

		local t_scale = scale_min + (fs * 0.5 + 0.5) * (scale_max - scale_min)
		local yaw = fy * 180.0

		local entity = world:spawn("tree_" .. i)
		world:add_component(entity, nil, "Transform")
		local t = world:get_component(entity, "Transform")
		t.position = { x = x, y = y, z = z }
		t.scale = { x = t_scale, y = t_scale, z = t_scale }

		-- read it back immediately
		local t2 = world:get_component(entity, "Transform")

		world:add_component(entity, template_mesh, "Mesh")
		world:add_component(entity, template_material, "Material")
	end

	set_label(widgets, "trees.status", string.format("spawned %d trees", count))
end

local function despawn_trees(world, widgets)
	-- Despawn existing spawned trees
	local to_despawn = {}
	for _, e in ipairs(world:query({ req = { "EntityName" } })) do
		if e.EntityName.name:sub(1, 5) == "tree_" then
			to_despawn[#to_despawn + 1] = e.entity
		end
	end
	local count = 0
	for _, entity in ipairs(to_despawn) do
		world:despawn(entity)
		count = count + 1
	end
	set_label(widgets, "trees.status", string.format("despawn %d trees", count))
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
-- DAY / NIGHT CYCLE HELPERS
---------------------------------------------------------------------

local TAU = math.pi * 2

local function lerp(a, b, t)
	return a + (b - a) * t
end

--- Returns 0..1 brightness given normalised time-of-day (0=midnight, 0.5=noon, 1=midnight)
local function day_brightness(t)
	-- sine peaks at noon (t=0.5), is 0 at t=0 and t=1
	local raw = math.sin(t * math.pi) -- 0..1..0
	return math.max(0, raw) -- clamp negatives (below horizon)
end

local function apply_day_night(world, widgets, dt)
	-- Sliders
	local enabled = slider(widgets, "dnc.enabled")
	local speed_s = slider(widgets, "dnc.speed")
	local tod_s = slider(widgets, "dnc.time_of_day")
	local peak_s = slider(widgets, "dnc.peak_brightness")
	local ambient_s = slider(widgets, "dnc.ambient_min")

	if not (enabled and speed_s and tod_s and peak_s and ambient_s) then
		return
	end

	-- Advance time when cycle is running (enabled > 0.5 = on)
	if enabled.value > 0.5 then
		tod_s.value = (tod_s.value + speed_s.value * dt) % 1.0
	end

	local t = tod_s.value -- 0..1
	local bright = day_brightness(t) -- 0..1
	local peak = peak_s.value
	local amb_min = ambient_s.value

	-- Sun position: arc across the sky (XZ plane, Y is up)
	local angle = t * math.pi -- 0=east horizon, pi=west horizon
	local sun_x = math.cos(angle - math.pi * 0.5) * 500
	local sun_y = math.sin(angle) * 500
	local sun_z = 0.0

	-- Sun / directional light colour: warm at dawn/dusk, white at noon
	local dawn_t = math.max(0, 1 - bright * 4) -- 1 near horizon, 0 when high
	local sun_r = lerp(1.0, 1.0, dawn_t)
	local sun_g = lerp(1.0, lerp(0.9, 0.5, dawn_t), bright)
	local sun_b = lerp(1.0, lerp(0.6, 0.1, dawn_t), bright)

	local sun_brightness = bright * peak

	-- Update DirectionalLight
	local dir_light = world:get_resource("DirectionalLight")
	local dl = dir_light
	dl.from_position = { x = sun_x, y = sun_y, z = sun_z }
	dl.color = {
		x = sun_r * sun_brightness,
		y = sun_g * sun_brightness,
		z = sun_b * sun_brightness,
	}
	dir_light = dl

	-- Update AmbientLight: stays dim but never fully black
	local amb_light = world:get_resource("AmbientLight")

	local al = amb_light ---@type AmbientLight
	local intensity = lerp(amb_min, 1.0, bright)
	-- Night tint: cool blue; day: neutral
	al.color = {
		x = lerp(0.3, 1.0, bright),
		y = lerp(0.4, 1.0, bright),
		z = lerp(0.8, 1.0, bright),
	}
	al.intensity = intensity
	amb_light = al

	-- Human-readable time label  (00:00 – 24:00)
	local hour = math.floor(t * 24)
	local minute = math.floor((t * 24 - hour) * 60)
	set_label(widgets, "dnc.clock", string.format("%02d:%02d", hour, minute))
end

---------------------------------------------------------------------
-- INIT UI
---------------------------------------------------------------------

local function init_ui(ui)
	ui.widgets = {
		-- Tree Spawning --
		["trees.count"] = Slider("Tree Count", 200, 1, 2000),
		["trees.spread"] = Slider("Spread Radius", 500, 10, 5000),
		["trees.scale_min"] = Slider("Scale Min", 5.8, 0.01, 100.0),
		["trees.scale_max"] = Slider("Scale Max", 9.0, 0.01, 100.0),
		["trees.seed"] = Slider("Random Seed", 42, 0, 9999),
		["trees.btn.spawn"] = Button("Spawn Trees"),
		["trees.btn.despawn"] = Button("Clear Trees"),
		["trees.status"] = Label("Status", "idle"),
		-- Day / Night Cycle --
		["dnc.enabled"] = Slider("Cycle Running (0=off 1=on)", 1.0, 0.0, 1.0),
		["dnc.speed"] = Slider("Cycle Speed", 0.002, 0.0, 5.0),
		["dnc.time_of_day"] = Slider("Time of Day (0=midnight 0.5=noon)", 0.3, 0.0, 1.0),
		["dnc.peak_brightness"] = Slider("Peak Sun Brightness", 1.5, 0.0, 5.0),
		["dnc.ambient_min"] = Slider("Night Ambient Level", 0.05, 0.0, 0.5),
		["dnc.clock"] = Label("Clock", "00:00"),
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
					{
						type = "Scroll",
						data = { visible_lines = 10, items = { { type = "Field", data = "entity.list" } } },
					},

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
			{
				name = "Day / Night Cycle",
				fields = {
					{ type = "Field", data = "dnc.clock" },
					{ type = "Field", data = "dnc.enabled" },
					{ type = "Field", data = "dnc.speed" },
					{ type = "Field", data = "dnc.time_of_day" },
					{ type = "Field", data = "dnc.peak_brightness" },
					{ type = "Field", data = "dnc.ambient_min" },
				},
			},
			{
				name = "Tree Spawning",
				fields = {
					{ type = "Field", data = "trees.count" },
					{ type = "Field", data = "trees.spread" },
					{ type = "Field", data = "trees.scale_min" },
					{ type = "Field", data = "trees.scale_max" },
					{ type = "Field", data = "trees.seed" },
					{ type = "Field", data = "trees.btn.spawn" },
					{ type = "Field", data = "trees.btn.despawn" },
					{ type = "Field", data = "trees.status" },
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

	local time_res = world:get_resource("Time") ---@type Time
	apply_day_night(world, widgets, time_res and time_res.delta_time or 0.016)

	-----------------------------------------------------------------
	-- PLAYER & WORLD SETTINGS (live, every frame)
	-----------------------------------------------------------------

	for _, p in ipairs(world:query({ req = { "Controllable", "Animated", "RigidBody" } })) do
		p.Controllable.sprint_speed = slider(widgets, "player.sprint_speed").value
		local animated = p.Animated ---@type Animated
		local rigidbody = p.RigidBody ---@type RigidBody
		local speed = math.sqrt(
			rigidbody.velocity.y * rigidbody.velocity.y
				+ rigidbody.velocity.x * rigidbody.velocity.x
				+ rigidbody.velocity.z * rigidbody.velocity.z
		) * 0.35
		if speed > 0 then
			animated.speed = speed
			animated.mode = { type = "Loop", data = animated.available_animations[1] }
		else
			animated.mode = { type = "Stopped" }
			animated.stop_blend_progress = 0
		end
		p.Animated = animated
	end

	offset.offset = {
		x = slider(widgets, "camera.offset.x").value,
		y = slider(widgets, "camera.offset.y").value,
		z = slider(widgets, "camera.offset.z").value,
	}

	config.gravity_strength = slider(widgets, "world.gravity").value
	terrainmap.resolution = slider(widgets, "terrain.heightmap.res").value

	-----------------------------------------------------------------
	-- TREE SPAWNING BUTTONS
	-----------------------------------------------------------------

	if button(widgets, "trees.btn.spawn").clicked then
		spawn_trees(world, widgets)
	end
	if button(widgets, "trees.btn.despawn").clicked then
		despawn_trees(world, widgets)
	end
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
