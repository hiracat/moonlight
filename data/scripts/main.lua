---@type boolean
local ui_initialized = false

---@param widgets table<string, Widget>
---@param name string
---@return Slider?
local function slider(widgets, name)
	local w = widgets[name]
	return (w and w.type == "Slider") and w.data or nil
end

---@param widgets table<string, Widget>
---@param name string
---@return Button?
local function button(widgets, name)
	local w = widgets[name]
	return (w and w.type == "Button") and w.data or nil
end

---@param widgets table<string, Widget>
---@param name string
---@return TextInput?
local function text_input(widgets, name)
	local w = widgets[name]
	return (w and w.type == "TextInput") and w.data or nil
end

---@param widgets table<string, Widget>
---@param name string
---@return NumberInput?
local function number_input(widgets, name)
	local w = widgets[name]
	return (w and w.type == "NumberInput") and w.data or nil
end

---@param widgets table<string, Widget>
---@param name string
---@param text string
local function set_label(widgets, name, text)
	local w = widgets[name]
	if w and w.type == "Label" then
		w.data.text = text
	end
end

---@param world any
---@param name string?
---@return any?
local function find_entity(world, name)
	if not name or name == "" then
		return nil
	end
	return world:find(name)
end

---@param label string
---@param value number
---@param min number
---@param max number
---@return Widget_Slider
local function Slider(label, value, min, max)
	return { type = "Slider", data = { value = value, min = min, max = max, label = label } }
end

---@param label string
---@return Widget_Button
local function Button(label)
	return { type = "Button", data = { clicked = false, label = label } }
end

---@param label string
---@param placeholder string?
---@return Widget_TextInput
local function TextInput(label, placeholder)
	return { type = "TextInput", data = { value = placeholder or "", label = label } }
end

---@param label string
---@param value number
---@param min number
---@param max number
---@return Widget_NumberInput
local function NumberInput(label, value, min, max)
	return { type = "NumberInput", data = { value = value, min = min, max = max, label = label } }
end

---@param label string
---@param default_text string?
---@return Widget_Label
local function Label(label, default_text)
	return { type = "Label", label = label, data = { text = default_text or "idle" } }
end

---@return Widget_Separator
local function Separator()
	return { type = "Separator" }
end

local function spawn_trees(world, widgets)
	local count = math.floor(slider(widgets, "trees.count").value)
	local spread = slider(widgets, "trees.spread").value
	local scale_min = slider(widgets, "trees.scale_min").value
	local scale_max = slider(widgets, "trees.scale_max").value

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
		t.rotation = { pitch = 0, yaw = yaw, roll = 0 }

		world:add_component(entity, template_mesh, "Mesh")
		world:add_component(entity, template_material, "Material")
	end

	set_label(widgets, "trees.status", string.format("spawned %d trees", count))
end

local function despawn_trees(world, widgets)
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
	set_label(widgets, "trees.status", string.format("despawned %d trees", count))
end

---------------------------------------------------------------------
-- DAY / NIGHT CYCLE
---------------------------------------------------------------------

local function lerp(a, b, t)
	return a + (b - a) * t
end

local function day_brightness(t)
	local raw = math.sin(t * math.pi)
	return math.max(0, raw)
end

local function apply_day_night(world, widgets, dt)
	local enabled = slider(widgets, "dnc.enabled")
	local speed_s = slider(widgets, "dnc.speed")
	local tod_s = slider(widgets, "dnc.time_of_day")
	local peak_s = slider(widgets, "dnc.peak_brightness")
	local ambient_s = slider(widgets, "dnc.ambient_min")

	if not (enabled and speed_s and tod_s and peak_s and ambient_s) then
		return
	end

	if enabled.value > 0.5 then
		tod_s.value = (tod_s.value + speed_s.value * dt) % 1.0
	end

	local t = tod_s.value
	local bright = day_brightness(t)
	local peak = peak_s.value

	local angle = t * math.pi
	local sun_x = math.cos(angle - math.pi * 0.5) * 500
	local sun_y = math.sin(angle) * 500
	local sun_z = 0.0

	local dawn_t = math.max(0, 1 - bright * 4)
	local sun_r = 1.0
	local sun_g = lerp(lerp(0.9, 0.5, dawn_t), 1.0, 1 - bright)
	local sun_b = lerp(lerp(0.6, 0.1, dawn_t), 1.0, 1 - bright)
	local sun_br = bright * peak

	local dir_light = world:get_resource("DirectionalLight") ---@type DirectionalLight
	dir_light.sun_position = { x = sun_x, y = sun_y, z = sun_z }
	dir_light.sun_color = { x = sun_r * sun_br, y = sun_g * sun_br, z = sun_b * sun_br }

	local hour = math.floor(t * 24)
	local minute = math.floor((t * 24 - hour) * 60)
	set_label(widgets, "dnc.clock", string.format("%02d:%02d", hour, minute))
end

---------------------------------------------------------------------
-- SUN / SKY LIGHT HELPERS
---------------------------------------------------------------------

local function read_sun_sky(world, widgets)
	local dl = world:get_resource("DirectionalLight") ---@type DirectionalLight
	if not dl then
		set_label(widgets, "sun.status", "no DirectionalLight resource")
		return
	end

	-- Sun position
	slider(widgets, "sun.pos.x").value = dl.sun_position.x
	slider(widgets, "sun.pos.y").value = dl.sun_position.y
	slider(widgets, "sun.pos.z").value = dl.sun_position.z

	-- Sun color
	slider(widgets, "sun.color.r").value = dl.sun_color.x
	slider(widgets, "sun.color.g").value = dl.sun_color.y
	slider(widgets, "sun.color.b").value = dl.sun_color.z

	-- Sky zenith color
	slider(widgets, "sky.zenith.r").value = dl.sky_zenith_color.x
	slider(widgets, "sky.zenith.g").value = dl.sky_zenith_color.y
	slider(widgets, "sky.zenith.b").value = dl.sky_zenith_color.z

	-- Sky horizon color
	slider(widgets, "sky.horizon.r").value = dl.sky_horizon_color.x
	slider(widgets, "sky.horizon.g").value = dl.sky_horizon_color.y
	slider(widgets, "sky.horizon.b").value = dl.sky_horizon_color.z

	-- Sky gradient sharpness
	slider(widgets, "sky.gradient_sharpness").value = dl.sky_gradient_sharpness

	set_label(widgets, "sun.status", "read ok")
end

local function apply_sun_sky(world, widgets)
	local dl = world:get_resource("DirectionalLight") ---@type DirectionalLight
	if not dl then
		set_label(widgets, "sun.status", "no DirectionalLight resource")
		return
	end

	dl.sun_position = {
		x = slider(widgets, "sun.pos.x").value,
		y = slider(widgets, "sun.pos.y").value,
		z = slider(widgets, "sun.pos.z").value,
	}
	dl.sun_color = {
		x = slider(widgets, "sun.color.r").value,
		y = slider(widgets, "sun.color.g").value,
		z = slider(widgets, "sun.color.b").value,
	}
	dl.sky_zenith_color = {
		x = slider(widgets, "sky.zenith.r").value,
		y = slider(widgets, "sky.zenith.g").value,
		z = slider(widgets, "sky.zenith.b").value,
	}
	dl.sky_horizon_color = {
		x = slider(widgets, "sky.horizon.r").value,
		y = slider(widgets, "sky.horizon.g").value,
		z = slider(widgets, "sky.horizon.b").value,
	}
	dl.sky_gradient_sharpness = slider(widgets, "sky.gradient_sharpness").value

	set_label(widgets, "sun.status", "applied ok")
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
-- APPLY / READ POINT LIGHT HELPERS
---------------------------------------------------------------------
local function read_light(world, widgets)
	local entity = find_entity(world, text_input(widgets, "light.name").value)
	if not entity then
		set_label(widgets, "light.status", "not found")
		return
	end

	local light = world:get_component(entity, "PointLight") ---@type PointLight
	if not light then
		set_label(widgets, "light.status", "no PointLight")
		return
	end

	slider(widgets, "light.color.r").value = light.color.x
	slider(widgets, "light.color.g").value = light.color.y
	slider(widgets, "light.color.b").value = light.color.z
	slider(widgets, "light.size").value = light.size
	set_label(widgets, "light.status", "light read")
end

local function apply_light(world, widgets)
	local entity = find_entity(world, text_input(widgets, "light.name").value)
	if not entity then
		set_label(widgets, "light.status", "not found")
		return
	end

	local light = world:get_component(entity, "PointLight") ---@type PointLight
	if not light then
		set_label(widgets, "light.status", "no PointLight")
		return
	end

	light.color = {
		x = slider(widgets, "light.color.r").value,
		y = slider(widgets, "light.color.g").value,
		z = slider(widgets, "light.color.b").value,
	}
	light.size = slider(widgets, "light.size").value
	set_label(widgets, "light.status", "light applied")
end

---------------------------------------------------------------------
-- INIT UI
---------------------------------------------------------------------

local function init_ui(ui)
	ui.widgets = {
		-- ── Tree Spawning ────────────────────────────────────────────
		["trees.count"] = Slider("Tree Count", 200, 1, 2000),
		["trees.spread"] = Slider("Spread Radius", 500, 10, 5000),
		["trees.scale_min"] = Slider("Scale Min", 5.8, 0.01, 100.0),
		["trees.scale_max"] = Slider("Scale Max", 9.0, 0.01, 100.0),
		["trees.seed"] = Slider("Random Seed", 42, 0, 9999),
		["trees.btn.spawn"] = Button("Spawn Trees"),
		["trees.btn.despawn"] = Button("Clear Trees"),
		["trees.status"] = Label("Status", "idle"),

		-- ── Day / Night Cycle ────────────────────────────────────────
		["dnc.enabled"] = Slider("Cycle Running (0=off 1=on)", 1.0, 0.0, 1.0),
		["dnc.speed"] = Slider("Cycle Speed", 0.002, 0.0, 5.0),
		["dnc.time_of_day"] = Slider("Time of Day (0=midnight  0.5=noon)", 0.3, 0.0, 1.0),
		["dnc.peak_brightness"] = Slider("Peak Sun Brightness", 1.5, 0.0, 5.0),
		["dnc.ambient_min"] = Slider("Night Ambient Level", 0.05, 0.0, 0.5),
		["dnc.clock"] = Label("Clock", "00:00"),

		-- ── Sun & Sky Lighting ───────────────────────────────────────
		-- Sun position (world-space)
		["sun.pos.x"] = Slider("Sun Pos X", 0.0, -1000.0, 1000.0),
		["sun.pos.y"] = Slider("Sun Pos Y", 500.0, -1000.0, 1000.0),
		["sun.pos.z"] = Slider("Sun Pos Z", 0.0, -1000.0, 1000.0),
		-- Sun color (HDR, so > 1 is fine for brightness)
		["sun.color.r"] = Slider("Sun Color R", 1.5, 0.0, 10.0),
		["sun.color.g"] = Slider("Sun Color G", 1.4, 0.0, 10.0),
		["sun.color.b"] = Slider("Sun Color B", 1.0, 0.0, 10.0),
		-- Sky zenith color
		["sky.zenith.r"] = Slider("Zenith R", 0.05, 0.0, 2.0),
		["sky.zenith.g"] = Slider("Zenith G", 0.1, 0.0, 2.0),
		["sky.zenith.b"] = Slider("Zenith B", 0.4, 0.0, 2.0),
		-- Sky horizon color
		["sky.horizon.r"] = Slider("Horizon R", 0.5, 0.0, 2.0),
		["sky.horizon.g"] = Slider("Horizon G", 0.6, 0.0, 2.0),
		["sky.horizon.b"] = Slider("Horizon B", 0.8, 0.0, 2.0),
		-- Sky gradient sharpness
		["sky.gradient_sharpness"] = Slider("Sky Gradient Sharpness", 4.0, 0.1, 20.0),
		-- Read / Apply buttons + status
		["sun.btn.read"] = Button("Read Sun & Sky"),
		["sun.btn.apply"] = Button("Apply Sun & Sky"),
		["sun.status"] = Label("Status", "idle"),

		-- ── Terrain / Heightmap ──────────────────────────────────────
		["terrain.heightmap.res"] = Slider("Resolution", 1000, 5, 5000),

		-- ── Player & Camera ──────────────────────────────────────────
		["player.sprint_speed"] = Slider("Sprint Speed", 20, 5, 200),
		["camera.offset.x"] = Slider("Offset X", 0, -300, 300),
		["camera.offset.y"] = Slider("Offset Y", 0, -100, 300),
		["camera.offset.z"] = Slider("Offset Z", 0, -100, 300),
		["world.gravity"] = Slider("Gravity", 9.8, -10, 30),

		-- ── Entity Transform ─────────────────────────────────────────
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

		-- ── Point Light ──────────────────────────────────────────────
		-- Color channels go up to 20 to support HDR / overbright lights.
		-- brightness, linear, quadratic match PointLight fields exactly.
		["light.name"] = TextInput("Light Entity Name"),
		["light.color.r"] = Slider("Color R", 1.0, 0.0, 20.0),
		["light.color.g"] = Slider("Color G", 1.0, 0.0, 20.0),
		["light.color.b"] = Slider("Color B", 1.0, 0.0, 20.0),
		["light.size"] = Slider("Size", 3.0, 0.0, 20.0),
		["light.btn.read"] = Button("Read Light"),
		["light.btn.apply"] = Button("Apply Light"),
		["light.status"] = Label("Status", "idle"),
		["separator"] = Separator(),
	}

	ui.schema = {
		windows = {
			-- ── Entity Control ───────────────────────────────────────
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

			-- ── Terrain ──────────────────────────────────────────────
			{
				name = "Terrain",
				fields = {
					{ type = "Field", data = "terrain.heightmap.res" },
				},
			},

			-- ── Player & Camera ──────────────────────────────────────
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

			-- ── Point Light Control ──────────────────────────────────
			{
				name = "Point Light Control",
				fields = {
					{ type = "Field", data = "light.name" },
					{ type = "Field", data = "light.color.r" },
					{ type = "Field", data = "light.color.g" },
					{ type = "Field", data = "light.color.b" },
					{ type = "Field", data = "light.size" },
					{ type = "Field", data = "light.btn.read" },
					{ type = "Field", data = "light.btn.apply" },
					{ type = "Field", data = "light.status" },
				},
			},

			-- ── Sun & Sky Lighting ───────────────────────────────────
			{
				name = "Sun & Sky Lighting",
				fields = {
					{ type = "Field", data = "sun.pos.x" },
					{ type = "Field", data = "sun.pos.y" },
					{ type = "Field", data = "sun.pos.z" },
					{ type = "Field", data = "separator" },
					{ type = "Field", data = "sun.color.r" },
					{ type = "Field", data = "sun.color.g" },
					{ type = "Field", data = "sun.color.b" },
					{ type = "Field", data = "separator" },
					{ type = "Field", data = "sky.zenith.r" },
					{ type = "Field", data = "sky.zenith.g" },
					{ type = "Field", data = "sky.zenith.b" },
					{ type = "Field", data = "separator" },
					{ type = "Field", data = "sky.horizon.r" },
					{ type = "Field", data = "sky.horizon.g" },
					{ type = "Field", data = "sky.horizon.b" },
					{ type = "Field", data = "separator" },
					{ type = "Field", data = "sky.gradient_sharpness" },
					{ type = "Field", data = "sun.btn.read" },
					{ type = "Field", data = "sun.btn.apply" },
					{ type = "Field", data = "sun.status" },
				},
			},

			-- ── Day / Night Cycle ────────────────────────────────────
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

			-- ── Tree Spawning ────────────────────────────────────────
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
		local vel = rigidbody.velocity
		local speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) * 0.35
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
	-- SUN & SKY BUTTONS
	-----------------------------------------------------------------

	if button(widgets, "sun.btn.read").clicked then
		read_sun_sky(world, widgets)
	end
	if button(widgets, "sun.btn.apply").clicked then
		apply_sun_sky(world, widgets)
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
