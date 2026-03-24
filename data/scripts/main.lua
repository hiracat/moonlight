Index = Index or 0
local ui_initialized = false

function Update(world, engine)
	local results = world:query({ req = { "Controllable", "Animated", "Transform", "RigidBody" } })
	local light = world:query({ req = { "PointLight", "Transform" } })

	for _, p in ipairs(results) do
		local rigidbody = p.RigidBody ---@type RigidBody
		local velocity = rigidbody.velocity
		local animated = p.Animated ---@type Animated
		local control = p.Controllable ---@type Controllable
		if velocity.x == 0.0 and velocity.z == 0.0 then
			animated.current_playing = nil
		else
			animated.current_playing = animated.animations[1]
		end
		Index = Index + 1
		local transform = p.Transform ---@type Transform

		local pos = transform.position
		transform.position = pos

		for _, u in ipairs(light) do
			local ui = world:get_resource("UIStuff") ---@type UIStuff
			local terrainmap = world:get_resource("TerrainMap") ---@type TerrainMap
			if not ui_initialized then
				ui.sliders = {
					{ label = "Red", value = 1.0, min = 0.0, max = 1.0 },
					{ label = "Green", value = 0.8, min = 0.0, max = 1.0 },
					{ label = "Blue", value = 0.4, min = 0.0, max = 1.0 },
					{ label = "Brightness", value = 3.0, min = 0.0, max = 20.0 },
					{ label = "Linear", value = 0.22, min = 0.0, max = 2.0 },
					{ label = "Quadratic", value = 0.20, min = 0.0, max = 5.0 },
					{ label = "HeightMap Resolution", value = 1000, min = 5, max = 5000 },
					{ label = "sprint speed", value = 20, min = 5, max = 200 },

					{ label = "Cam side to side", value = 0.0, min = -300, max = 300 },
					{ label = "cam up down", value = 0.0, min = -100, max = 300 },
					{ label = "cam back", value = 0.0, min = -100, max = 300 },
				}
				ui_initialized = true
			end

			local r = ui.sliders[1].value
			local g = ui.sliders[2].value
			local b = ui.sliders[3].value
			local brightness = ui.sliders[4].value
			local linear = ui.sliders[5].value
			local quadratic = ui.sliders[6].value
			local heightmap_res = ui.sliders[7].value
			terrainmap.resolution = heightmap_res
			control.sprint_speed = ui.sliders[8].value

			local offset = world:get_resource("CameraOffset") ---@type CameraOffset
			offset.offset = { x = ui.sliders[9].value, y = ui.sliders[10].value, z = ui.sliders[11].value }

			local tmp_light = u.PointLight ---@type PointLight
			tmp_light.brightness = brightness
			tmp_light.linear = linear
			tmp_light.quadratic = quadratic
			local color = tmp_light.color
			color.x = r
			color.y = g
			color.z = b
			tmp_light.color = color

			local light_transform = u.Transform ---@type Transform
			local new_pos = transform.position
			new_pos.y = new_pos.y + 0.2
			light_transform.position = new_pos
		end
	end
	results = world:query({ req = { "Collider", "Transform" }, ["not"] = { "Controllable" } })

	for _, p in ipairs(results) do
		Index = Index + 1
		local collider = p.Collider ---@type Collider
		local transform = p.Transform ---@type Transform

		local pos = transform.position
		pos.y = 144
		transform.position = pos
	end
end
