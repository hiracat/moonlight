Time = Time or 0.0
Gem_collected = Gem_collected or false
Platform_entities = Platform_entities or {}
Gem_entity = Gem_entity or nil
Initialized = Initialized or false
local platform_defs = {
	{ x = 3, base_y = 2, z = 3, speed = 1.0, phase = 0.0, amp = 0.8 },
	{ x = 6, base_y = 4, z = 5, speed = 1.3, phase = 1.0, amp = 1.2 },
	{ x = 9, base_y = 5, z = 3, speed = 0.8, phase = 2.5, amp = 1.5 },
	{ x = 12, base_y = 7, z = 1, speed = 1.8, phase = 0.5, amp = 0.6 },
	{ x = 10, base_y = 9, z = -3, speed = 1.1, phase = 3.0, amp = 1.0 },
	{ x = 7, base_y = 11, z = -6, speed = 0.6, phase = 1.5, amp = 1.8 },
	{ x = 4, base_y = 13, z = -4, speed = 2.0, phase = 0.0, amp = 0.5 },
	{ x = 1, base_y = 15, z = -2, speed = 1.0, phase = 4.0, amp = 1.0 },
}

function Update(world, engine)
	if not Initialized then
		Initialized = true

		local cube_mesh, _ = engine:load_gltf_asset("data/models/large_cube.glb")
		local platform_tex = engine:create_texture("data/models/textures/ground.jpg")
		for i, def in ipairs(platform_defs) do
			local e = world:spawn({
				Transform = { x = def.x, y = def.base_y, z = def.z, sx = 2.5, sy = 0.3, sz = 2.5 },
				Mesh = cube_mesh,
				Material = { albedo = platform_tex },
				Collider = { extent_x = 2, extent_y = 2, extent_z = 2, offset_x = 0, offset_y = 0, offset_z = 0 },
			})
			Platform_entities[i] = e
		end

		local gem_mesh, _ = engine:load_gltf_asset("data/models/diamond.glb")
		local gem_tex = engine:create_texture("data/models/textures/diamond.png")
		Gem_entity = world:spawn({
			Transform = { x = 1, y = 17, z = -2, sx = 0.5, sy = 0.5, sz = 0.5 },
			Mesh = gem_mesh,
			Material = { albedo = gem_tex },
		})
	end

	local dt = world:get_resource("Time").delta_time
	Time = Time + dt

	-- bob platforms
	for i, def in ipairs(platform_defs) do
		local e = Platform_entities[i]
		if e then
			local t = world:get_component(e, "Transform")
			t.y = def.base_y + math.sin(Time * def.speed + def.phase) * def.amp
		end
	end

	-- spin gem and check collection
	if not Gem_collected and Gem_entity then
		local gt = world:get_component(Gem_entity, "Transform")
		gt:set_rotation_xz(Time * 2.0)

		local results = world:query({ req = { "Transform", "Controllable" } })
		local fox_x, fox_y, fox_z = 0, 0, 0
		for _, row in ipairs(results) do
			fox_x = row.Transform.x
			fox_y = row.Transform.y
			fox_z = row.Transform.z
		end
		world:end_query()

		local dx = gt.x - fox_x
		local dy = gt.y - fox_y
		local dz = gt.z - fox_z
		local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
		if dist < 1.5 then
			Gem_collected = true
			gt.sx = 0
			gt.sy = 0
			gt.sz = 0
		end
	end
end
