function Update(world, dt)
	local results = world:query({ req = { "Transform" }, opt = { "RigidBody" } })
	print("results type:", type(results))
	print("results length:", #results)

	for _, row in ipairs(results) do
		local t = row.Transform.z
		print(t)
		row.Transform.y = 0
		local t = row.Transform.z
		print(t)
		print(row.RigidBody)
	end
end
