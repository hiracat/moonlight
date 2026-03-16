local Run = false
local cube_mesh = nil

function Update(world, engine)
	if not Run then
		Run = true
		local mesh, _ = engine:load_gltf_asset("data/models/large_cube.glb")
		for i = 1, 5 do
			print(mesh)
			world:spawn({
				Transform = { x = i * 3.0, y = i * 2.0, z = 0.0 },
				Mesh = mesh,
			})
		end
	end
end
