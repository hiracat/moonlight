# A toy Vulkan renderer,
uses a custom basic ecs for rendering, look at game/lib.rs, it shows basics.
the only components that matter to the renderer are model, point light, directioal light, ambientlight, and the camera resource.
there is no limit on the number of anything, i dont know how well it would handle 1500 point lights tho, as each is its own draw call :(

there is a built in physics, but only axis alligned bounding boxes are supprted, as well as ray box intersection tests
