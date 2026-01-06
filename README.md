# Moonlight(Name is going to change eventually)

A custom Vulkan renderer written in Rust, featuring a homemade Entity Component System and physics engine. Built from scratch to understand graphics programming

## Running
first download the whole project folder

to build from scratch: clone repo and from the projects root just run ``cargo run`` or ``cargo run --release``, depends on a working vulkan driver

a full example with prebuilt executables is available at this link: https://drive.proton.me/urls/PYJJB301MM#Ar6WVVxiseo3
just run corruption or corruption.exe depending on platform

### windows
windows requies the visual c++ runtime, which can be found here: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version

## Overview
Moonlight is my project to understand graphics programming. It is a deferred renderer with a simple physics engine and egui integration.

## Features

### Rendering
- **Deferred rendering pipeline** with geometry pass and lighting pass
- **Material system** with texture support (albedo, with plans for more PBR properties)
- **Multiple light types**:
  - Point lights (unlimited count, though performance degrades with many lights)
  - One global directional light
  - One global property to set the minimum lighting for anything
- **Skybox rendering**
- **Third-person camera** That im incredibly proud of, the controls feel very nice
- **UI integration** via egui

### Core Systems
- **Custom ECS** with efficient querying and component storage
- **Resource Management**:
  - Resource system for gpu state, hopefully to allow for serialization in the future
  - GLTF model loading support
- **Physics Engine**:
  - Axis-aligned bounding box collision detection
  - Euler integration

### Technical Highlights
- **Shader reflection** using `rspriv-reflect` should make writing shaders require less effort
- **Modular architecture** with separate modules for rendering, physics, ECS, and game logic

## Architecture

The renderer uses a **deferred rendering** approach:
1. **Geometry Pass**: Renders scene geometry to G-buffers (position, normal, albedo)
2. **Lighting Pass**: Calculates lighting using G-buffer data and outputs to screen

The ECS stores only CPU-side data with handles/IDs to GPU resources. Each frame:
- Components are queried via closures per shader
- The closures gather the necessary data from the Resource manager to make draw jobs
- Draw jobs are generated and executed

### Key Components
**Rendering-relevant ECS components:**
- `Mesh` - Reference to mesh geometry via handle
- `Material` - Currently contains albedo texture handle
- `Transform` - Position, rotation, scale
- `PointLight` - Position and color

**Resources:**
- `Camera` - View and projection matrices
- `DirectionalLight` - Direction and color
- `AmbientLight` - Global ambient color
- `Keyboard` - All keys pressed this frame


## Using this engine(if for some insane reason you like it)
- See game/lib.rs for an example, its relatively simple

## Known Limitations

- Physics only supports axis-aligned bounding boxes (no arbitrary collision shapes)
- Each point light requires its own draw call (performance degrades with many lights)
- Materials currently only support albedo textures (PBR pipeline incomplete)
- No frustum culling or other rendering optimizations yet

## Future Plans
1. Animation support
2. shadow mapping
3. Scene serialization/deserialization
4. terrain

## Building

Requires Rust 1.90.0 or later and a Vulkan-compatible GPU with updated drivers.

```bash
cargo build --release
cargo run --release
```
