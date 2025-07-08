#![allow(clippy::cast_possible_truncation)]
#![allow(dead_code)]
use ash::vk::{self, Extent3D};
use bytemuck::{bytes_of, Pod, Zeroable};
use gpu_allocator::vulkan::*;
use half::f16;
use std::{
    ffi::CStr,
    fs,
    io::Write,
    marker::PhantomData,
    os::raw::c_void,
    ptr::{self},
    slice,
    sync::Arc,
    time::Instant,
};
use winit::dpi::PhysicalSize;

const VALIDATION_ENABLE: bool = true;

use ultraviolet::{projection, Rotor3, Vec3};
//INFO: idk how to fix this, one wants raw window handles and the other says no
#[allow(deprecated)]
use winit::{
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

pub type SubpassIndex = u32;
pub const GEOMETRY_SUBPASS: SubpassIndex = 0;
pub const LIGHTING_SUBPASS: SubpassIndex = 1;

#[derive(Default)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Rotor3,
    pub fov_rads: f32,
    pub near: f32,
    pub far: f32,
    pub pitch: f32,
    pub yaw: f32,

    aspect_ratio: f32,

    pub(crate) u_buffer: Option<Vec<vk::Buffer>>,
    // write to this to upload values, only need to update desriptors to point at this once
    pub(crate) allocations: Option<Vec<Allocation>>,

    pub(crate) descriptor_set: Option<Vec<vk::DescriptorSet>>,
}

impl Camera {
    pub fn new(position: Vec3, fov: f32, near: f32, far: f32, window: &Arc<Window>) -> Self {
        let size: [f32; 2] = window.inner_size().into();
        let aspect_ratio = size[0] / size[1];
        let fov_rads = fov * (std::f32::consts::PI / 180.0);
        let rotation = Rotor3::identity();

        Self {
            pitch: 0.0,
            yaw: 0.0,
            position,
            rotation,
            fov_rads,
            near,
            far,
            aspect_ratio,

            u_buffer: None,
            descriptor_set: None,
            allocations: None,
        }
    }
    fn get_ubo_data(&self) -> CameraUBO {
        let forward = (-Vec3::unit_z()).rotated_by(self.rotation);
        let look_at = self.position + forward;
        CameraUBO {
            view: ultraviolet::Mat4::look_at(
                self.position,
                look_at,
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            ),
            proj: projection::perspective_vk(self.fov_rads, self.aspect_ratio, self.near, self.far),
        }
    }
    fn populate_u_buffer(&mut self, device: &ash::Device, memory_allocator: &mut Allocator) {
        let mut camera_buffers = vec![];
        let mut camera_allocations = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            let buffer = unsafe {
                device.create_buffer(
                    &vk::BufferCreateInfo {
                        size: size_of::<CameraUBO>() as u64,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap();
            let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

            let mut alloc = memory_allocator
                .allocate(&AllocationCreateDesc {
                    name: "ambient buffer",
                    requirements,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap();
            //HACK: needs wrapper, allocations is just to keep alloc from dropping

            alloc
                .mapped_slice_mut()
                .unwrap()
                .write_all(bytes_of(&self.get_ubo_data()));
            unsafe {
                device
                    .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                    .unwrap()
            }

            camera_allocations.push(alloc);
            camera_buffers.push(buffer);
        }
        dbg!(&camera_buffers);
        self.u_buffer = Some(camera_buffers);
        self.allocations = Some(camera_allocations);
    }
}

pub fn alloc_buffer(
    memory_allocator: &mut Allocator,
    buffer_count: usize,
    size: u64,
    device: &ash::Device,
    sharing: vk::SharingMode,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
    linear: bool,
    data: &[u8],
) -> (Vec<vk::Buffer>, Vec<Allocation>) {
    let mut buffers = vec![];
    let mut allocations = vec![];
    for _ in 0..buffer_count {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    sharing_mode: sharing,
                    usage,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut alloc = memory_allocator
            .allocate(&AllocationCreateDesc {
                name: "ambient buffer",
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        //HACK: needs wrapper, allocations is just to keep alloc from dropping

        alloc.mapped_slice_mut().unwrap().write_all(data);
        unsafe {
            device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                .unwrap()
        }

        allocations.push(alloc);
        buffers.push(buffer);
    }
    dbg!(&buffers);
    (buffers, allocations)
}

#[derive(Default, Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct CameraUBO {
    view: ultraviolet::Mat4,
    proj: ultraviolet::Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f16; 3],
    _padding: [u8; 2],
}

impl Vertex {
    pub fn new(postion: Vec3, normal: Vec3, color: Vec3) -> Self {
        let color: [f16; 3] = [
            f16::from_f32(color.x),
            f16::from_f32(color.y),
            f16::from_f32(color.z),
        ];
        Self {
            position: *postion.as_array(),
            normal: *normal.as_array(),
            color,
            _padding: [0, 0],
        }
    }
}

pub const FRAMES_IN_FLIGHT: usize = 3;

#[repr(C)]
pub struct DummyVertex {
    pub position: [f32; 2],
} // not necessary, defined in shader
impl DummyVertex {
    pub const fn screen_quad() -> [DummyVertex; 6] {
        [
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [-1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, -1.0],
            },
        ]
    }
}

pub struct Renderer {
    pub window: Arc<Window>,
    pub recreate_swapchain: bool,
    pub memory_allocator: Allocator,
    pub start_time: Instant,
    pub pipelines: Vec<vk::Pipeline>,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) command_pool: vk::CommandPool,

    pub(crate) queue: vk::Queue,
    pub(crate) queue_family_index: QueueFamilyIndex,

    surface: vk::SurfaceKHR,
    aspect_ratio: f32,
    instance: ash::Instance,
    pub(crate) descriptor_layouts: DescriptorLayouts,

    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) surface_loader: ash::khr::surface::Instance,
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) debug_utils_loader: ash::ext::debug_utils::Instance,
    pub(crate) swapchain_image_format: vk::SurfaceFormatKHR,
    pub(crate) pipeline_layouts: Vec<vk::PipelineLayout>,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    frame_buffer_allocations: Vec<Allocation>,
    uniform_buffer_allocations: Vec<Allocation>,

    // wait before writing to swapchain image
    image_available_semaphore: Vec<vk::Semaphore>,
    // wait before presenting
    render_finished_semaphore: Vec<vk::Semaphore>,
    // wait for the frame that was FRAMES_IN_FLIGHT frames ago, but has the same current_frame
    // since modulo
    in_flight_fence: Vec<vk::Fence>,
    // in case it is still in use
    swapchain_image_still_in_use: Vec<Option<vk::Fence>>,

    current_frame: usize,
    frame_counter: u64,

    pub(crate) swapchain: vk::SwapchainKHR,
    render_pass: vk::RenderPass,

    viewport: vk::Viewport,

    framebuffers: Vec<vk::Framebuffer>,

    pub(crate) color_buffers: Color,
    pub(crate) normal_buffers: Normal,
    pub(crate) position_buffers: Position,

    swapchain_image_index: usize,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION_ENABLE {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl Renderer {
    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, world: &mut World) {
        self.window.request_redraw();
        let window_size = self.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence[self.current_frame]], true, 0);
            self.device
                .reset_fences(&[self.in_flight_fence[self.current_frame]]);
        };

        self.frame_counter += 1;
        dbg!(self.frame_counter);

        if self.recreate_swapchain {
            // Use the new dimensions of the window.
            dbg!(self.recreate_swapchain);
            let new_images;
            (self.swapchain, new_images) = create_swapchain(
                &self.surface_loader,
                &self.swapchain_loader,
                self.physical_device,
                Some(self.swapchain),
                &self.window,
                self.surface,
                &self.swapchain_image_format,
                &[self.queue_family_index],
            );

            for ele in self.frame_buffer_allocations.drain(..) {
                self.memory_allocator.free(ele).unwrap()
            }

            // Because framebuffers contains a reference to the old swapchain, we need to
            // recreate framebuffers as well.
            (
                self.framebuffers,
                self.color_buffers,
                self.normal_buffers,
                self.position_buffers,
                self.frame_buffer_allocations,
            ) = create_framebuffers(
                &self.device,
                &new_images,
                self.render_pass,
                &mut self.memory_allocator,
                window_size,
                self.swapchain_image_format.format,
            );

            self.viewport = vk::Viewport {
                width: self.window.inner_size().width as f32,
                height: self.window.inner_size().height as f32,
                ..Default::default()
            };
            self.aspect_ratio = self.viewport.width / self.viewport.height;

            let camera = world
                .resource_get_mut::<Camera>()
                .expect("camera should definately exist or something is very wrong");
            camera.aspect_ratio = self.aspect_ratio;
            camera.populate_u_buffer(&self.device, &mut self.memory_allocator);

            write_descriptor_sets(
                world,
                &self.device,
                self.descriptor_pool,
                &self.descriptor_layouts,
                &self.color_buffers,
                &self.normal_buffers,
                &self.position_buffers,
                new_images.len(),
            );

            self.recreate_swapchain = false;
        }

        // NOTE: RENDERING START
        let is_suboptimal;
        (self.swapchain_image_index, is_suboptimal) = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                2 * 1_000_000_000,
                self.image_available_semaphore[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok((index, suboptimal)) => (index as usize, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };

        if let Some(previous_fence) = self.swapchain_image_still_in_use[self.swapchain_image_index]
        {
            if previous_fence != vk::Fence::null() {
                unsafe {
                    self.device
                        .wait_for_fences(&[previous_fence], true, u64::MAX)
                        .unwrap();
                }
            }
        }
        self.swapchain_image_still_in_use[self.swapchain_image_index] =
            Some(self.in_flight_fence[self.current_frame]);

        if is_suboptimal {
            self.recreate_swapchain = true;
        }

        dbg!(self.current_frame);

        {
            let camera = world
                .resource_get_mut::<Camera>()
                .expect("should have a camera resource");

            let data = camera.get_ubo_data();
            let data = bytes_of(&data);
            let mem = camera.allocations.as_mut().unwrap();
            for _ in 0..5 {
                match mem[self.current_frame].mapped_slice_mut() {
                    Some(mut write) => {
                        write.write(data);
                        break;
                    }
                    None => {
                        eprintln!("camera memory not accessable");
                    }
                }
            }
        }

        {
            let entities = world.query_entities::<PointLight>();
            for entity in entities {
                let transform;
                if !world.has_component::<Transform>(entity) {
                    eprintln!(
                        "WARNING: light {:?} does not have transform component, giving identity matrix", entity
                    );
                    transform = Transform::new();
                    world.component_add(entity, transform);
                } else {
                    transform = *world
                        .component_get::<Transform>(entity)
                        .expect("ecs corrputed, panic");
                }
                let point_light = world
                    .component_get_mut::<PointLight>(entity)
                    .expect("ecs corrupted, panic");

                let t = point_light.as_point_ubo(transform);
                let data = bytes_of(&t);
                let mem = point_light.allocations.as_mut().unwrap();

                for _ in 0..5 {
                    match mem[self.current_frame].mapped_slice_mut() {
                        Some(mut write) => {
                            let _ = write.write(data);
                            break;
                        }
                        None => {
                            eprintln!("point light {:?} memory is not accessable", entity)
                        }
                    }
                }
            }
        }

        {
            let entities = world.query_entities::<Model>();
            for entity in entities {
                let mut transform;
                if !world.has_component::<Transform>(entity) {
                    eprintln!(
                        "WARNING: model {:?} does not have transform component, giving identity matrix", entity
                    );
                    transform = Transform::new();
                    let _ = world.component_add(entity, transform);
                } else {
                    transform = *world
                        .component_get::<Transform>(entity)
                        .expect("ecs corrupted, panic");
                }
                let model = world
                    .component_get_mut::<Model>(entity)
                    .expect("ecs corrputed, panic");

                let t = &transform.as_model_ubo();
                let data = bytes_of(t);
                let mem = model.allocations.as_mut().unwrap();
                for _ in 0..5 {
                    match mem[self.current_frame].mapped_slice_mut() {
                        Some(mut write) => {
                            write.write(data);
                            break;
                        }
                        None => {
                            eprintln!("model {:?} memory is not accessable", entity)
                        }
                    }
                }
            }
        }

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        let command_buffer = unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()[0]
        };
        let clear_values = vec![
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: f32::MAX,
                    stencil: 0,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        ];
        unsafe {
            self.device.cmd_begin_render_pass2(
                command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[self.swapchain_image_index],
                    p_clear_values: clear_values.as_ptr(),
                    clear_value_count: clear_values.len() as u32,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: window_size.width,
                            height: window_size.height,
                        },
                    },
                    ..Default::default()
                },
                &vk::SubpassBeginInfo {
                    contents: vk::SubpassContents::INLINE,
                    ..Default::default()
                },
            );
            self.device
                .cmd_set_viewport(command_buffer, 0, &[self.viewport]);
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[0],
            );
        };

        let camera_descripor_set = world
            .resource_get::<Camera>()
            .unwrap()
            .descriptor_set
            .as_ref()
            .unwrap()
            .clone();

        let entities = world.query_mut::<Model>();

        for (_entity, model) in entities {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layouts[0],
                    0,
                    &[
                        camera_descripor_set[self.current_frame],
                        model.descriptor_set.as_ref().unwrap()[self.current_frame],
                    ],
                    &[],
                );
                self.device
                    .cmd_bind_vertex_buffers(command_buffer, 0, &[model.vertex_buffer], &[]);
                self.device.cmd_bind_index_buffer(
                    command_buffer,
                    model.index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                self.device.cmd_draw_indexed(
                    command_buffer,
                    model.index_buffer_len as u32,
                    1,
                    0,
                    0,
                    0,
                );
            };
        }

        unsafe {
            self.device
                .cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[1],
            );
            let descriptor_set = world
                .resource_get::<DirectionalLight>()
                .unwrap()
                .descriptor_set
                .as_ref()
                .unwrap()[self.current_frame];
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layouts[1],
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_draw(command_buffer, 6, 1, 0, 0);
        };

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[2],
            );
            let descriptor_set = world
                .resource_get::<AmbientLight>()
                .unwrap()
                .descriptor_set
                .as_ref()
                .unwrap()[self.current_frame];
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layouts[2],
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_draw(command_buffer, 6, 1, 0, 0);
        }

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[3],
            );
            let entities = world.query_mut::<PointLight>();
            for (_entity, light) in entities {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layouts[3],
                    0,
                    &[light.descriptor_set[self.current_frame]],
                    &[],
                );
                self.device.cmd_draw(command_buffer, 6, 1, 0, 0);
            }
        }
        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
        }

        let queue_submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.image_available_semaphore[self.current_frame],
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: [command_buffer].as_ptr(),
            p_signal_semaphores: &self.render_finished_semaphore[self.current_frame],
            signal_semaphore_count: 1,
            ..Default::default()
        };

        unsafe {
            self.device.queue_submit(
                self.queue,
                &[queue_submit_info],
                self.in_flight_fence[self.current_frame],
            );
        }

        let mut present_results: [vk::Result; 1] = [Default::default(); 1];
        let present_info = vk::PresentInfoKHR {
            p_wait_semaphores: &self.render_finished_semaphore[self.current_frame],
            wait_semaphore_count: 1,
            p_swapchains: &self.swapchain,
            swapchain_count: 1,
            p_image_indices: &(self.swapchain_image_index as u32),
            p_results: present_results.as_mut_ptr(),
            ..Default::default()
        };

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info);
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(event_loop: &ActiveEventLoop, world: &mut World, window: &Arc<Window>) -> Self {
        let start_time = Instant::now();
        let entry = create_entry();
        eprintln!("created entry");
        let instance = create_instance(&entry, event_loop);
        eprintln!("created instance");
        let surface = unsafe {
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                // idk what to do about this, i need the raw handle
                event_loop.raw_display_handle().unwrap(),
                window.raw_window_handle().unwrap(),
                None,
            )
            .unwrap();
            surface
        };
        eprintln!("created surface");
        //need to make this optional to put stuff inside
        if VALIDATION_ENABLE {}
        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = setup_debug_utils(&debug_utils_loader);
        eprintln!("set up debug utility");

        let required_extensions = vec![ash::vk::KHR_SWAPCHAIN_NAME];
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, surface, &required_extensions);
        let (device, queue) = create_device(
            &instance,
            physical_device,
            queue_family_index,
            &required_extensions,
        );

        let window_size = window.inner_size();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;

        let mut memory_allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let swapchain_image_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images) = create_swapchain(
            &surface_loader,
            &swapchain_loader,
            physical_device,
            None,
            window,
            surface,
            &swapchain_image_format,
            &[queue_family_index],
        );

        let render_pass = create_renderpass(&device, swapchain_image_format.format);

        let (
            framebuffers,
            color_buffers,
            normal_buffers,
            position_buffers,
            frame_buffer_allocations,
        ) = create_framebuffers(
            &device,
            &swapchain_images,
            render_pass,
            &mut memory_allocator,
            window_size,
            swapchain_image_format.format,
        );
        // used to keep uniform buffer memory alive
        //
        let mut uniform_buffer_allocations = vec![];
        let mut directional_buffers = vec![];
        if let Some(directional) = world.resource_get_mut::<DirectionalLight>() {
            for _ in 0..swapchain_images.len() {
                let buffer = unsafe {
                    device.create_buffer(
                        &vk::BufferCreateInfo {
                            size: size_of::<DirectionalLightUBO>() as u64,
                            sharing_mode: vk::SharingMode::EXCLUSIVE,
                            usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_DST,
                            ..Default::default()
                        },
                        None,
                    )
                }
                .unwrap();
                let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

                let mut alloc = memory_allocator
                    .allocate(&AllocationCreateDesc {
                        name: "directional buffer",
                        requirements,
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();

                alloc
                    .mapped_slice_mut()
                    .unwrap()
                    .write_all(bytes_of(&DirectionalLightUBO {
                        color: directional.color,
                        position: directional.position,
                    }));
                unsafe {
                    device
                        .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                        .unwrap()
                }
                //HACK: needs wrapper, allocations is just to keep alloc from dropping
                uniform_buffer_allocations.push(alloc);

                directional_buffers.push(buffer);
            }
            dbg!(&directional_buffers);
            directional.u_buffer = Some(directional_buffers);
        } else {
            eprintln!("no  directional resource, skipping");
        }

        let mut ambient_allocations = vec![];
        let mut ambient_buffers = vec![];
        if let Some(ambient) = world.resource_get_mut::<AmbientLight>() {
            for _ in 0..swapchain_images.len() {
                let buffer = unsafe {
                    device.create_buffer(
                        &vk::BufferCreateInfo {
                            size: size_of::<AmbientLightUBO>() as u64,
                            sharing_mode: vk::SharingMode::EXCLUSIVE,
                            usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_DST,
                            ..Default::default()
                        },
                        None,
                    )
                }
                .unwrap();
                let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

                let mut alloc = memory_allocator
                    .allocate(&AllocationCreateDesc {
                        name: "ambient buffer",
                        requirements,
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                //HACK: needs wrapper, allocations is just to keep alloc from dropping

                alloc
                    .mapped_slice_mut()
                    .unwrap()
                    .write_all(bytes_of(&AmbientLightUBO {
                        color: ambient.color,
                        intensity: ambient.intensity,
                    }));
                unsafe {
                    device
                        .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                        .unwrap()
                }

                dbg!(&ambient_buffers);
                ambient_allocations.push(alloc);
                ambient_buffers.push(buffer);
            }
            ambient.u_buffer = Some(ambient_buffers);
            ambient.allocations = Some(ambient_allocations);
        } else {
            eprintln!("no  directional resource, skipping");
        }

        let camera = world
            .resource_get_mut::<Camera>()
            .expect("should create camera resource");
        camera.populate_u_buffer(&device, &mut memory_allocator);

        let descriptor_layouts = DescriptorLayouts::init(&device);
        let (pipelines, pipeline_layouts) = create_graphics_pipelines(
            &device,
            render_pass,
            GEOMETRY_SUBPASS,
            LIGHTING_SUBPASS,
            &descriptor_layouts,
        );

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size.width as f32,
            height: window_size.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let mut frames_resources_free = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo {
                        flags: vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap();
            frames_resources_free.push(fence);
        }

        // 15 is my estimate for how many sets per frame, based on 4 pipelines, and varying numbers
        // per pipeline, 2 is the safety margin
        //
        let required_sets: u32 = (15.0 * FRAMES_IN_FLIGHT as f32 * 1.5) as u32;

        let pool_sizes = vec![
            vk::DescriptorPoolSize {
                descriptor_count: 4 * required_sets,
                ty: vk::DescriptorType::INPUT_ATTACHMENT,
            },
            vk::DescriptorPoolSize {
                descriptor_count: 3 * required_sets,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
            },
        ];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: required_sets,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        pool_size_count: pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        write_descriptor_sets(
            world,
            &device,
            descriptor_pool,
            &descriptor_layouts,
            &color_buffers,
            &normal_buffers,
            &position_buffers,
            swapchain_images.len(),
        );

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            ..Default::default()
        };
        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        let mut image_available_semaphore = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphore.push(
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .expect("failed to create semaphore"),
                );
            }
        }
        let mut render_finished_semaphore = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            unsafe {
                render_finished_semaphore.push(
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .expect("failed to create semaphore"),
                );
            }
        }

        let mut swapchain_image_still_in_use = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            swapchain_image_still_in_use.push(None);
        }

        Renderer {
            swapchain_image_still_in_use,
            render_finished_semaphore,
            image_available_semaphore,
            debug_messenger,
            debug_utils_loader,
            instance,
            descriptor_layouts,
            recreate_swapchain: false,
            current_frame: 0,
            frame_counter: 0,
            swapchain,
            framebuffers,
            pipeline_layouts,
            in_flight_fence: frames_resources_free,
            frame_buffer_allocations,
            swapchain_image_format,
            render_pass,
            uniform_buffer_allocations,
            memory_allocator,
            viewport,
            surface,
            surface_loader,
            swapchain_loader,
            command_pool,
            device,
            queue,
            start_time,
            window: window.clone(),
            aspect_ratio,
            descriptor_pool,
            queue_family_index,
            physical_device,

            color_buffers,
            normal_buffers,
            position_buffers,
            pipelines,
            swapchain_image_index: 0,
        }
    }
}

fn create_swapchain(
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    previous_swapchain: Option<vk::SwapchainKHR>,
    window: &Arc<Window>,
    surface: vk::SurfaceKHR,
    image_format: &vk::SurfaceFormatKHR,
    queue_family_indices: &[QueueFamilyIndex],
) -> (vk::SwapchainKHR, Vec<vk::Image>) {
    let window_size: [u32; 2] = window.inner_size().into();
    let window_size = vk::Extent2D {
        width: window_size[0],
        height: window_size[1],
    };

    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .unwrap();

    let image_count = if capabilities.min_image_count == capabilities.max_image_count {
        capabilities.max_image_array_layers
    } else {
        capabilities.min_image_count + 1
    };

    let swapchain_create_info = vk::SwapchainCreateInfoKHR {
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        surface,
        min_image_count: image_count,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        image_format: image_format.format,
        image_extent: window_size,
        present_mode: vk::PresentModeKHR::FIFO,
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        old_swapchain: previous_swapchain.unwrap_or(vk::SwapchainKHR::null()),
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        clipped: vk::TRUE,
        image_array_layers: 1,
        p_queue_family_indices: queue_family_indices.as_ptr(),
        queue_family_index_count: queue_family_indices.len() as u32,

        ..Default::default()
    };

    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };
    let swapchain_images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Failed to get Swapchain Images.")
    };
    (swapchain, swapchain_images)
}

#[derive(Debug)]
pub(crate) struct DescriptorLayouts {
    pub(crate) model: vk::DescriptorSetLayout,
    pub(crate) camera: vk::DescriptorSetLayout,
    pub(crate) directional_light: vk::DescriptorSetLayout,
    pub(crate) ambient_light: vk::DescriptorSetLayout,
    pub(crate) point_light: vk::DescriptorSetLayout,
}

impl DescriptorLayouts {
    fn init(device: &ash::Device) -> Self {
        let model_camera_bindings = vec![vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        }];

        let camera_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: model_camera_bindings.as_ptr(),
            binding_count: model_camera_bindings.len() as u32,
            ..Default::default()
        };
        let model_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: model_camera_bindings.as_ptr(),
            binding_count: model_camera_bindings.len() as u32,
            ..Default::default()
        };

        let ambient_bindings = vec![
            //color attachment
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // normals
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            //ambient light data
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let ambient_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: ambient_bindings.as_ptr(),
            binding_count: ambient_bindings.len() as u32,
            ..Default::default()
        };
        let point_layout = vec![
            //color attachment
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // normals
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // position
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            //point light data
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let point_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: point_layout.as_ptr(),
            binding_count: point_layout.len() as u32,
            ..Default::default()
        };
        let directional_bindings = vec![
            //color attachment
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // normals
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            //directional light data
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let directional_layout = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: directional_bindings.as_ptr(),
            binding_count: directional_bindings.len() as u32,
            ..Default::default()
        };

        let point = unsafe {
            device
                .create_descriptor_set_layout(&point_layout, None)
                .unwrap()
        };
        let directional = unsafe {
            device
                .create_descriptor_set_layout(&directional_layout, None)
                .unwrap()
        };
        let ambient = unsafe {
            device
                .create_descriptor_set_layout(&ambient_layout, None)
                .unwrap()
        };
        let model = unsafe {
            device
                .create_descriptor_set_layout(&model_layout, None)
                .unwrap()
        };
        let camera = unsafe {
            device
                .create_descriptor_set_layout(&camera_layout, None)
                .unwrap()
        };

        DescriptorLayouts {
            model,
            camera,
            point_light: point,
            directional_light: directional,
            ambient_light: ambient,
        }
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn write_descriptor_sets(
    world: &mut World,

    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layouts: &DescriptorLayouts,

    color_buffer: &[vk::ImageView],
    normal_buffer: &[vk::ImageView],
    position_buffer: &[vk::ImageView],

    swapchain_image_count: usize,
) {
    let models = world.query_mut::<Model>();
    for (_entity, model) in models {
        let mut model_sets = vec![];
        for i in 0..FRAMES_IN_FLIGHT {
            let model_set = unsafe {
                device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool,
                        p_set_layouts: [descriptor_layouts.model].as_ptr(),
                        descriptor_set_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
            }[0];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: model_set,
                        p_buffer_info: [vk::DescriptorBufferInfo {
                            buffer: model.u_buffer.as_ref().unwrap()[i],
                            offset: 0,
                            range: size_of::<ModelUBO>() as u64,
                        }]
                        .as_ptr(),
                        descriptor_count: 1,
                        dst_binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        dst_array_element: 0,
                        ..Default::default()
                    }],
                    &[],
                );
            };

            model_sets.push(model_set);
        }
        model.descriptor_set = Some(model_sets);
    }
    eprintln!("where is the error?????");

    if let Some(camera) = world.resource_get_mut::<Camera>() {
        let mut camera_sets = vec![];
        for i in 0..FRAMES_IN_FLIGHT {
            let camera_set = unsafe {
                device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool,
                        p_set_layouts: [descriptor_layouts.camera].as_ptr(),
                        descriptor_set_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
            }[0];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: camera_set,
                        p_buffer_info: [vk::DescriptorBufferInfo {
                            buffer: camera.u_buffer.as_ref().unwrap()[i],
                            offset: 0,
                            range: size_of::<CameraUBO>() as u64,
                        }]
                        .as_ptr(),
                        descriptor_count: 1,
                        dst_binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        dst_array_element: 0,
                        ..Default::default()
                    }],
                    &[],
                );
            };
            camera_sets.push(camera_set);
        }
        camera.descriptor_set = Some(camera_sets);
    }

    let point_lights = world.query_mut::<PointLight>();
    for (_entity, point) in point_lights {
        let mut pt_sets = vec![];
        for i in 0..FRAMES_IN_FLIGHT {
            let pt_set = unsafe {
                device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool,
                        p_set_layouts: [descriptor_layouts.point_light].as_ptr(),
                        descriptor_set_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
            }[0];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet {
                            dst_set: pt_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: normal_buffer[i],
                                image_layout: vk::ImageLayout::UNDEFINED,
                            },
                            descriptor_count: 1,
                            dst_binding: 1,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: pt_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: position_buffer[i],
                                image_layout: vk::ImageLayout::UNDEFINED,
                            },
                            descriptor_count: 1,
                            dst_binding: 2,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: pt_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: color_buffer[i],
                                image_layout: vk::ImageLayout::UNDEFINED,
                            },
                            descriptor_count: 0,
                            dst_binding: 0,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: pt_set,
                            p_buffer_info: [vk::DescriptorBufferInfo {
                                buffer: point.u_buffers.as_ref().unwrap()[i],
                                offset: 0,
                                range: size_of::<PointLightUBO>() as u64,
                            }]
                            .as_ptr(),
                            descriptor_count: 1,
                            dst_binding: 3,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                    ],
                    &[],
                );
            };
            pt_sets.push(pt_set);
        }
        point.descriptor_set = pt_sets;
    }
    if let Some(ambient) = world.resource_get_mut::<AmbientLight>() {
        let mut ambient_sets = Vec::new();
        for i in 0..FRAMES_IN_FLIGHT {
            let ambient_set = unsafe {
                device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool,
                        p_set_layouts: [descriptor_layouts.ambient_light].as_ptr(),
                        descriptor_set_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
            }[0];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet {
                            dst_set: ambient_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                image_layout: vk::ImageLayout::UNDEFINED,
                                image_view: color_buffer[i],
                                ..Default::default()
                            },
                            descriptor_count: 1,
                            dst_binding: 0,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: ambient_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                image_layout: vk::ImageLayout::UNDEFINED,
                                image_view: normal_buffer[i],
                                ..Default::default()
                            },
                            descriptor_count: 1,
                            dst_binding: 1,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: ambient_set,
                            p_buffer_info: [vk::DescriptorBufferInfo {
                                buffer: ambient.u_buffer.as_ref().unwrap()[i],
                                offset: 0,
                                range: size_of::<AmbientLightUBO>() as u64,
                            }]
                            .as_ptr(),
                            descriptor_count: 1,
                            dst_binding: 2,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                    ],
                    &[],
                );
            };

            ambient_sets.push(ambient_set);
        }
        ambient.descriptor_set = Some(ambient_sets);
    }

    let directional_light = world.resource_get_mut::<DirectionalLight>();
    if let Some(dir_light) = directional_light {
        let mut dir_sets = Vec::new();
        for i in 0..FRAMES_IN_FLIGHT {
            let directional_set = unsafe {
                device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool,
                        p_set_layouts: [descriptor_layouts.ambient_light].as_ptr(),
                        descriptor_set_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
            }[0];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet {
                            dst_set: directional_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                image_layout: vk::ImageLayout::UNDEFINED,
                                image_view: color_buffer[i],
                                ..Default::default()
                            },
                            descriptor_count: 1,
                            dst_binding: 0,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: directional_set,
                            p_image_info: &vk::DescriptorImageInfo {
                                image_layout: vk::ImageLayout::UNDEFINED,
                                image_view: normal_buffer[i],
                                ..Default::default()
                            },
                            descriptor_count: 1,
                            dst_binding: 1,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: directional_set,
                            p_buffer_info: [vk::DescriptorBufferInfo {
                                buffer: dir_light.u_buffer.as_ref().unwrap()[i],
                                offset: 0,
                                range: size_of::<DirectionalLightUBO>() as u64,
                            }]
                            .as_ptr(),
                            descriptor_count: 1,
                            dst_binding: 2,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            dst_array_element: 0,
                            ..Default::default()
                        },
                    ],
                    &[],
                );
            }
            dir_sets.push(directional_set);
        }
        dir_light.descriptor_set = Some(dir_sets);
    }
}

type Color = Vec<vk::ImageView>;
type Normal = Vec<vk::ImageView>;
type Position = Vec<vk::ImageView>;
pub fn create_framebuffers(
    device: &ash::Device,
    swapchain_images: &[vk::Image],
    render_pass: vk::RenderPass,
    allocator: &mut Allocator,
    swapchain_image_extent: PhysicalSize<u32>,
    swapchain_image_format: vk::Format,
    // HACK: the vec of allocations is to keep the allocatiosn from dropping and freeing underlying
    // memory, this should be replaced with some kind of wrapper or just one allocation per
    // framebuffer
) -> (
    Vec<vk::Framebuffer>,
    Color,
    Normal,
    Position,
    Vec<Allocation>,
) {
    let mut framebuffers = vec![];
    let mut color_buffers = vec![];
    let mut normal_buffers = vec![];
    let mut position_buffers = vec![];
    let mut allocations = vec![];
    let extent = Extent3D {
        width: swapchain_image_extent.width,
        height: swapchain_image_extent.height,
        depth: 1,
    };

    dbg!(swapchain_images);
    let color_format = vk::Format::A2B10G10R10_UNORM_PACK32;
    let depth_format = vk::Format::D32_SFLOAT;
    let normal_format = vk::Format::R16G16B16A16_SFLOAT;
    let position_format = vk::Format::R32G32B32A32_SFLOAT;
    for i in 0..swapchain_images.len() {
        let depth_create_info = vk::ImageCreateInfo {
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            format: depth_format,
            array_layers: 1,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: ptr::null(), // ignored because sharing mode is exclusive
            queue_family_index_count: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let depth_image = unsafe { device.create_image(&depth_create_info, None).unwrap() };

        let depth_requirements = unsafe { device.get_image_memory_requirements(depth_image) };
        let depth_allocation = allocator
            .allocate(&AllocationCreateDesc {
                requirements: depth_requirements,
                name: "depth image in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device.bind_image_memory(
                depth_image,
                depth_allocation.memory(),
                depth_allocation.offset(),
            )
        };

        let color_create_info = vk::ImageCreateInfo {
            extent: Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            format: color_format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            image_type: vk::ImageType::TYPE_2D,
            p_queue_family_indices: ptr::null(), // ignored because sharing mode is exclusive
            queue_family_index_count: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            array_layers: 1,
            mip_levels: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let color_image = unsafe { device.create_image(&color_create_info, None).unwrap() };

        let color_requirements = unsafe { device.get_image_memory_requirements(color_image) };
        let color_allocation = allocator
            .allocate(&AllocationCreateDesc {
                requirements: color_requirements,
                name: "color image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device.bind_image_memory(
                color_image,
                color_allocation.memory(),
                color_allocation.offset(),
            )
        };

        let normal_create_info = vk::ImageCreateInfo {
            extent,
            format: normal_format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            p_queue_family_indices: ptr::null(), // ignored because sharing mode is exclusive
            queue_family_index_count: 1,
            image_type: vk::ImageType::TYPE_2D,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let normal_image = unsafe { device.create_image(&normal_create_info, None).unwrap() };

        let normal_requirements = unsafe { device.get_image_memory_requirements(normal_image) };
        let normal_allocation = allocator
            .allocate(&AllocationCreateDesc {
                requirements: normal_requirements,
                name: "normal image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device.bind_image_memory(
                normal_image,
                normal_allocation.memory(),
                normal_allocation.offset(),
            )
        };

        // HACK: needs to be removed, you can reconstruct the worldspace coordinates from the depth
        // buffer and the world matrix and screen coords, this is unnecessary
        let position_create_info = vk::ImageCreateInfo {
            extent,
            format: position_format,
            image_type: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            p_queue_family_indices: ptr::null(), // ignored because sharing mode is exclusive
            queue_family_index_count: 1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let position_image = unsafe { device.create_image(&position_create_info, None).unwrap() };

        let position_requirements = unsafe { device.get_image_memory_requirements(position_image) };
        let position_allocation = allocator
            .allocate(&AllocationCreateDesc {
                requirements: position_requirements,
                name: "normal image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device.bind_image_memory(
                position_image,
                position_allocation.memory(),
                position_allocation.offset(),
            )
        };

        //HACK: keeping stuff around
        allocations.push(position_allocation);
        allocations.push(color_allocation);
        allocations.push(normal_allocation);
        allocations.push(depth_allocation);
        let color_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            base_mip_level: 0,
            base_array_layer: 0,
        };
        let depth_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            level_count: 1,
            layer_count: 1,
            base_mip_level: 0,
            base_array_layer: 0,
        };

        let final_color_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    format: swapchain_image_format,
                    image: swapchain_images[i],
                    view_type: vk::ImageViewType::TYPE_2D,
                    subresource_range: color_subresource_range,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let color_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    subresource_range: color_subresource_range,
                    format: color_format,
                    image: color_image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let normal_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    format: normal_format,
                    subresource_range: color_subresource_range,
                    image: normal_image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let position_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    format: position_format,
                    image: position_image,
                    subresource_range: color_subresource_range,
                    view_type: vk::ImageViewType::TYPE_2D,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        let depth_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    format: depth_format,
                    image: depth_image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    subresource_range: depth_subresource_range,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();

        color_buffers.push(color_view.clone());
        normal_buffers.push(normal_view.clone());
        position_buffers.push(position_view.clone());

        let attachments = vec![
            final_color_view,
            color_view,
            normal_view,
            depth_view,
            position_view,
        ];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo {
                        render_pass,
                        p_attachments: attachments.as_ptr(),
                        attachment_count: attachments.len() as u32,
                        width: swapchain_image_extent.width,
                        height: swapchain_image_extent.height,
                        layers: 1,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        framebuffers.push(framebuffer);
    }

    (
        framebuffers,
        color_buffers,
        normal_buffers,
        position_buffers,
        allocations,
    )
}

#[allow(clippy::too_many_lines)]
fn create_renderpass(device: &ash::Device, swapchain_image_format: vk::Format) -> vk::RenderPass {
    let attachments = vec![
        // final color attachment
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            format: swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
        // color attachment (gbuffer)
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            format: vk::Format::A2B10G10R10_UNORM_PACK32,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
        // normal attachment (gbuffer)
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            format: vk::Format::R16G16B16A16_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
        // depth attachment(gbuffer)
        vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
        // position attachment (gbuffer)
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            format: vk::Format::R32G32B32A32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
    ];
    let geometry_color_attachment_ref = vec![
        // gcolor
        vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        // gnormal
        vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        // gposition
        vk::AttachmentReference {
            attachment: 4,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
    ];
    let geometry_depth_attachment_ref = vk::AttachmentReference {
        attachment: 3,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let lighting_input_attachment_ref = vec![
        // gcolor attachment
        vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
        // gnormal
        vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
        // gposition
        vk::AttachmentReference {
            attachment: 4,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
    ];
    let lighting_color_attachment_ref = vec![vk::AttachmentReference {
        attachment: 0, // Color attachment
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ..Default::default()
    }];

    let subpasses = vec![
        // geometry pass
        vk::SubpassDescription {
            p_color_attachments: geometry_color_attachment_ref.as_ptr(),
            p_depth_stencil_attachment: &geometry_depth_attachment_ref,
            p_input_attachments: ptr::null(),
            p_preserve_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            color_attachment_count: geometry_color_attachment_ref.len() as u32,
            input_attachment_count: 0,
            preserve_attachment_count: 0,
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        },
        // lighting/final pass
        vk::SubpassDescription {
            p_input_attachments: lighting_input_attachment_ref.as_ptr(),
            p_color_attachments: lighting_color_attachment_ref.as_ptr(),
            p_depth_stencil_attachment: ptr::null(),
            p_preserve_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            input_attachment_count: lighting_input_attachment_ref.len() as u32,
            color_attachment_count: lighting_color_attachment_ref.len() as u32,
            preserve_attachment_count: 0,
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        },
    ];

    let dependancies = vec![vk::SubpassDependency {
        src_subpass: 0,
        dst_subpass: 1,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,

        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ..Default::default()
    }];

    let renderpass_create_info = vk::RenderPassCreateInfo {
        p_attachments: attachments.as_ptr(),
        attachment_count: attachments.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        subpass_count: subpasses.len() as u32,
        dependency_count: dependancies.len() as u32,
        p_dependencies: dependancies.as_ptr(),
        ..Default::default()
    };

    unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    }
}
use crate::components::{DirectionalLight, DirectionalLightUBO, PointLightUBO};

use crate::{
    components::{AmbientLight, AmbientLightUBO, Transform},
    components::{Model, ModelUBO, PointLight},
    ecs::World,
};

fn as_u32_slice(bytes: &[u8]) -> &[u32] {
    assert!(bytes.len() % 4 == 0, "length must be multiple of 4");

    unsafe { slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
}
fn create_shader_module(device: &ash::Device, path: &str) -> vk::ShaderModule {
    let path = format! {"{}/{}",env!{"OUT_DIR"}, path};
    dbg!(&path);
    let code = fs::read(path).expect("failed to read file");
    let code = as_u32_slice(&code);

    let module = vk::ShaderModuleCreateInfo {
        p_code: code.as_ptr(),
        code_size: code.len() * 4,
        ..Default::default()
    };
    unsafe { device.create_shader_module(&module, None).unwrap() }
}
static ENTRY_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

#[allow(clippy::too_many_lines)]
fn create_graphics_pipelines(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    deferred_subpass: SubpassIndex,
    lighting_subpass: SubpassIndex,
    descriptor_layouts: &DescriptorLayouts,
) -> (Vec<vk::Pipeline>, Vec<vk::PipelineLayout>) {
    let deferred_vert = create_shader_module(device, "shaders/deferred_vert.spv");
    let deferred_frag = create_shader_module(device, "shaders/deferred_frag.spv");
    let directional_vert = create_shader_module(device, "shaders/directional_vert.spv");
    let directional_frag = create_shader_module(device, "shaders/directional_frag.spv");
    let ambient_vert = create_shader_module(device, "shaders/ambient_vert.spv");
    let ambient_frag = create_shader_module(device, "shaders/ambient_frag.spv");
    let point_vert = create_shader_module(device, "shaders/point_vert.spv");
    let point_frag = create_shader_module(device, "shaders/point_frag.spv");

    let vertex_bindings = vec![vk::VertexInputBindingDescription {
        binding: 0,
        stride: 32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let vertex_attributes = vec![
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 12,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R16G16B16_SFLOAT,
            offset: 24,
        },
    ];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        p_vertex_binding_descriptions: vertex_bindings.as_ptr(),
        p_vertex_attribute_descriptions: vertex_attributes.as_ptr(),
        vertex_binding_description_count: vertex_bindings.len() as u32,
        vertex_attribute_description_count: vertex_attributes.len() as u32,
        ..Default::default()
    };

    let dummy_vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

    let deferred_stages = [
        vk::PipelineShaderStageCreateInfo {
            module: deferred_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            p_name: ENTRY_NAME.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            module: deferred_frag,
            p_name: ENTRY_NAME.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let directional_stages = [
        vk::PipelineShaderStageCreateInfo {
            module: directional_vert,
            p_name: ENTRY_NAME.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: ENTRY_NAME.as_ptr(),
            module: directional_frag,
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];
    let ambient_stages = [
        vk::PipelineShaderStageCreateInfo {
            p_name: ENTRY_NAME.as_ptr(),
            module: ambient_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: ENTRY_NAME.as_ptr(),
            module: ambient_frag,
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];
    let point_stages = [
        vk::PipelineShaderStageCreateInfo {
            p_name: ENTRY_NAME.as_ptr(),
            module: point_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: ENTRY_NAME.as_ptr(),
            module: point_frag,
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    // We must now create a **pipeline layout** object, which describes the locations and
    // types of descriptor sets and push constants used by the shaders in the pipeline.
    //
    // Multiple pipelines can share a common layout object, which is more efficient. The
    // shaders in a pipeline must use a subset of the resources described in its pipeline
    // layout, but the pipeline layout is allowed to contain resources that are not present
    // in the shaders; they can be used by shaders in other pipelines that share the same
    // layout. Thus, it is a good idea to design shaders so that many pipelines have common
    // resource locations, which allows them to share pipeline layouts.

    let deferred_layout = {
        // let bindings = vec![vk::DescriptorSetLayoutBinding {
        //     binding: 0,
        //     descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //     descriptor_count: 1,
        //     stage_flags: vk::ShaderStageFlags::VERTEX,
        //     ..Default::default()
        // }];
        //
        // let camera_layout = vk::DescriptorSetLayoutCreateInfo {
        //     p_bindings: bindings.as_ptr(),
        //     binding_count: bindings.len(),
        //     ..Default::default()
        // };
        // let model_layout = vk::DescriptorSetLayoutCreateInfo {
        //     p_bindings: bindings.as_ptr(),
        //     binding_count: bindings.len(),
        //     ..Default::default()
        // };

        // let deferred_set_layouts = vec![camera_layout, model_layout];

        let deferred_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: [descriptor_layouts.camera, descriptor_layouts.model].as_ptr(),
            set_layout_count: 2,
            ..Default::default()
        };

        let deferred_layout = unsafe {
            device
                .create_pipeline_layout(&deferred_layout_create_info, None)
                .unwrap()
        };
        deferred_layout
    };

    let directional_layout = {
        // let bindings = vec![
        //     //color attachment
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 0,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     // normals
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 1,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     //directional light data
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 2,
        //         descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        // ];
        //
        // let layout = vk::DescriptorSetLayoutCreateInfo {
        //     p_bindings: bindings.as_ptr(),
        //     binding_count: bindings.len(),
        //     ..Default::default()
        // };
        //
        // let directional_set_layouts = vec![layout];

        let directional_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: [descriptor_layouts.directional_light].as_ptr(),
            set_layout_count: [descriptor_layouts.directional_light].len() as u32,
            ..Default::default()
        };

        let directional_layout = unsafe {
            device
                .create_pipeline_layout(&directional_layout_create_info, None)
                .unwrap()
        };
        directional_layout
    };

    let ambient_layout = {
        // let bindings = vec![
        //     //color attachment
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 0,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     // normals
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 1,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     //ambient light data
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 2,
        //         descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        // ];
        //
        // let layout = vk::DescriptorSetLayoutCreateInfo {
        //     p_bindings: bindings.as_ptr(),
        //     binding_count: bindings.len(),
        //     ..Default::default()
        // };
        //
        // let ambient_set_layouts = vec![layout];

        let ambient_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: [descriptor_layouts.ambient_light].as_ptr(),
            set_layout_count: 1,
            ..Default::default()
        };

        let ambient_layout = unsafe {
            device
                .create_pipeline_layout(&ambient_layout_create_info, None)
                .unwrap()
        };
        ambient_layout
    };

    let point_layout = {
        // let bindings = vec![
        //     //color attachment
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 0,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     // normals
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 1,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     // position
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 2,
        //         descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     //point light data
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 3,
        //         descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        // ];
        //
        // let layout = vk::DescriptorSetLayoutCreateInfo {
        //     p_bindings: bindings.as_ptr(),
        //     binding_count: bindings.len(),
        //     ..Default::default()
        // };
        //
        // let point_set_layouts = vec![layout];

        let point_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: [descriptor_layouts.point_light].as_ptr(),
            set_layout_count: 1,
            ..Default::default()
        };

        let point_layout = unsafe {
            device
                .create_pipeline_layout(&point_layout_create_info, None)
                .unwrap()
        };
        point_layout
    };

    let multisample_state = &vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let rasterization_state = &vk::PipelineRasterizationStateCreateInfo {
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };
    let input_assembly_state = &vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    // We have to indicate which subpass of which render pass this pipeline is going to be
    // used in. The pipeline will only be usable from this particular subpass.
    let deferred_graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: deferred_stages.as_ptr(),
        stage_count: deferred_stages.len() as u32,
        // How vertex data is read from the vertex buffers into the vertex shader.
        p_vertex_input_state: &vertex_input_state,
        // How vertices are arranged into primitive shapes. The default primitive shape
        // is a triangle.
        p_input_assembly_state: input_assembly_state,
        // How primitives are transformed and clipped to fit the framebuffer. We use a
        // resizable viewport, set to draw over the entire window.
        p_viewport_state: &vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1),
        // How polygons are culled and converted into a raster of pixels. The default
        // value does not perform any culling.
        p_rasterization_state: rasterization_state,
        // How multiple fragment shader samples are converted to a single pixel value.
        // The default value does not perform any multisampling.
        p_multisample_state: multisample_state,
        // How pixel values are combined with the values already present in the
        // framebuffer. The default value overwrites the old value with the new one,
        // without any blending.
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                ..Default::default()
            }; FRAMES_IN_FLIGHT]
                .as_ptr(),

            attachment_count: FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        },
        p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            stencil_test_enable: vk::FALSE,
            depth_bounds_test_enable: vk::FALSE,
            ..Default::default()
        },

        // Dynamic states allows us to specify parts of the pipeline settings when
        // recording the command buffer, before we perform drawing. Here, we specify
        // that the viewport should be dynamic.
        p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR].as_ptr(),
            dynamic_state_count: 2,
            ..Default::default()
        },
        p_tessellation_state: &vk::PipelineTessellationStateCreateInfo {
            ..Default::default()
        },

        layout: deferred_layout,
        // the renderpass this graphics pipeline is one
        render_pass,
        // the subpass index
        subpass: deferred_subpass,
        // if deriving from a graphics pipeline, the index
        base_pipeline_index: 0,
        // and the handle to that pipeline
        base_pipeline_handle: vk::Pipeline::null(),
        ..Default::default()
    };

    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        color_blend_op: vk::BlendOp::ADD,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ONE,
        alpha_blend_op: vk::BlendOp::MAX,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE,
        ..Default::default()
    };
    let lighting_blend_attachments = vec![blend_attachment; 1];

    let directional_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: directional_stages.as_ptr(),
        stage_count: directional_stages.len() as u32,
        layout: directional_layout,
        p_vertex_input_state: &dummy_vertex_input_state,
        p_input_assembly_state: input_assembly_state,
        p_viewport_state: &vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1),
        p_rasterization_state: rasterization_state,
        p_multisample_state: multisample_state,
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: lighting_blend_attachments.as_ptr(),
            attachment_count: lighting_blend_attachments.len() as u32,
            ..Default::default()
        },
        p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::FALSE, // dont update depth values
            depth_compare_op: vk::CompareOp::EQUAL,
            stencil_test_enable: vk::FALSE,
            depth_bounds_test_enable: vk::FALSE,
            ..Default::default()
        },
        p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR].as_ptr(),
            dynamic_state_count: 2,
            ..Default::default()
        },

        subpass: lighting_subpass,
        render_pass,
        ..Default::default()
    };

    let ambient_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: ambient_stages.as_ptr(),
        stage_count: ambient_stages.len() as u32,
        p_vertex_input_state: &dummy_vertex_input_state,
        p_rasterization_state: rasterization_state,
        layout: ambient_layout,
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: lighting_blend_attachments.as_ptr(),
            attachment_count: lighting_blend_attachments.len() as u32,
            ..Default::default()
        },
        p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR].as_ptr(),
            dynamic_state_count: 2,
            ..Default::default()
        },
        p_input_assembly_state: input_assembly_state,
        p_multisample_state: multisample_state,
        p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::FALSE, // dont update depth values
            depth_compare_op: vk::CompareOp::EQUAL,
            stencil_test_enable: vk::FALSE,
            depth_bounds_test_enable: vk::FALSE,
            ..Default::default()
        },
        p_viewport_state: &vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1),

        subpass: lighting_subpass,
        render_pass,
        ..Default::default()
    };

    let point_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: point_stages.as_ptr(),
        stage_count: point_stages.len() as u32,
        p_vertex_input_state: &dummy_vertex_input_state,
        p_input_assembly_state: input_assembly_state,
        p_rasterization_state: rasterization_state,
        layout: point_layout,
        p_multisample_state: multisample_state,
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: lighting_blend_attachments.as_ptr(),
            attachment_count: lighting_blend_attachments.len() as u32,
            ..Default::default()
        },
        p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR].as_ptr(),
            dynamic_state_count: 2,
            ..Default::default()
        },
        p_viewport_state: &vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1),
        p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::FALSE, // dont update depth values
            depth_compare_op: vk::CompareOp::EQUAL,
            stencil_test_enable: vk::FALSE,
            depth_bounds_test_enable: vk::FALSE,
            ..Default::default()
        },

        subpass: lighting_subpass,
        render_pass,
        ..Default::default()
    };

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[
                    deferred_graphics_pipeline_create_info,
                    directional_pipeline_create_info,
                    ambient_pipeline_create_info,
                    point_pipeline_create_info,
                ],
                None,
            )
            .unwrap()
    };
    (
        pipelines,
        vec![
            deferred_layout,
            directional_layout,
            ambient_layout,
            point_layout,
        ],
    )
}

pub fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
}
fn create_entry() -> ash::Entry {
    unsafe { ash::Entry::load().unwrap() }
}
fn setup_debug_utils(
    debug_utils_loader: &ash::ext::debug_utils::Instance,
) -> vk::DebugUtilsMessengerEXT {
    let messenger_ci = populate_debug_messenger_create_info();

    let utils_messenger = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&messenger_ci, None)
            .expect("Debug Utils Callback")
    };

    utils_messenger
}

fn create_instance(entry: &ash::Entry, event_loop: &ActiveEventLoop) -> ash::Instance {
    let engine_name = CStr::from_bytes_with_nul(b"moonlight\0").unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        p_engine_name: engine_name.as_ptr(),
        ..Default::default()
    };

    let required_instance_extensions =
        ash_window::enumerate_required_extensions(event_loop.raw_display_handle().unwrap())
            .unwrap();

    let mut instance_extensions = vec![];
    for rq in required_instance_extensions {
        instance_extensions.push(*rq);
    }
    instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());

    let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();

    let validation_layers = if VALIDATION_ENABLE {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,

        enabled_layer_count: validation_layers.len() as u32,
        pp_enabled_layer_names: validation_layers.as_ptr(),

        pp_enabled_extension_names: instance_extensions.as_ptr(),
        enabled_extension_count: instance_extensions.len() as u32,
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
    instance
}
type QueueFamilyIndex = u32;
fn create_physical_device(
    instance: &ash::Instance,
    // BUG: should use this to test surface support but dont feel like it
    _surface: vk::SurfaceKHR,
    required_extensions: &Vec<&CStr>,
) -> (vk::PhysicalDevice, QueueFamilyIndex) {
    let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };

    physical_devices
        .iter()
        .copied()
        // this is desctructuring, think of it as &physical_device is like a (&,physcal_device) tuple,
        // and its doing let (&,physical_device) = physical_device_reference
        // the reference matches the reference, and the name matches the underlying data
        .filter(|&physical_device| {
            let properties = unsafe {
                instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap()
            };

            required_extensions.iter().all(|&required_name| {
                properties.iter().any(|extension_properties| {
                    *required_name == *extension_properties.extension_name_as_c_str().unwrap()
                })
            })
        })
        .filter_map(|physical_device| {
            let queue_family_properties = unsafe {
                let len =
                    instance.get_physical_device_queue_family_properties2_len(physical_device);
                let mut out = vec![vk::QueueFamilyProperties2::default(); len];

                instance.get_physical_device_queue_family_properties2(physical_device, &mut out);
                out
            };

            queue_family_properties
                .iter() // iterate over all available queues for each device
                .enumerate() // returns an iterator of type (physicaldevice, u32)
                .position(|(_, queue_family_properties)| {
                    queue_family_properties
                        .queue_family_properties
                        .queue_flags
                        .intersects(vk::QueueFlags::GRAPHICS)
                })
                .map(|index| (physical_device, index as u32))
        })
        .min_by_key(|(physical_device, _)| {
            let mut properties = vk::PhysicalDeviceProperties2::default();
            unsafe {
                instance.get_physical_device_properties2(*physical_device, &mut properties);
            };
            match properties.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                vk::PhysicalDeviceType::OTHER => 4,
                _ => 5,
            }
        })
        .expect("no qualified gpu")
}
fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: QueueFamilyIndex,
    required_extensions: &Vec<&CStr>,
) -> (ash::Device, vk::Queue) {
    let enabled_features = vk::PhysicalDeviceFeatures {
        ..Default::default()
    };
    let queue_create_infos = vec![vk::DeviceQueueCreateInfo {
        queue_family_index: queue_family_index,
        queue_count: 1,
        p_queue_priorities: [1.0].as_ptr(),
        ..Default::default()
    }];
    let required_extensions: Vec<*const i8> = required_extensions
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect();

    let create_info = vk::DeviceCreateInfo {
        p_enabled_features: &enabled_features,
        pp_enabled_extension_names: required_extensions.as_ptr(),
        enabled_extension_count: required_extensions.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        queue_create_info_count: queue_create_infos.len() as u32,

        ..Default::default()
    };

    let device = unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .unwrap()
    };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    (device, queue)
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}
fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
        _marker: PhantomData,
    }
}
