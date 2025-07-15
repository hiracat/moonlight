#![allow(clippy::cast_possible_truncation)]
use crate::layouts::{self, BINDINGS};
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
    sync::{Arc, Mutex},
    time::Instant,
};
use winit::dpi::PhysicalSize;

const VALIDATION_ENABLE: bool = true;
type SharedAllocator = Arc<Mutex<Allocator>>;

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
}

impl Camera {
    pub fn create(position: Vec3, fov: f32, near: f32, far: f32, renderer: &mut Renderer) -> Self {
        let size: [f32; 2] = renderer.window.inner_size().into();
        let aspect_ratio = size[0] / size[1];
        let fov_rads = fov * (std::f32::consts::PI / 180.0);
        let rotation = Rotor3::identity();

        let mut camera = Self {
            pitch: 0.0,
            yaw: 0.0,
            position,
            rotation,
            fov_rads,
            near,
            far,
            aspect_ratio,

            u_buffer: None,
            allocations: None,
        };
        camera.populate_u_buffers(
            &renderer.device,
            renderer.allocator.clone(),
            &mut (),
            FRAMES_IN_FLIGHT,
        );
        camera.write_descriptor_sets(
            &renderer.device,
            &renderer.geometry_per_frame_1,
            BINDINGS.camera,
        );
        camera
    }
}

impl HasUBO for Camera {
    type UBOData = CameraUBO;
    type Context = ();

    #[allow(unused)]
    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData {
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
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>> {
        self.u_buffer.as_ref()
    }
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocations: Vec<Allocation>) {
        self.u_buffer = Some(buffer);
        self.allocations = Some(allocations);
    }
}

pub(crate) trait HasUBO {
    type UBOData: bytemuck::Pod + bytemuck::Zeroable;
    type Context;

    fn get_ubo_data(&self, context: &mut Self::Context) -> Self::UBOData;
    fn set_ubo_buffers(&mut self, buffer: Vec<vk::Buffer>, allocation: Vec<Allocation>);
    fn get_ubo_buffers(&self) -> Option<&Vec<vk::Buffer>>;

    fn write_descriptor_sets(
        &self,
        device: &ash::Device,
        descriptor_sets: &[vk::DescriptorSet],
        dst_binding: u32,
    ) {
        let mut buffer_infos = vec![];
        for i in 0..descriptor_sets.len() {
            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer: self.get_ubo_buffers().as_ref().unwrap()[i],
                offset: 0,
                range: size_of::<Self::UBOData>() as u64,
            })
        }
        let mut writes = vec![];
        for i in 0..descriptor_sets.len() {
            writes.push(vk::WriteDescriptorSet {
                dst_set: descriptor_sets[i],
                p_buffer_info: &buffer_infos[i],
                descriptor_count: 1,
                dst_binding,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                dst_array_element: 0,
                ..Default::default()
            })
        }
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn populate_u_buffers(
        &mut self,
        device: &ash::Device,
        memory_allocator: SharedAllocator,
        context: &mut Self::Context,
        count: usize,
    ) {
        let mut uniform_buffer = vec![];
        let mut memory = vec![];
        for _ in 0..count {
            let buffer = unsafe {
                device.create_buffer(
                    &vk::BufferCreateInfo {
                        size: size_of::<Self::UBOData>() as u64,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap();
            let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

            let mut alloc = memory_allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "uniform buffer",
                    requirements,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap();
            alloc
                .mapped_slice_mut()
                .unwrap()
                .write_all(bytes_of(&self.get_ubo_data(context)))
                .unwrap();
            unsafe {
                device
                    .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
                    .unwrap()
            }

            memory.push(alloc);
            uniform_buffer.push(buffer);
        }
        dbg!(&uniform_buffer);
        self.set_ubo_buffers(uniform_buffer, memory);
    }
}

pub fn alloc_buffer(
    memory_allocator: SharedAllocator,
    buffer_count: usize,
    size: u64,
    device: &ash::Device,
    sharing: vk::SharingMode,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
    linear: bool,
    data: &[u8],
    name: &str,
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
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        alloc.mapped_slice_mut().unwrap().write_all(data).unwrap();
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
pub(crate) struct CameraUBO {
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

pub struct Renderer {
    // needs to be kept alive, dont forget is very important
    // anything that starts with ash:: and not vk:: impliment drop
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) surface_loader: ash::khr::surface::Instance,
    pub(crate) debug_utils_loader: ash::ext::debug_utils::Instance,

    pub window: Arc<Window>,
    surface: vk::SurfaceKHR,

    pub framebuffer_resized: bool,
    pub start_time: Instant,
    swapchain_image_index: usize,
    frame_counter: u64,
    current_frame: usize,

    render_pass: vk::RenderPass,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family_index: QueueFamilyIndex,

    // there needs to be a better way to do this,
    // this is a struct that defines the layouts
    // for each shader, but i would like shader
    // reflection or something better, maybe a
    // proper save system/serilization or
    // something
    pub(crate) descriptor_layouts: layouts::DescriptorLayouts,

    pub(crate) model_pool_0: vk::DescriptorPool,
    pub(crate) geometry_per_frame_1: Vec<vk::DescriptorSet>,

    pub(crate) lighting_per_frame_sets_1: Vec<vk::DescriptorSet>,

    pub(crate) lighting_per_light_pool_2: vk::DescriptorPool,

    graphics_pipelines: Vec<GraphicsPipeline>,
    pub(crate) swapchain: SwapchainResources,
    per_frame: Vec<PerFrame>,
    pub(crate) allocator: SharedAllocator,
}

struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

pub(crate) struct SwapchainResources {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_image_format: vk::SurfaceFormatKHR,
    // one per swapchain image
    pub(crate) per_swapchain_image_descriptor_sets: Vec<vk::DescriptorSet>,

    framebuffers: Vec<vk::Framebuffer>,
    frame_buffer_allocations: Vec<Allocation>,

    pub(crate) color_buffers: Color,
    pub(crate) normal_buffers: Normal,
    pub(crate) position_buffers: Position,

    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    render_pass: vk::RenderPass,
    surface: vk::SurfaceKHR,
    per_swapchain_image_set_layout: vk::DescriptorSetLayout,
    queue_family_indices: Vec<QueueFamilyIndex>,
    allocator: SharedAllocator,
}

struct PerFrame {
    pub(crate) command_pool: vk::CommandPool,

    command_buffer: vk::CommandBuffer,
    // wait before writing to swapchain image
    image_available: vk::Semaphore,
    // wait before presenting
    render_finished: vk::Semaphore,
    // wait for the frame that was FRAMES_IN_FLIGHT frames ago, but has the same current_frame
    // since modulo
    in_flight: vk::Fence,
}

impl PerFrame {
    fn create(device: &ash::Device, queue_family_index: QueueFamilyIndex) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()
                .first()
                .copied()
                .unwrap()
        };
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        let image_available = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to create semaphore")
        };
        let render_finished = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to create semaphore")
        };
        let in_flight = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        Self {
            command_pool,
            command_buffer,
            image_available,
            render_finished,
            in_flight,
        }
    }
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

        let frame = &self.per_frame[self.current_frame];
        unsafe {
            self.device
                .wait_for_fences(&[frame.in_flight], true, u64::MAX)
                .unwrap();
        };

        self.frame_counter += 1;
        dbg!(self.frame_counter);

        let is_suboptimal;
        (self.swapchain_image_index, is_suboptimal) = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            ) {
                Ok((index, suboptimal)) => (index as usize, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.framebuffer_resized = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };

        if self.framebuffer_resized {
            // Use the new dimensions of the window.
            self.swapchain.recreate(window_size);
            world.resource_get_mut::<Camera>().unwrap().aspect_ratio =
                window_size.width as f32 / window_size.height as f32
        }
        if is_suboptimal {
            self.framebuffer_resized = true;
        }

        // NOTE: RENDERING START

        // if let Some(still_in_use) = self.swapchain_image_still_in_use[self.swapchain_image_index] {
        //     unsafe {
        //         self.device
        //             .wait_for_fences(&[still_in_use], true, 3_000_000_000)
        //             .unwrap();
        //     }
        // }
        // self.swapchain_image_still_in_use[self.swapchain_image_index] =
        //     Some(self.in_flight_fence[self.current_frame]);

        dbg!(self.current_frame);

        {
            let camera = world
                .resource_get_mut::<Camera>()
                .expect("should have a camera resource");

            let data = camera.get_ubo_data(&mut ());
            let data = bytes_of(&data);
            let mem = camera.allocations.as_mut().unwrap();
            for _ in 0..5 {
                match mem[self.current_frame].mapped_slice_mut() {
                    Some(mut write) => {
                        write.write(data).unwrap();
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
                let mut transform;
                if !world.has_component::<Transform>(entity) {
                    eprintln!(
                        "WARNING: light {:?} does not have transform component, giving identity matrix", entity
                    );
                    transform = Transform::new();
                    world.component_add(entity, transform).unwrap();
                } else {
                    transform = *world
                        .component_get::<Transform>(entity)
                        .expect("ecs corrputed, panic");
                }
                let point_light = world
                    .component_get_mut::<PointLight>(entity)
                    .expect("ecs corrupted, panic");

                let t = point_light.get_ubo_data(&mut transform);
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

                let t = &model.get_ubo_data(&mut transform);
                let data = bytes_of(t);
                let mem = model.allocations.as_mut().unwrap();
                for _ in 0..5 {
                    match mem[self.current_frame].mapped_slice_mut() {
                        Some(mut write) => {
                            write.write(data).unwrap();
                            break;
                        }
                        None => {
                            eprintln!("model {:?} memory is not accessable", entity)
                        }
                    }
                }
            }
        }

        unsafe { self.device.reset_fences(&[frame.in_flight]).unwrap() };

        unsafe {
            self.device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }

        unsafe {
            self.device
                .begin_command_buffer(
                    frame.command_buffer,
                    &vk::CommandBufferBeginInfo {
                        p_inheritance_info: &vk::CommandBufferInheritanceInfo {
                            pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
                            subpass: 0, // ingored
                            render_pass: self.render_pass,
                            framebuffer: self.swapchain.framebuffers[self.swapchain_image_index],
                            query_flags: vk::QueryControlFlags::empty(),
                            occlusion_query_enable: vk::FALSE,
                            ..Default::default()
                        },
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap()
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
                    depth: 1.0,
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
                frame.command_buffer,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.swapchain.framebuffers[self.swapchain_image_index],
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
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: window_size.width as f32,
                height: window_size.height as f32,
                max_depth: 1.0,
                min_depth: 0.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
            };
            self.device
                .cmd_set_viewport(frame.command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(frame.command_buffer, 0, &[scissor]);
            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[0].pipeline,
            );
        };

        let camera_descripor_set = &self.geometry_per_frame_1;

        let entities = world.query_mut::<Model>();

        for (_entity, model) in entities {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    frame.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipelines[0].pipeline_layout,
                    0,
                    &[
                        camera_descripor_set[self.current_frame],
                        model.descriptor_set.as_ref().unwrap()[self.current_frame],
                    ],
                    &[],
                );

                self.device.cmd_bind_vertex_buffers(
                    frame.command_buffer,
                    0,
                    &[model.vertex_buffer],
                    &[0],
                );

                self.device.cmd_bind_index_buffer(
                    frame.command_buffer,
                    model.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_draw_indexed(
                    frame.command_buffer,
                    model.index_count as u32,
                    1,
                    0,
                    0,
                    0,
                );
            };
        }

        unsafe {
            self.device
                .cmd_next_subpass(frame.command_buffer, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[1].pipeline,
            );
            let attachment_descriptor_set =
                self.swapchain.per_swapchain_image_descriptor_sets[self.swapchain_image_index];
            let light_per_frame_descriptor_set = self.lighting_per_frame_sets_1[self.current_frame];
            self.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[1].pipeline_layout,
                0,
                &[attachment_descriptor_set, light_per_frame_descriptor_set],
                &[],
            );
            self.device.cmd_draw(frame.command_buffer, 10, 1, 0, 0);

            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[2].pipeline,
            );
            self.device.cmd_draw(frame.command_buffer, 10, 1, 0, 0);
        }

        unsafe {
            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[3].pipeline,
            );
            let entities = world.query_mut::<PointLight>();
            for (_entity, light) in entities {
                self.device.cmd_bind_descriptor_sets(
                    frame.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipelines[3].pipeline_layout,
                    0,
                    &[light.descriptor_set.as_ref().unwrap()[self.current_frame]],
                    &[],
                );
                self.device.cmd_draw(frame.command_buffer, 10, 1, 0, 0);
            }
        }
        unsafe {
            self.device.cmd_end_render_pass(frame.command_buffer);
            self.device
                .end_command_buffer(frame.command_buffer)
                .unwrap();
        }

        let queue_submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &frame.image_available,
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: [frame.command_buffer].as_ptr(),
            p_signal_semaphores: &frame.render_finished,
            signal_semaphore_count: 1,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.queue, &[queue_submit_info], frame.in_flight)
                .unwrap();
        }

        let mut present_results: [vk::Result; 1] = [Default::default(); 1];
        let present_info = vk::PresentInfoKHR {
            p_wait_semaphores: &frame.render_finished,
            wait_semaphore_count: 1,
            p_swapchains: &self.swapchain.swapchain,
            swapchain_count: 1,
            p_image_indices: &(self.swapchain_image_index as u32),
            p_results: present_results.as_mut_ptr(),
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.framebuffer_resized = false;
            self.swapchain.recreate(window_size);
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    #[allow(clippy::too_many_lines)]
    pub fn init(event_loop: &ActiveEventLoop, window: &Arc<Window>) -> Self {
        let start_time = Instant::now();
        let entry = unsafe { ash::Entry::load().unwrap() };
        eprintln!("created entry");
        let instance = create_instance(&entry, event_loop);
        eprintln!("created instance");
        let surface = unsafe {
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                // idk what to do about this, i need the raw handle
                #[allow(deprecated)]
                event_loop.raw_display_handle().unwrap(),
                #[allow(deprecated)]
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

        let required_extensions = vec![
            ash::vk::KHR_SWAPCHAIN_NAME,
            // ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME,
        ];
        let (physical_device, queue_family_index) =
            create_physical_device(&instance, surface, &required_extensions);
        let (device, queue) = create_device(
            &instance,
            physical_device,
            queue_family_index,
            &required_extensions,
        );

        let window_size = window.inner_size();
        let descriptor_set_layouts = layouts::DescriptorLayouts::init(&device);

        let memory_allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let swapchain_image_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let image_format =
            SwapchainResources::choose_swapchain_format(&swapchain_image_formats).unwrap();

        let render_pass = create_renderpass(&device, image_format.format);
        let shared_allocator = Arc::new(Mutex::new(memory_allocator));
        let swapchain = SwapchainResources::create(
            &surface_loader,
            &swapchain_loader,
            &device,
            physical_device,
            render_pass,
            None,
            image_format,
            window_size,
            shared_allocator.clone(),
            surface,
            descriptor_set_layouts.lighting_per_swapchain_image_attachment_0,
            &[queue_family_index],
        );

        let pipelines = create_graphics_pipelines(
            &device,
            render_pass,
            GEOMETRY_SUBPASS,
            LIGHTING_SUBPASS,
            &descriptor_set_layouts,
        );

        eprintln!("creating sync objects");

        let mut per_frame = vec![];
        for _ in 0..FRAMES_IN_FLIGHT {
            per_frame.push(PerFrame::create(&device, queue_family_index));
        }

        let pool_sizes = vec![vk::DescriptorPoolSize {
            descriptor_count: 5,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
        }];
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: FRAMES_IN_FLIGHT as u32,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        pool_size_count: pool_sizes.len() as u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        let alloc_info = vk::DescriptorSetAllocateInfo {
            p_set_layouts: [descriptor_set_layouts.geometry_per_frame_layout_1; FRAMES_IN_FLIGHT]
                .as_ptr(),
            descriptor_pool,
            descriptor_set_count: FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        let pool_sizes = vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 3,
        }];
        let per_object_create_info = vk::DescriptorPoolCreateInfo {
            max_sets: 100,
            p_pool_sizes: pool_sizes.as_ptr(),
            pool_size_count: pool_sizes.len() as u32,
            ..Default::default()
        };
        let lighting_per_light = unsafe {
            device
                .create_descriptor_pool(&per_object_create_info, None)
                .unwrap()
        };
        let geometry_per_model = unsafe {
            device
                .create_descriptor_pool(&per_object_create_info, None)
                .unwrap()
        };

        let lighting_per_frame_pool_sizes = vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 6,
        }];
        let lighting_per_frame_create_info = vk::DescriptorPoolCreateInfo {
            max_sets: FRAMES_IN_FLIGHT as u32,
            p_pool_sizes: lighting_per_frame_pool_sizes.as_ptr(),
            pool_size_count: lighting_per_frame_pool_sizes.len() as u32,
            ..Default::default()
        };
        let lighting_per_frame_pool = unsafe {
            device
                .create_descriptor_pool(&lighting_per_frame_create_info, None)
                .unwrap()
        };

        let alloc_info = vk::DescriptorSetAllocateInfo {
            p_set_layouts: [descriptor_set_layouts.lighting_per_frame_layout_1; FRAMES_IN_FLIGHT]
                .as_ptr(),
            descriptor_pool: lighting_per_frame_pool,
            descriptor_set_count: FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };
        let lighting_per_frame = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        Renderer {
            model_pool_0: geometry_per_model,
            lighting_per_light_pool_2: lighting_per_light,
            lighting_per_frame_sets_1: lighting_per_frame,
            _entry: entry,
            debug_messenger,
            debug_utils_loader,
            instance,
            descriptor_layouts: descriptor_set_layouts,
            framebuffer_resized: false,
            current_frame: 0,
            frame_counter: 0,
            swapchain,
            render_pass,
            allocator: shared_allocator,
            geometry_per_frame_1: sets,
            surface,
            surface_loader,
            swapchain_loader: swapchain_loader.clone(),
            device,
            queue,
            start_time,
            window: window.clone(),
            queue_family_index,
            physical_device,
            graphics_pipelines: pipelines,
            per_frame,

            swapchain_image_index: 0,
        }
    }
}

/// ## Args
/// - **descriptor_pool**: Pool for descriptor set allocation
/// - **attachment_layout**: Descriptor set layout defining binding structure
/// - **attachments**: Interleaved attachment views (e.g., `[color0, normal0, pos0, color1, ...]`)
/// - **binding_index**: Binding indices for each attachment type (length must evenly divide
/// attachments.len())

#[allow(clippy::too_many_lines)]
pub(crate) fn create_attachment_descriptor_sets(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    attachment_layout: vk::DescriptorSetLayout,
    attachments: &[vk::ImageView],
    binding_indices: &[u32],
) -> Vec<vk::DescriptorSet> {
    debug_assert_eq!(
        attachments.len() % binding_indices.len(),
        0,
        "attachments.len() ({}) must be evenly divisable by binding_index.len() ({})",
        attachments.len(),
        binding_indices.len()
    );
    debug_assert!(
        attachments.len() >= binding_indices.len(),
        "Must have at least {} attachments for one complete frame",
        binding_indices.len()
    );

    let frame_count = attachments.len() / binding_indices.len();

    let set_layouts = vec![attachment_layout; frame_count];

    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                p_set_layouts: set_layouts.as_ptr(),
                descriptor_set_count: set_layouts.len() as u32,
                ..Default::default()
            })
            .unwrap()
    };
    let mut descriptor_writes = vec![];
    for i in 0..attachments.len() {
        let frame_index = i / binding_indices.len(); // Which frame/set this attachment belongs to
        let binding_index = i % binding_indices.len(); // Which binding within that frame

        descriptor_writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_sets[frame_index],
            p_image_info: &vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: attachments[i],
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            descriptor_count: 1,
            dst_binding: binding_indices[binding_index],
            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
            dst_array_element: 0,
            ..Default::default()
        })
    }

    let writes = &descriptor_writes;
    dbg!(writes);

    unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }

    descriptor_sets
}

type Color = Vec<vk::ImageView>;
type Normal = Vec<vk::ImageView>;
type Position = Vec<vk::ImageView>;
pub fn create_framebuffers(
    device: &ash::Device,
    swapchain_images: &[vk::Image],
    render_pass: vk::RenderPass,
    allocator: SharedAllocator,
    swapchain_image_extent: vk::Extent2D,
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
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                requirements: depth_requirements,
                name: "depth image in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device
                .bind_image_memory(
                    depth_image,
                    depth_allocation.memory(),
                    depth_allocation.offset(),
                )
                .unwrap()
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
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                requirements: color_requirements,
                name: "color image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device
                .bind_image_memory(
                    color_image,
                    color_allocation.memory(),
                    color_allocation.offset(),
                )
                .unwrap()
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
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                requirements: normal_requirements,
                name: "normal image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device
                .bind_image_memory(
                    normal_image,
                    normal_allocation.memory(),
                    normal_allocation.offset(),
                )
                .unwrap()
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
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                requirements: position_requirements,
                name: "normal image allocation in create_framebuffers",
                linear: false,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            device
                .bind_image_memory(
                    position_image,
                    position_allocation.memory(),
                    position_allocation.offset(),
                )
                .unwrap()
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
            store_op: vk::AttachmentStoreOp::STORE,
            format: vk::Format::A2B10G10R10_UNORM_PACK32,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
        // normal attachment (gbuffer)
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
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
            store_op: vk::AttachmentStoreOp::STORE,
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

        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,

        dst_access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::BY_REGION,
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

use crate::{
    components::Transform,
    components::{Model, PointLight},
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
static SHADER_ENTRY_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

#[allow(clippy::too_many_lines)]
fn create_graphics_pipelines(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    deferred_subpass: SubpassIndex,
    lighting_subpass: SubpassIndex,
    descriptor_layouts: &layouts::DescriptorLayouts,
) -> Vec<GraphicsPipeline> {
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
        // position
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        // normal
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 12,
        },
        // color
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

    // required even if empty
    let dummy_vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
    let dummy_vertex_input_state = [dummy_vertex_input_state].as_ptr();

    let deferred_stages = [
        vk::PipelineShaderStageCreateInfo {
            module: deferred_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            module: deferred_frag,
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let directional_stages = [
        vk::PipelineShaderStageCreateInfo {
            module: directional_vert,
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            module: directional_frag,
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];
    let ambient_stages = [
        vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            module: ambient_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            module: ambient_frag,
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];
    let point_stages = [
        vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRY_NAME.as_ptr(),
            module: point_vert,
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRY_NAME.as_ptr(),
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
        let deferred_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: [
                descriptor_layouts.geometry_per_model_layout_0,
                descriptor_layouts.geometry_per_frame_layout_1,
            ]
            .as_ptr(),
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

    let set_layouts = [
        descriptor_layouts.lighting_per_swapchain_image_attachment_0,
        descriptor_layouts.lighting_per_frame_layout_1,
        descriptor_layouts.lighting_per_light_layout_2,
    ];

    let lighting_layout = {
        let directional_layout_create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_set_layouts: set_layouts.as_ptr(),
            set_layout_count: set_layouts.len() as u32,
            ..Default::default()
        };

        let lighting_layout = unsafe {
            device
                .create_pipeline_layout(&directional_layout_create_info, None)
                .unwrap()
        };
        lighting_layout
    };

    let multisample_state = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let rasterization_state = &vk::PipelineRasterizationStateCreateInfo {
        cull_mode: vk::CullModeFlags::NONE,
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
        p_multisample_state: &multisample_state,
        // How pixel values are combined with the values already present in the
        // framebuffer. The default value overwrites the old value with the new one,
        // without any blending.
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                ..Default::default()
            }; 3]
                .as_ptr(),

            attachment_count: 3,
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
        blend_enable: vk::TRUE,
        ..Default::default()
    };
    let lighting_blend_attachments = vec![blend_attachment; 1];

    let directional_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_stages: directional_stages.as_ptr(),
        stage_count: directional_stages.len() as u32,
        layout: lighting_layout,
        p_vertex_input_state: dummy_vertex_input_state,
        p_input_assembly_state: input_assembly_state,
        p_viewport_state: &vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1),
        p_rasterization_state: rasterization_state,
        p_multisample_state: &multisample_state,
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            p_attachments: lighting_blend_attachments.as_ptr(),
            attachment_count: lighting_blend_attachments.len() as u32,
            ..Default::default()
        },
        p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE, // dont update depth values
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
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
        p_vertex_input_state: dummy_vertex_input_state,
        p_rasterization_state: rasterization_state,
        layout: lighting_layout,
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
        p_multisample_state: &multisample_state,
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
        p_vertex_input_state: dummy_vertex_input_state,
        p_input_assembly_state: input_assembly_state,
        p_rasterization_state: rasterization_state,
        layout: lighting_layout,
        p_multisample_state: &multisample_state,
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
    vec![
        GraphicsPipeline {
            pipeline: pipelines[0],
            pipeline_layout: deferred_layout,
        },
        GraphicsPipeline {
            pipeline: pipelines[1],
            pipeline_layout: lighting_layout,
        },
        GraphicsPipeline {
            pipeline: pipelines[2],
            pipeline_layout: lighting_layout,
        },
        GraphicsPipeline {
            pipeline: pipelines[3],
            pipeline_layout: lighting_layout,
        },
    ]
}

pub fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("moonlight"))
            .expect("failed to create window"),
    )
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

    #[allow(deprecated)]
    let required_instance_extensions =
        ash_window::enumerate_required_extensions(event_loop.raw_display_handle().unwrap())
            .unwrap();

    let mut instance_extensions = vec![];
    for rq in required_instance_extensions {
        instance_extensions.push(*rq);
    }
    instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());

    let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
    let validation_features = [
        vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
        vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
        vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
        vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
        // vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
    ];
    let validation_features_info = vk::ValidationFeaturesEXT {
        enabled_validation_feature_count: validation_features.len() as u32,
        p_enabled_validation_features: validation_features.as_ptr(),
        disabled_validation_feature_count: 0,
        p_disabled_validation_features: std::ptr::null(),
        ..Default::default()
    };

    let validation_layers = if VALIDATION_ENABLE {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    let create_info = vk::InstanceCreateInfo {
        p_next: &validation_features_info as *const _ as *const std::ffi::c_void,
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
    // TODO: should use this to test surface support but dont feel like it
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
    let enabled_features = unsafe { instance.get_physical_device_features(physical_device) };
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
    println!("{}{}{:?}", severity, types, message);

    vk::FALSE
}
fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,

        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
        _marker: PhantomData,
    }
}
impl SwapchainResources {
    pub(crate) fn recreate(&mut self, size: PhysicalSize<u32>) {
        *self = SwapchainResources::create(
            &self.surface_loader,
            &self.swapchain_loader,
            &self.device,
            self.physical_device,
            self.render_pass,
            Some(self.swapchain),
            self.swapchain_image_format,
            size,
            self.allocator.clone(),
            self.surface,
            self.per_swapchain_image_set_layout,
            &self.queue_family_indices,
        );
    }
    pub(crate) fn create(
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        render_pass: vk::RenderPass,
        previous_swapchain: Option<vk::SwapchainKHR>,
        swapchain_image_format: vk::SurfaceFormatKHR,
        window_size: PhysicalSize<u32>,
        allocator: SharedAllocator,
        surface: vk::SurfaceKHR,
        per_swapchain_image_set_layout: vk::DescriptorSetLayout,
        queue_family_indices: &[QueueFamilyIndex],
    ) -> Self {
        let window_size = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };

        eprintln!(
            "Surface loader valid: {}",
            !(surface_loader as *const _ as usize == 0)
        );
        let image_format = swapchain_image_format;

        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let image_count = if capabilities.min_image_count == capabilities.max_image_count {
            capabilities.max_image_count
        } else {
            capabilities.min_image_count + 1
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            surface,
            min_image_count: image_count,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            image_color_space: image_format.color_space,
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

        let (framebuffers, color_buffers, normal_buffers, position_buffers, allocations) =
            create_framebuffers(
                device,
                &swapchain_images,
                render_pass,
                allocator.clone(),
                window_size,
                image_format.format,
            );

        // one set per swapchain image, * 4 for color normal position and final_color * 3 for
        // safety
        let required_sets: u32 = (swapchain_images.len() * 4 * 3) as u32;

        let pool_sizes = vec![vk::DescriptorPoolSize {
            // per uniform buffer, so six descriptors per
            // set(2x what is needed for safety
            descriptor_count: 6 * required_sets,
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
        }];

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

        let mut interleaved = vec![];
        for i in 0..color_buffers.len() {
            interleaved.push(color_buffers[i]);
            interleaved.push(normal_buffers[i]);
            interleaved.push(position_buffers[i]);
        }

        let descriptor_sets = create_attachment_descriptor_sets(
            device,
            descriptor_pool,
            per_swapchain_image_set_layout,
            &interleaved,
            &[0, 1, 2],
        );

        eprintln!("swapchain created successfully");

        Self {
            swapchain,
            swapchain_image_format: image_format,
            color_buffers,
            normal_buffers,
            position_buffers,
            frame_buffer_allocations: allocations,
            framebuffers,
            allocator,
            per_swapchain_image_descriptor_sets: descriptor_sets,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            device: device.clone(),
            physical_device,
            render_pass,
            surface,
            per_swapchain_image_set_layout,
            queue_family_indices: queue_family_indices.to_vec(),
        }
    }
    fn choose_swapchain_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
        let prefered_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let prefered_formats = vec![
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
        ];

        for available in formats {
            for format in &prefered_formats {
                if available.format == *format && available.color_space == prefered_color_space {
                    return Some(*available);
                }
            }
        }
        return formats.first().copied();
    }
}
