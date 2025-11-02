use std::ptr;

use ash::vk;

#[allow(clippy::too_many_lines)]
pub fn create_renderpass(
    device: &ash::Device,
    swapchain_image_format: vk::Format,
) -> vk::RenderPass {
    let attachments = [
        // final color attachment
        vk::AttachmentDescription {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            format: swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,

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
    let geometry_color_attachment_ref = [
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
    let lighting_input_attachment_ref = [
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
    let lighting_color_attachment_ref = [vk::AttachmentReference {
        attachment: 0, // Color attachment
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ..Default::default()
    }];

    let subpasses = [
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
            p_depth_stencil_attachment: &geometry_depth_attachment_ref,
            p_preserve_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            input_attachment_count: lighting_input_attachment_ref.len() as u32,
            color_attachment_count: lighting_color_attachment_ref.len() as u32,
            preserve_attachment_count: 0,
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            ..Default::default()
        },
    ];

    let dependencies = [
        // transitions color attachemtns that are first used in subpass 1
        vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 1,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::BY_REGION,
        },
        vk::SubpassDependency {
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
        },
    ];

    let renderpass_create_info = vk::RenderPassCreateInfo {
        p_attachments: attachments.as_ptr(),
        attachment_count: attachments.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        subpass_count: subpasses.len() as u32,
        dependency_count: dependencies.len() as u32,
        p_dependencies: dependencies.as_ptr(),
        ..Default::default()
    };

    unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    }
}
