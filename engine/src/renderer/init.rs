use std::{
    ffi::{c_void, CStr},
    marker::PhantomData,
    ptr,
    sync::Arc,
}; //INFO: idk how to fix this, one wants raw window handles and the other says no
#[allow(deprecated)]
use winit::{event_loop::ActiveEventLoop, raw_window_handle::HasRawDisplayHandle, window::Window};

use ash::vk;

use crate::{renderer::draw::VALIDATION_ENABLE, vulkan::QueueFamilyIndex};



