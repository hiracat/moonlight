#![allow(unused_variables)]
use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};
fn main() {
    let library = VulkanLibrary::new().expect("no vulkan library/dll found");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
    println!("{:#?}", instance);
}
