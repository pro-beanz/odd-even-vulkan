use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage}, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo}, sync::{self, GpuFuture}, VulkanLibrary};

// Sorting algorithm GLSL code.
mod sort {
    vulkano_shaders::shader!{
        ty: "compute",
        src: /*glsl*/ r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] = idx;
            }
        ",
    }
}

fn main() {
    // Find a device.
    let physical_device = Instance::new(
        VulkanLibrary::new().expect("no local Vulkan library/DLL"),
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        })
        .expect("failed to create Vulkan instance")
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    // Create a compute queue.
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: physical_device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_queue_family_index, queue_family_properties)| {
                        queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE)
                    })
                    .expect("couldn't find a compute queue family") as u32,
                ..Default::default()
            }],
            ..Default::default()
        })
        .expect("failed to create device");
    let queue = queues
        .next()
        .unwrap();

    // Create a randomly sorted set of 32-bit unsigned integers.
    let data: Vec<u32> = (0..100).map(|_| 1).collect();
    // let mut rng = rand::thread_rng();
    // let mut data: Vec<u32> = (0..100).collect();
    // data.shuffle(&mut rng);

    // Create a memory allocator.
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // Create a buffer for data source.
    let src = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
        )
        .expect("failed to create buffer");
    
    // Build the compute pipeline.
    let stage = PipelineShaderStageCreateInfo::new(
        sort::load(device.clone())
        .expect("failed to create shader module")
        .entry_point("main")
        .unwrap()
        );
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
        )
        .unwrap();
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

    // Create a descriptor set.
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let descriptor_set_layout_index = 0;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_pipeline
            .layout()
            .set_layouts()
            .get(descriptor_set_layout_index)
            .unwrap()
            .clone(),
        [WriteDescriptorSet::buffer(0, src.clone())],
        []
        )
        .unwrap();

    // Create a command buffer.
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
        );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set
        )
        .unwrap()
        .dispatch([1024, 1, 1])
        .unwrap();

    let command_buffer = builder
        .build()
        .unwrap();

    // Build and execute the command.
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    // Validate results.
    let content = src.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32);
    }
    println!("everything succeeded!");
}
