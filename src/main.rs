use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage}, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo}, sync::{self, GpuFuture}, VulkanLibrary};

fn main() {
    // Create Vulkan instance.
    let vulkan = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(
        vulkan,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create Vulkan instance");

    // Find a device.
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    // Create a graphical queue.
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphiacl queue family") as u32;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    // Create a memory allocator.
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // Create a randomly sorted set of 32-bit unsigned integers.
    let data: Vec<u32> = (0..100).map(|_| 1).collect();
    // let mut rng = rand::thread_rng();
    // let mut data: Vec<u32> = (0..100).collect();
    // data.shuffle(&mut rng);

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

    // Sorting algorithm GLSL code.
    mod sort {
        vulkano_shaders::shader!{
            ty: "compute",
            src: r"
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
    
    // Build the compute pipeline.
    let shader = sort::load(device.clone()).expect("failed to create shader module");
    let sort = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(sort);
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
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
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

    let command_buffer = builder.build().unwrap();

    // Build the command.
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // Execute the command.
    future.wait(None).unwrap();

    let content = src.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32);
    }
    println!("everything succeeded!");
}
