use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;

use vulkano::instance::PhysicalDevice;

use vulkano::device::Device;
use vulkano::device::Features;
use vulkano::device::RawDeviceExtensions;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;

use vulkano::sync::GpuFuture; 

use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;

use rand::{thread_rng, Rng};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/shader/compute/particle-update.glsl"
    }
}

const PARTICLE_COUNT: usize = 2048;

#[derive(Copy, Clone, Debug)]
struct Particle {
    x: f32,
    y: f32,
    speedx: f32,
    speedy: f32,
}

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    //TODO: list devices, choose based on user input
    for p in PhysicalDevice::enumerate(&instance) {
        print!("{}", p.name());
        println!(", driver version: {}", p.driver_version());
    }

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let mut dev_exts = RawDeviceExtensions::new(std::iter::empty());
    dev_exts.insert(std::ffi::CString::new("VK_KHR_storage_buffer_storage_class").unwrap());

    let (device, mut queues) = {
        Device::new(
            physical,
            &Features::none(),
            dev_exts,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();
    let particles = init_particles_buffer();
    let particles_buffer =
        CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), particles)
            .expect("failed to create buffer");


    let shader = cs::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_buffer(particles_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch([PARTICLE_COUNT as u32 / 32, 1, 1], compute_pipeline.clone(), set.clone(), ())
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
}

fn init_particles_buffer() -> [Particle; PARTICLE_COUNT] {
    let mut rng = thread_rng();
    let mut particles = [Particle {
        x: 0.0,
        y: 0.0,
        speedx: 0.0,
        speedy: 0.0,
    }; PARTICLE_COUNT];
    for i in 0..particles.len() {
        particles[i] = Particle {
            x: rng.gen_range(-1.0, 1.0),
            y: rng.gen_range(-1.0, 1.0),
            speedx: rng.gen_range(-0.1, 0.1),
            speedy: rng.gen_range(-0.1, 0.1),
        }
    }
    return particles;
}
