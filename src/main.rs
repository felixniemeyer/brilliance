use vulkano::instance::Instance;

use vulkano::instance::PhysicalDevice;

use vulkano::pipeline::GraphicsPipeline; 
use vulkano::pipeline::viewport::Viewport; 

use vulkano::device::Device;
use vulkano::device::Features;
use vulkano::device::RawDeviceExtensions;

use vulkano::framebuffer::{
    Framebuffer, 
    FramebufferAbstract, 
    Subpass, 
    RenderPassAbstract
};

use vulkano::image::SwapchainImage; 

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, 
    DynamicState
};

use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;

use vulkano::sync; 
use vulkano::sync::{
    GpuFuture, 
    FlushError
};

use vulkano::format::Format; 
use vulkano::image::{
    Dimensions, 
    StorageImage, 
    ImmutableImage, 
};

use vulkano::format::ClearValue; 

use rand::{
    thread_rng, 
    Rng
};

use image::{
    GenericImageView, 
    DynamicImage, 
};

use vulkano_win::VkSurfaceBuild; 
use winit::{
    EventsLoop, 
    WindowBuilder, 
    Window, 
    Event, 
    WindowEvent
};

use vulkano::swapchain; 
use vulkano::swapchain::{
    AcquireError, 
    Swapchain, 
    SurfaceTransform, 
    PresentMode, 
    SwapchainCreationError
}; 

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/shader/compute/particle-update.glsl"
    }
}

const PARTICLE_COUNT: usize = 2048;

#[derive(Copy, Clone, Debug)]
struct Particle {
    pos: [f32; 2],
    speed: [f32; 2],
    tail: [f32; 2],
    prev_pos: [f32; 2],
    prev_tail: [f32; 2],
}

// struct ImageDataIterator<P> {
//     originalIterator: image::buffer::Pie, 
//     length: usize
// }
// 
// impl<T> ImageDataIterator<T> { 
//     fn from_dynamic_image(img: &DynamicImage) -> ImageDataIterator<T> {
//         let dimensions = img.dimensions();
//         
//         ImageDataIterator { 
//             originalIterator: img.to_rgba().pixels(), 
//             length: ( dimensions.0 * dimensions.1 ) as usize
//         }
//     }
// }
// 
// impl<'a, T> Iterator for ImageDataIterator<'a, T> {
//     type Item = [u8; 4]; 
//     fn next(&mut self) -> Option<[u8; 4]> {
//         return match self.originalIterator.next() {
//             Some(pixel) => {
//                 let rgba = pixel.2;
//                 let data: [u8; 4] = [ rgba[0], rgba[1], rgba[2], rgba[3] ]; 
//                 return Some(data);
//             }, 
//             None => None
//         }
//     }
// }
// 
// impl<'a, T> ExactSizeIterator for ImageDataIterator<'a, T> {
//     fn len(&self) -> usize {
//         return self.length; 
//     }
// }



fn main() {

    let img = match image::open("./media/autumn.png") {
        Ok(image) => image, 
        Err(err) => panic!("{:?}", err)
    };

    {   // stdout image info
        println!("color {:?}", img.color()); 
        println!("dimensions {:?}", img.dimensions()); 
        println!("first pixel {:?}", img.pixels().next().unwrap()); 

        println!("first pixel {:?}", img.pixels().next().map(|item| item.2).unwrap()); 

        img.as_bgr8().into_iter().for_each(|item| {
           println!("yey: {:?}", item);
        }); 
    }


    let instance = {
        let inst_exts = vulkano_win::required_extensions(); 
        Instance::new(None, &inst_exts, None).expect("failed to create instance")
    };

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


    let (device, mut queues) = {
        let unraw_dev_exts = vulkano::device::DeviceExtensions {
            khr_swapchain: true, 
            .. vulkano::device::DeviceExtensions::none()
        };
        let mut all_dev_exts = RawDeviceExtensions::from(&unraw_dev_exts);
        all_dev_exts.insert(std::ffi::CString::new("VK_KHR_storage_buffer_storage_class").unwrap());

        Device::new(
            physical,
            &Features::none(),
            all_dev_exts,
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

    let image = StorageImage::new(
        device.clone(), 
        Dimensions::Dim2d { width: 1024, height: 1024 },
        Format::R8G8B8A8Unorm, 
        Some(queue.family()))
        .unwrap(); 

    let clear_command = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
        .unwrap()
        .build().unwrap();

    let mut events_loop = EventsLoop::new(); 
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window(); 


    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical) 
            .expect("failed to get surface capabilities"); 
        let usage = caps.supported_usage_flags; 
        let alpha = caps.supported_composite_alpha.iter().next().unwrap(); 
        let format = caps.supported_formats[0].0;

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else { 
            return;
        };
        Swapchain::new(device.clone(), surface.clone(), 
        caps.min_image_count, format, initial_dimensions, 1, usage, &queue, 
        SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None)
        .expect("failed to create swapchain")
    };

    #[derive(Default, Debug, Clone)]
    struct Vertex { 
        position: [f32; 2]
    }

    let vertex_buffer = { 
        vulkano::impl_vertex!(Vertex, position); 

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [-0.5, -0.5] }, 
            Vertex { position: [ 0.5, -0.5] }, 
            Vertex { position: [-0.5,  0.5] }, 
            Vertex { position: [ 0.5,  0.5] }, 
            Vertex { position: [-0.5,  0.5] }, 
            Vertex { position: [ 0.5, -0.5] }, 
        ].iter().cloned()).unwrap()
    }; 

    // texture
    let img_dim = img.dimensions();
    let autumn_texture = ImmutableImage::from_iter(
        img.to_rgba().pixels().map(|rgba| {
            let bytes : [u8; 4] = [rgba[0], rgba[1], rgba[2], rgba[3]]; 
            bytes
        }),
        Dimensions::Dim2d { width: img_dim.0, height: img_dim.1 },
        Format::R8G8B8A8Unorm, 
        queue.clone()
    )
    .unwrap();


    mod square_vs { 
        vulkano_shaders::shader!{
            ty: "vertex", 
            path: "./src/shader/compute/square_vs.glsl"
        }
    }

    mod square_fs { 
        vulkano_shaders::shader!{
            ty: "fragment", 
            path: "./src/shader/compute/square_fs.glsl"
        }
    }

    let square_vs = square_vs::Shader::load(device.clone()).unwrap(); 
    let square_fs = square_fs::Shader::load(device.clone()).unwrap(); 

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(), 
        attachments: {
            color: {
                load: Clear, 
                store: Store, 
                format: swapchain.format(), 
                samples: 1, 
            }
        }, 
        pass: {
            color: [color], 
            depth_stencil: {}
        }
    ).unwrap()); 

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(square_vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(square_fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()); 

    let mut dynamic_state = DynamicState { 
        line_width: None, 
        viewports: None, 
        scissors: None, 
        compare_mask: None, 
        write_mask: None, 
        reference: None 
    }; 
    
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state); 


    let mut recreate_swapchain = false; 

    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>; 

    loop {
        previous_frame_end.cleanup_finished(); 

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into(); 
                [dimensions.0, dimensions.1]
            } else {
                return ;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r, 
                Err(SwapchainCreationError::UnsupportedDimensions) => continue, 
                Err(err) => panic!("{:?}", err)
            }; 

            println!("recreated swapchain");

            swapchain = new_swapchain; 

            framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state); 
            recreate_swapchain = false; 
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None){
            Ok(r) => r, 
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true; 
                continue; 
            }, 
            Err(err) => panic!("{:?}", err)
        }; 

        if image_num > 2 { 
            recreate_swapchain = true; 
            continue; 
        } // ugly workaround for a situation when image_num is out of bounds


        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into()); 

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()

            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
            .unwrap()

            .end_render_pass()
            .unwrap()

            .build()
            .unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num) 
            .then_signal_fence_and_flush(); 

        match future {
            Ok(future) => {
                future.wait(None).unwrap(); 
                previous_frame_end = Box::new(future) as Box<_>; 
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true; 
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>; 
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false; 

        events_loop.poll_events(|event| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true, 
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

fn init_particles_buffer() -> [Particle; PARTICLE_COUNT] {
    let mut rng = thread_rng();
    let mut particles = [Particle {
        pos: [0.0, 0.0],
        tail: [0.0, 0.0],
        speed: [0.0, 0.0],
        prev_pos: [0.0, 0.0],
        prev_tail: [0.0, 0.0],
    }; PARTICLE_COUNT];
    for i in 0..particles.len() {
        particles[i].pos = [rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0)];
        particles[i].tail = particles[i].pos.clone();
        particles[i].speed = [rng.gen_range(-0.1, 0.1), rng.gen_range(-0.1, 0.1)];
    }
    return particles;
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>], 
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, 
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions(); 

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32], 
        depth_range: 0.0 .. 1.0, 
    }; 

    dynamic_state.viewports = Some(vec!(viewport)); 

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}

