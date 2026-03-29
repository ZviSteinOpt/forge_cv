use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use wgpu;
use wgpu::util::DeviceExt;

use crate::gpu_matrix::GpuMatrix;
use crate::mat::Mat;
use crate::scalar_type::ScalarType;

// ---------------------------------------------------------------------------
// Internal state — hidden behind the singleton
// ---------------------------------------------------------------------------

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub(crate) struct ComputeContextInner {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    adapter_name: String,
    shaders: HashMap<String, wgpu::ShaderModule>,
    pipelines: HashMap<String, CachedPipeline>,
}

static CTX: OnceLock<Mutex<ComputeContextInner>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Public API — all static methods, no instance needed
// ---------------------------------------------------------------------------

pub struct ComputeContext;

impl ComputeContext {
    /// Initialise the global GPU context. Call once at startup.
    pub fn init() {
        CTX.get_or_init(|| {
            Mutex::new(pollster::block_on(Self::new_async()))
        });
    }

    /// GPU adapter name (e.g. "Apple M1 Max").
    pub fn device_name() -> String {
        let inner = Self::lock();
        inner.adapter_name.clone()
    }

    /// Register a named kernel from WGSL source.
    /// The entry-point function in the shader must be called `main`.
    pub fn load_kernel(name: &str, wgsl_source: &str) {
        let mut inner = Self::lock();
        let module = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });
        inner.shaders.insert(name.to_string(), module);
    }

    // -- Memory management --------------------------------------------------

    /// Allocate a GPU buffer with the given shape and type.
    pub fn malloc(
        rows: usize,
        cols: usize,
        scalar_type: ScalarType,
        channels: usize,
    ) -> GpuMatrix {
        assert!(rows > 0 && cols > 0 && channels > 0, "Invalid dimensions");

        let inner = Self::lock();
        let elem_size = scalar_type.size() * channels;
        // wgpu requires buffer sizes aligned to COPY_BUFFER_ALIGNMENT (4 bytes)
        let raw_size = (rows * cols * elem_size) as u64;
        let size = (raw_size + 3) & !3;

        let buffer = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_matrix"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuMatrix {
            buffer: Some(buffer),
            rows,
            cols,
            channels,
            scalar_type,
        }
    }

    /// Upload CPU data (Mat) into a GPU buffer (GpuMatrix).
    pub fn memcpy_to_device(dst: &mut GpuMatrix, src: &Mat) {
        let buf = dst.wgpu_buffer();
        assert_eq!(src.bytes(), dst.bytes(), "memcpy_to_device: size mismatch");

        let inner = Self::lock();
        inner.queue.write_buffer(buf, 0, src.as_bytes());
    }

    /// Download GPU data (GpuMatrix) into a CPU buffer (Mat).
    pub fn memcpy_to_host(src: &GpuMatrix, dst: &mut Mat) {
        let buf = src.wgpu_buffer();
        let size = src.bytes() as u64;

        let inner = Self::lock();

        let staging = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = inner.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        inner.device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().expect("Buffer map failed");

        let mapped = slice.get_mapped_range();
        dst.resize(src.rows(), src.cols(), src.scalar_type(), src.channels());
        dst.as_bytes_mut().copy_from_slice(&mapped);
        drop(mapped);
        staging.unmap();
    }

    /// Download a partial region of GPU data into a CPU buffer (Mat).
    /// Downloads rows*cols elements with the source's scalar_type and channels.
    pub fn memcpy_to_host_partial(src: &GpuMatrix, dst: &mut Mat, rows: usize, cols: usize) {
        let buf = src.wgpu_buffer();
        let elem_size = src.scalar_type().size() * src.channels();
        let copy_bytes = (rows * cols * elem_size) as u64;
        // wgpu copy size must be aligned to 4
        let aligned_size = (copy_bytes + 3) & !3;

        let inner = Self::lock();

        let staging = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read_partial"),
            size: aligned_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = inner.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, aligned_size);
        inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        inner.device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().expect("Buffer map failed");

        let mapped = slice.get_mapped_range();
        dst.resize(rows, cols, src.scalar_type(), src.channels());
        dst.as_bytes_mut().copy_from_slice(&mapped[..copy_bytes as usize]);
        drop(mapped);
        staging.unmap();
    }

    // -- Compute dispatch ---------------------------------------------------

    /// Dispatch a compute kernel.
    ///
    /// - `kernel_name`: must match a previously loaded kernel.
    /// - `num_workgroups`: number of workgroups (NOT total threads).
    /// - `buffers`: GPU buffers bound to @binding(0), @binding(1), …
    /// - `uniform_data`: optional bytes for a uniform buffer at the next binding index.
    pub fn dispatch(
        kernel_name: &str,
        num_workgroups: u32,
        buffers: &[&wgpu::Buffer],
        uniform_data: Option<&[u8]>,
    ) {
        let mut inner = Self::lock();

        let n_storage = buffers.len();
        let n_uniform: usize = if uniform_data.is_some() { 1 } else { 0 };

        Self::get_or_create_pipeline(&mut inner, kernel_name, n_storage, n_uniform);

        let key = Self::pipeline_key(kernel_name, n_storage, n_uniform);
        let cached = inner.pipelines.get(&key).unwrap();

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (i, buf) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let uniform_buf;
        if let Some(data) = uniform_data {
            let padded_len = (data.len() + 15) & !15;
            let mut padded = vec![0u8; padded_len];
            padded[..data.len()].copy_from_slice(data);

            uniform_buf = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniform"),
                contents: &padded,
                usage: wgpu::BufferUsages::UNIFORM,
            });

            entries.push(wgpu::BindGroupEntry {
                binding: n_storage as u32,
                resource: uniform_buf.as_entire_binding(),
            });
        }

        let bind_group = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cached.bind_group_layout,
            entries: &entries,
        });

        let mut encoder = inner.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        inner.queue.submit(Some(encoder.finish()));
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    pub(crate) fn lock() -> std::sync::MutexGuard<'static, ComputeContextInner> {
        CTX.get().expect("ComputeContext::init() not called")
            .lock().unwrap()
    }

    async fn new_async() -> ComputeContextInner {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find a WebGPU adapter");

        let adapter_name = adapter.get_info().name;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("compute_device"),
                required_features: wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .expect("Failed to get WebGPU device");

        ComputeContextInner {
            device,
            queue,
            adapter_name,
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }

    fn pipeline_key(name: &str, n_storage: usize, n_uniform: usize) -> String {
        format!("{}:{}:{}", name, n_storage, n_uniform)
    }

    fn get_or_create_pipeline<'a>(
        inner: &'a mut ComputeContextInner,
        kernel_name: &str,
        n_storage: usize,
        n_uniform: usize,
    ) -> &'a CachedPipeline {
        let key = Self::pipeline_key(kernel_name, n_storage, n_uniform);

        if !inner.pipelines.contains_key(&key) {
            let module = inner
                .shaders
                .get(kernel_name)
                .unwrap_or_else(|| panic!("Kernel '{}' not loaded", kernel_name));

            let mut layout_entries = Vec::new();
            for i in 0..n_storage {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            for i in 0..n_uniform {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: (n_storage + i) as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }

            let bind_group_layout =
                inner.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &layout_entries,
                });

            let pipeline_layout =
                inner.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline =
                inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(kernel_name),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            inner.pipelines.insert(
                key.clone(),
                CachedPipeline {
                    pipeline,
                    bind_group_layout,
                },
            );
        }

        inner.pipelines.get(&key).unwrap()
    }
}
