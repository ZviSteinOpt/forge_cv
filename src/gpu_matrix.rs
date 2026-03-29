use crate::compute_context::ComputeContext;
use crate::mat::Mat;
use crate::scalar_type::ScalarType;
use wgpu;

/// A 2D GPU buffer with shape and type metadata.
pub struct GpuMatrix {
    pub(crate) buffer: Option<wgpu::Buffer>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) channels: usize,
    pub(crate) scalar_type: ScalarType,
}

impl GpuMatrix {
    pub fn new() -> Self {
        GpuMatrix {
            buffer: None,
            rows: 0,
            cols: 0,
            channels: 1,
            scalar_type: ScalarType::Float32,
        }
    }

    // -- Data transfer ------------------------------------------------------

    /// Upload from a Mat. Auto-allocates if shape doesn't match.
    pub fn upload(&mut self, src: &Mat) {
        let need_alloc = self.buffer.is_none()
            || self.rows != src.rows
            || self.cols != src.cols
            || self.channels != src.channels
            || self.scalar_type != src.scalar_type;

        if need_alloc {
            self.release();
            *self = ComputeContext::malloc(src.rows, src.cols, src.scalar_type, src.channels);
        }

        ComputeContext::memcpy_to_device(self, src);
    }

    /// Download into a Mat. Resizes the Mat to match.
    pub fn download(&self, dst: &mut Mat) {
        ComputeContext::memcpy_to_host(self, dst);
    }

    /// Download as a square Mat: side = floor(sqrt(rows * cols)).
    /// Useful for visualizing flat GPU buffers in the debugger.
    pub fn download_square(&self, dst: &mut Mat) {
        let total = self.rows * self.cols;
        let a = (total as f64).sqrt() as usize;
        ComputeContext::memcpy_to_host_partial(self, dst, a, a);
    }

    // -- GPU operations -----------------------------------------------------

    /// Free GPU memory and reset to empty state.
    pub fn release(&mut self) {
        self.buffer = None; // wgpu::Buffer is freed on drop
        self.rows = 0;
        self.cols = 0;
        self.channels = 1;
        self.scalar_type = ScalarType::Float32;
    }

    /// Device-to-device copy. Auto-allocates dst if shape doesn't match.
    pub fn copy_to(&self, dst: &mut GpuMatrix) {
        let src_buf = self.buffer.as_ref().expect("gpu_matrix: source not allocated");
        let size = self.bytes() as u64;

        let need_alloc = dst.buffer.is_none()
            || dst.rows != self.rows
            || dst.cols != self.cols
            || dst.channels != self.channels
            || dst.scalar_type != self.scalar_type;

        if need_alloc {
            dst.release();
            *dst = ComputeContext::malloc(self.rows, self.cols, self.scalar_type, self.channels);
        }

        let inner = ComputeContext::lock();
        let mut encoder = inner.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src_buf, 0, dst.wgpu_buffer(), 0, size);
        inner.queue.submit(Some(encoder.finish()));
    }

    /// Set all bytes to zero.
    pub fn set_to_zero(&self) {
        let buf = self.buffer.as_ref().expect("gpu_matrix: not allocated");
        let size = buf.size() as usize;
        let inner = ComputeContext::lock();
        inner.queue.write_buffer(buf, 0, &vec![0u8; size]);
    }

    /// Set all bytes to the given value.
    pub fn set_to(&self, value: u8) {
        let buf = self.buffer.as_ref().expect("gpu_matrix: not allocated");
        let size = buf.size() as usize;
        let inner = ComputeContext::lock();
        inner.queue.write_buffer(buf, 0, &vec![value; size]);
    }

    // -- Metadata -----------------------------------------------------------

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn channels(&self) -> usize { self.channels }
    pub fn scalar_type(&self) -> ScalarType { self.scalar_type }
    pub fn elem_size(&self) -> usize { self.scalar_type.size() * self.channels }
    pub fn bytes(&self) -> usize { self.rows * self.cols * self.elem_size() }
    pub fn is_empty(&self) -> bool { self.rows == 0 || self.cols == 0 }

    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("gpu_matrix: not allocated")
    }
}
