use crate::scalar_type::ScalarType;
use std::fs;
use std::io::Write;
use std::path::Path;

/// A CPU-side matrix — thin wrapper over raw bytes + shape/type metadata.
///
/// Mat is standalone. It knows nothing about GpuMatrix.
/// GpuMatrix knows Mat and can download into it / upload from it.
/// The VS Code extension reads Mat via the debugger.
pub struct Mat {
    pub data: Vec<u8>,
    /// Address as plain usize — debugger can read without casting
    pub data_ptr: usize,
    pub data_len: usize,
    pub rows: usize,
    pub cols: usize,
    pub channels: usize,
    pub scalar_type: ScalarType,
    pub name: String,
}

impl Mat {
    /// Keep data_ptr/data_len in sync with data vec.
    fn sync(&mut self) {
        self.data_ptr = self.data.as_ptr() as usize;
        self.data_len = self.data.len();
    }

    pub fn new(name: &str) -> Self {
        let mut m = Mat {
            data: Vec::new(),
            data_ptr: 0,
            data_len: 0,
            rows: 0,
            cols: 0,
            channels: 1,
            scalar_type: ScalarType::Float32,
            name: name.to_string(),
        };
        m.sync();
        m
    }

    pub fn with_shape(
        name: &str,
        rows: usize,
        cols: usize,
        scalar_type: ScalarType,
        channels: usize,
    ) -> Self {
        let size = rows * cols * channels * scalar_type.size();
        let mut m = Mat {
            data: vec![0u8; size],
            data_ptr: 0,
            data_len: 0,
            rows,
            cols,
            channels,
            scalar_type,
            name: name.to_string(),
        };
        m.sync();
        m
    }

    pub fn from_slice<T: bytemuck::Pod>(
        name: &str,
        rows: usize,
        cols: usize,
        scalar_type: ScalarType,
        channels: usize,
        data: &[T],
    ) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let expected = rows * cols * channels * scalar_type.size();
        assert_eq!(bytes.len(), expected, "Data size mismatch");

        let mut m = Mat {
            data: bytes.to_vec(),
            data_ptr: 0,
            data_len: 0,
            rows,
            cols,
            channels,
            scalar_type,
            name: name.to_string(),
        };
        m.sync();
        m
    }

    /// Load an image file (jpg, png, etc.) into a Mat.
    /// Returns RGB uint8, 3 channels.
    pub fn imread(path: &str) -> Self {
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("Failed to open '{}': {}", path, e))
            .to_rgb8();

        let (w, h) = img.dimensions();
        let data = img.into_raw();
        let name = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();

        let mut m = Mat {
            data,
            data_ptr: 0,
            data_len: 0,
            rows: h as usize,
            cols: w as usize,
            channels: 3,
            scalar_type: ScalarType::Uint8,
            name,
        };
        m.sync();
        m
    }

    // -- Accessors ----------------------------------------------------------

    pub fn name(&self) -> &str { &self.name }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn channels(&self) -> usize { self.channels }
    pub fn scalar_type(&self) -> ScalarType { self.scalar_type }
    pub fn elem_size(&self) -> usize { self.scalar_type.size() * self.channels }
    pub fn bytes(&self) -> usize { self.data.len() }

    pub fn as_bytes(&self) -> &[u8] { &self.data }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // Note: caller must call sync() after mutating if pointer may have changed.
        // In practice, resize() is always called first which handles sync.
        &mut self.data
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }

    pub fn as_slice_mut<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(&mut self.data)
    }

    pub fn value_at(&self, row: usize, col: usize, ch: usize) -> f64 {
        let idx = (row * self.cols + col) * self.channels + ch;
        match self.scalar_type {
            ScalarType::Uint8 => self.data[idx] as f64,
            ScalarType::Int32 => {
                let s: &[i32] = bytemuck::cast_slice(&self.data);
                s[idx] as f64
            }
            ScalarType::Float32 => {
                let s: &[f32] = bytemuck::cast_slice(&self.data);
                s[idx] as f64
            }
            ScalarType::Float64 => {
                let s: &[f64] = bytemuck::cast_slice(&self.data);
                s[idx]
            }
        }
    }

    // -- Conversions ----------------------------------------------------------

    /// Convert RGB → grayscale (1 channel). Weights: 0.299R + 0.587G + 0.114B.
    /// Returns a new Mat with the same scalar type but 1 channel.
    pub fn cvt_color_gray(&self) -> Mat {
        assert_eq!(self.channels, 3, "cvt_color_gray requires 3-channel input");
        let n = self.rows * self.cols;
        match self.scalar_type {
            ScalarType::Uint8 => {
                let mut out = vec![0u8; n];
                for i in 0..n {
                    let r = self.data[i * 3] as f32;
                    let g = self.data[i * 3 + 1] as f32;
                    let b = self.data[i * 3 + 2] as f32;
                    out[i] = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
                let mut m = Mat {
                    data: out,
                    data_ptr: 0, data_len: 0,
                    rows: self.rows, cols: self.cols,
                    channels: 1,
                    scalar_type: ScalarType::Uint8,
                    name: self.name.clone(),
                };
                m.sync(); m
            }
            ScalarType::Float32 => {
                let src: &[f32] = bytemuck::cast_slice(&self.data);
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    out[i] = 0.299 * src[i * 3] + 0.587 * src[i * 3 + 1] + 0.114 * src[i * 3 + 2];
                }
                let mut m = Mat {
                    data: bytemuck::cast_slice(&out).to_vec(),
                    data_ptr: 0, data_len: 0,
                    rows: self.rows, cols: self.cols,
                    channels: 1,
                    scalar_type: ScalarType::Float32,
                    name: self.name.clone(),
                };
                m.sync(); m
            }
            _ => panic!("cvt_color_gray: unsupported type {:?}", self.scalar_type),
        }
    }

    /// Convert scalar type, with optional scale: dst = src * scale.
    /// E.g. Uint8 → Float32 with scale=1.0/255.0 to normalize to [0,1].
    pub fn convert_to(&self, dst_type: ScalarType, scale: f64) -> Mat {
        let n = self.rows * self.cols * self.channels;
        let mut out_data = vec![0u8; n * dst_type.size()];

        // Read each element as f64
        for i in 0..n {
            let val = match self.scalar_type {
                ScalarType::Uint8 => self.data[i] as f64,
                ScalarType::Int32 => {
                    let s: &[i32] = bytemuck::cast_slice(&self.data);
                    s[i] as f64
                }
                ScalarType::Float32 => {
                    let s: &[f32] = bytemuck::cast_slice(&self.data);
                    s[i] as f64
                }
                ScalarType::Float64 => {
                    let s: &[f64] = bytemuck::cast_slice(&self.data);
                    s[i]
                }
            } * scale;

            // Write as dst type
            match dst_type {
                ScalarType::Uint8 => {
                    out_data[i] = val.round().clamp(0.0, 255.0) as u8;
                }
                ScalarType::Int32 => {
                    let dst: &mut [i32] = bytemuck::cast_slice_mut(&mut out_data);
                    dst[i] = val.round() as i32;
                }
                ScalarType::Float32 => {
                    let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut out_data);
                    dst[i] = val as f32;
                }
                ScalarType::Float64 => {
                    let dst: &mut [f64] = bytemuck::cast_slice_mut(&mut out_data);
                    dst[i] = val;
                }
            }
        }

        let mut m = Mat {
            data: out_data,
            data_ptr: 0, data_len: 0,
            rows: self.rows, cols: self.cols,
            channels: self.channels,
            scalar_type: dst_type,
            name: self.name.clone(),
        };
        m.sync();
        m
    }

    // -- Resize (called by gpu_matrix.download) -------------------------------

    pub fn resize(
        &mut self,
        rows: usize,
        cols: usize,
        scalar_type: ScalarType,
        channels: usize,
    ) {
        self.rows = rows;
        self.cols = cols;
        self.scalar_type = scalar_type;
        self.channels = channels;
        let size = rows * cols * channels * scalar_type.size();
        self.data.resize(size, 0);
        self.sync();
    }

    // -- Serialization (for .gmat files) --------------------------------------

    fn debug_dir() -> &'static str {
        "/tmp/gmat_debug"
    }

    pub fn save_debug(&self) {
        self.save_to_dir(Self::debug_dir());
    }

    pub fn save_to_dir(&self, dir: &str) {
        fs::create_dir_all(dir).expect("Failed to create debug directory");

        let path = Path::new(dir).join(format!("{}.gmat", self.name));
        let mut file = fs::File::create(&path).expect("Failed to create debug file");

        file.write_all(b"GMAT").unwrap();
        file.write_all(&(self.rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(self.cols as u32).to_le_bytes()).unwrap();
        file.write_all(&(self.channels as u32).to_le_bytes()).unwrap();

        let type_id: u32 = match self.scalar_type {
            ScalarType::Uint8 => 0,
            ScalarType::Int32 => 1,
            ScalarType::Float32 => 2,
            ScalarType::Float64 => 3,
        };
        file.write_all(&type_id.to_le_bytes()).unwrap();

        let name_bytes = self.name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes()).unwrap();
        file.write_all(name_bytes).unwrap();

        file.write_all(&self.data).unwrap();
    }

    pub fn load_from_file(path: &str) -> Self {
        let bytes = fs::read(path).expect("Failed to read .gmat file");
        Self::from_gmat_bytes(&bytes)
    }

    pub fn from_gmat_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 24, "File too small");
        assert_eq!(&bytes[0..4], b"GMAT", "Invalid magic");

        let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let cols = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let channels = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let type_id = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let name_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;

        let scalar_type = match type_id {
            0 => ScalarType::Uint8,
            1 => ScalarType::Int32,
            2 => ScalarType::Float32,
            3 => ScalarType::Float64,
            _ => panic!("Unknown scalar type: {}", type_id),
        };

        let name = String::from_utf8(bytes[24..24 + name_len].to_vec()).unwrap();
        let data_start = 24 + name_len;
        let data = bytes[data_start..].to_vec();

        let mut m = Mat {
            data,
            data_ptr: 0,
            data_len: 0,
            rows,
            cols,
            channels,
            scalar_type,
            name,
        };
        m.sync();
        m
    }
}
