/// Scalar element types — mirrors the Metal version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    Uint8,
    Int32,
    Float32,
    Float64, // no hardware support on most GPUs — use with care
}

impl ScalarType {
    pub fn size(self) -> usize {
        match self {
            ScalarType::Uint8 => 1,
            ScalarType::Int32 => 4,
            ScalarType::Float32 => 4,
            ScalarType::Float64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ScalarType::Uint8 => "uint8",
            ScalarType::Int32 => "int32",
            ScalarType::Float32 => "float32",
            ScalarType::Float64 => "float64",
        }
    }
}
