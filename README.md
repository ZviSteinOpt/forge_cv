# forge_cv

GPU-accelerated computer vision library for Rust, built on [wgpu](https://wgpu.rs).

Cross-platform GPU compute (Metal, Vulkan, DX12, WebGPU) with a CPU-side `Mat` type for image and matrix operations, plus a VS Code / Cursor extension for live matrix visualization during debugging.

## Core types

| Type | Description |
|---|---|
| `Mat` | CPU-side matrix — raw bytes + shape/type metadata |
| `GpuMatrix` | 2D GPU buffer backed by `wgpu::Buffer` |
| `ComputeContext` | Singleton GPU context — device, queue, shader cache, compute dispatch |
| `ScalarType` | Element type enum: `Uint8`, `Int32`, `Float32`, `Float64` |

## Quick start

```rust
use forge_cv::{ComputeContext, GpuMatrix, Mat};

ComputeContext::init();

// Load image → upload to GPU → download back
let src = Mat::imread("image.jpg");
let mut gpu = GpuMatrix::new();
gpu.upload(&src);

let mut dst = Mat::new("result");
gpu.download(&mut dst);
```

### Mat operations

```rust
use forge_cv::{Mat, ScalarType};

let img = Mat::imread("photo.jpg");          // RGB uint8, 3ch
let gray = img.cvt_color_gray();             // grayscale, 1ch
let norm = gray.convert_to(ScalarType::Float32, 1.0 / 255.0);  // [0,1]

// Pixel access
let val = img.value_at(row, col, channel);   // returns f64

// Serialize for debugger
img.save_debug();  // writes /tmp/gmat_debug/<name>.gmat
```

## GMAT Viewer — live matrix inspection

The **GMAT Viewer** VS Code / Cursor extension visualizes `Mat` matrices in real time during debugging. Set a breakpoint, right-click a `Mat` variable, and select **GMAT: View Matrix Variable**.

- Browse multiple matrices in a sidebar
- Supports uint8, int32, float32, float64 with any number of channels
- Linked pan/zoom across images
- Pixel-level inspection with per-channel values on hover
- Zoom in to see individual pixel values overlaid on the image
- Normalize toggle for float data

### Overview — switch between RGB and grayscale

![overview](https://raw.githubusercontent.com/ZviSteinOpt/optFlowMetal/WithMetalFile/gmat-viewer/readme/overview.gif)

### Pixel values — zoom in to inspect individual values

![pixel values](https://raw.githubusercontent.com/ZviSteinOpt/optFlowMetal/WithMetalFile/gmat-viewer/readme/pixel_values.gif)

### Install the extension

```
cursor --install-extension gmat-viewer-0.3.0.vsix
```

## Dependencies

- [wgpu](https://crates.io/crates/wgpu) — cross-platform GPU compute
- [bytemuck](https://crates.io/crates/bytemuck) — safe transmutation for GPU data
- [image](https://crates.io/crates/image) — image file I/O

## License

MIT
