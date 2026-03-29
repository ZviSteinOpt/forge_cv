use forge_cv::{ComputeContext, GpuMatrix, Mat};

fn main() {
    ComputeContext::init();
    println!("Device: {}\n", ComputeContext::device_name());

    // 1. Load image from disk into CPU Mat
    let src = Mat::imread("tetsIm1.jpg");
    println!("Loaded: {}x{} channels={} type={}",
        src.rows, src.cols, src.channels, src.scalar_type.name());

    // 2. Upload to GPU
    let mut gpu = GpuMatrix::new();
    gpu.upload(&src);
    println!("Uploaded to GPU: {}x{} ({} bytes)", gpu.rows(), gpu.cols(), gpu.bytes());

    // 3. Download back to a new CPU Mat
    let mut dst = Mat::new("downloaded");
    gpu.download(&mut dst);
    println!("Downloaded: {}x{} channels={} type={}",
        dst.rows, dst.cols, dst.channels, dst.scalar_type.name());

    // 4. Verify: compare first 10 pixels
    println!("\nPixel comparison (row=0, first 10 cols):");
    let max_col = std::cmp::min(10, src.cols);
    for col in 0..max_col {
        let r_src = src.value_at(0, col, 0) as u8;
        let g_src = src.value_at(0, col, 1) as u8;
        let b_src = src.value_at(0, col, 2) as u8;

        let r_dst = dst.value_at(0, col, 0) as u8;
        let g_dst = dst.value_at(0, col, 1) as u8;
        let b_dst = dst.value_at(0, col, 2) as u8;

        let ok = if r_src == r_dst && g_src == g_dst && b_src == b_dst { "OK" } else { "MISMATCH" };
        println!("  [0,{}] src=({},{},{}) dst=({},{},{}) {}",
            col, r_src, g_src, b_src, r_dst, g_dst, b_dst, ok);
    }

    // 5. Full byte-level comparison
    let match_count = src.as_bytes().iter()
        .zip(dst.as_bytes().iter())
        .filter(|(a, b)| a == b)
        .count();
    let total = src.bytes();
    println!("\nFull comparison: {}/{} bytes match ({}%)",
        match_count, total, match_count * 100 / total);

    // SET BREAKPOINT HERE → inspect `src` and `dst` in the debugger
    println!("\nDone. Set breakpoint here to inspect src/dst in gmat-viewer.");
}
