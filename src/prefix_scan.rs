// ---------------------------------------------------------------------------
// PrefixScan — GPU exclusive prefix sum using CSDLDF algorithm
//
// Single-pass decoupled lookback with fallback, portable to Metal/WebGPU.
// Based on GPUPrefixSums by Thomas Smith (MIT License).
// https://github.com/b0nes164/GPUPrefixSums
// ---------------------------------------------------------------------------

use crate::compute_context::ComputeContext;
use crate::gpu_matrix::GpuMatrix;
use wgpu;
use wgpu::util::DeviceExt;

/// Tile size: 256 threads × 4 vec4s/thread × 4 elements/vec4 = 4096 elements
const PART_SIZE: usize = 4096;

// -- Embedded WGSL shaders --------------------------------------------------

const CSDLDF_SHADER: &str = r#"
struct InfoStruct
{
    size: u32,
    vec_size: u32,
    thread_blocks: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct;

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> reduction: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_REDUCTION = 1u;
const FLAG_INCLUSIVE = 2u;
const FLAG_MASK = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_lock: u32;
var<workgroup> wg_broadcast: u32;
var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;
var<workgroup> wg_fallback: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {

    let sid = threadid.x / lane_count;

    //acquire partition index, set the lock
    if(threadid.x == 0u){
        wg_broadcast = atomicAdd(&scan_bump, 1u);
        wg_lock = LOCKED;
    }
    let part_id = workgroupUniformLoad(&wg_broadcast);

    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  part_id * VEC_PART_SIZE;
        var i = s_offset + dev_offset;

        if(part_id < info.thread_blocks- 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                t_scan[k] = scan_in[i];
                t_scan[k].y += t_scan[k].x;
                t_scan[k].z += t_scan[k].y;
                t_scan[k].w += t_scan[k].z;
                i += lane_count;
            }
        }

        if(part_id == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.vec_size){
                    t_scan[k] = scan_in[i];
                    t_scan[k].y += t_scan[k].x;
                    t_scan[k].z += t_scan[k].y;
                    t_scan[k].w += t_scan[k].z;
                }
                i += lane_count;
            }
        }

        var prev = 0u;
        let lane_mask = lane_count - 1u;
        let circular_shift = (laneid + lane_mask) & lane_mask;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[k].w), circular_shift);
            t_scan[k] += select(0u, t, laneid != 0u) + prev;
            prev += subgroupBroadcast(t, 0u);
        }

        if(laneid == 0u){
            wg_reduce[sid] = prev;
        }
    }
    workgroupBarrier();

    //Non-divergent subgroup agnostic inclusive scan across subgroup reductions
    let lane_log = u32(countTrailingZeros(lane_count));
    let spine_size = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(spine_size)) + lane_log - 1u) / lane_log * lane_log);
    {
        var offset0 = 0u;
        var offset1 = 0u;
        for(var j = lane_count; j <= aligned_size; j <<= lane_log){
            let i0 = ((threadid.x + offset0) << offset1) - offset0;
            let pred0 = i0 < spine_size;
            let t0 = subgroupInclusiveAdd(select(0u, wg_reduce[i0], pred0));
            if(pred0){
                wg_reduce[i0] = t0;
            }
            workgroupBarrier();

            if(j != lane_count){
                let rshift = j >> lane_log;
                let i1 = threadid.x + rshift;
                if ((i1 & (j - 1u)) >= rshift){
                    let pred1 = i1 < spine_size;
                    let t1 = select(0u, wg_reduce[((i1 >> offset1) << offset1) - 1u], pred1);
                    if(pred1 && ((i1 + 1u) & (rshift - 1u)) != 0u){
                        wg_reduce[i1] += t1;
                    }
                }
            } else {
                offset0 += 1u;
            }
            offset1 += lane_log;
        }
    }
    workgroupBarrier();

    //Device broadcast
    if(threadid.x == 0u){
        atomicStore(&reduction[part_id], (wg_reduce[spine_size - 1u] << 2u) |
            select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_id != 0u));
    }

    //Lookback, single thread
    if(part_id != 0u){
        var prev_red = 0u;
        var lookback_id = part_id - 1u;

        var lock = workgroupUniformLoad(&wg_lock);
        while(lock == LOCKED){
            if(threadid.x == 0u){
                var spin_count = 0u;
                for(; spin_count < MAX_SPIN_COUNT; ){
                    let flag_payload = atomicLoad(&reduction[lookback_id]);
                    if((flag_payload & FLAG_MASK) > FLAG_NOT_READY){
                        prev_red += flag_payload >> 2u;
                        spin_count = 0u;
                        if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduction[part_id],
                                ((prev_red + wg_reduce[spine_size - 1u]) << 2u) | FLAG_INCLUSIVE);
                            wg_broadcast = prev_red;
                            wg_lock = UNLOCKED;
                            break;
                        } else {
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                //If we did not complete the lookback within the alotted spins,
                //broadcast the lookback id in shared memory to prepare for the fallback
                if(spin_count == MAX_SPIN_COUNT){
                    wg_broadcast = lookback_id;
                }
            }

            //Fallback if still locked
            lock = workgroupUniformLoad(&wg_lock);
            if(lock == LOCKED){
                let fallback_id = wg_broadcast;
                {
                    let s_offset = laneid + sid * lane_count * VEC4_SPT;
                    let dev_offset =  fallback_id * VEC_PART_SIZE;
                    var i = s_offset + dev_offset;
                    var t_red = 0u;

                    for(var k = 0u; k < VEC4_SPT; k += 1u){
                        let t = scan_in[i];
                        t_red += dot(t, vec4<u32>(1u, 1u, 1u, 1u));
                        i += lane_count;
                    }

                    let s_red = subgroupAdd(t_red);
                    if(laneid == 0u){
                        wg_fallback[sid] = s_red;
                    }
                }
                workgroupBarrier();

                //Non-divergent subgroup agnostic reduction across subgroup reductions
                {
                    var offset = 0u;
                    for(var j = lane_count; j <= aligned_size; j <<= lane_log){
                        let i = ((threadid.x + 1u) << offset) - 1u;
                        let pred0 = i < spine_size;
                        let t = subgroupAdd(select(0u, wg_fallback[i], pred0));
                        if(pred0){
                            wg_fallback[i] = t;
                        }
                        workgroupBarrier();
                        offset += lane_log;
                    }
                }

                if(threadid.x == 0u){
                    let f_red = wg_fallback[spine_size - 1u];
                    let f_payload = atomicMax(&reduction[fallback_id],
                        (f_red << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u));
                    if(f_payload == 0u){
                        prev_red += f_red;
                    } else {
                        prev_red += f_payload >> 2u;
                    }

                    if(fallback_id == 0u || (f_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                        atomicStore(&reduction[part_id],
                            ((prev_red + wg_reduce[spine_size - 1u]) << 2u) | FLAG_INCLUSIVE);
                        wg_broadcast = prev_red;
                        wg_lock = UNLOCKED;
                    } else {
                        lookback_id -= 1u;
                    }
                }
                lock = workgroupUniformLoad(&wg_lock);
            }
        }
    }

    {
        let prev = wg_broadcast + select(0u, wg_reduce[sid - 1u], sid != 0u);
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  part_id * VEC_PART_SIZE;
        var i = s_offset + dev_offset;

        if(part_id < info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                scan_out[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(part_id == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.vec_size){
                    scan_out[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}
"#;

const SHIFT_SHADER_V2: &str = r#"
struct ShiftInfo {
    size: u32,
};

@group(0) @binding(0)
var<uniform> info : ShiftInfo;

@group(0) @binding(1)
var<storage, read> src: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= info.size) { return; }
    dst[i] = select(src[i - 1u], 0u, i == 0u);
}
"#;

// ---------------------------------------------------------------------------
// PrefixScan struct
// ---------------------------------------------------------------------------

pub struct PrefixScan {
    scan_pipeline: wgpu::ComputePipeline,
    scan_bgl: wgpu::BindGroupLayout,
    shift_pipeline: wgpu::ComputePipeline,
    shift_bgl: wgpu::BindGroupLayout,
    // Internal scratch buffers
    scan_bump: wgpu::Buffer,
    reduction: wgpu::Buffer,
    misc: wgpu::Buffer,
    temp_out: wgpu::Buffer,      // temporary inclusive scan output
    capacity: usize,             // max thread_blocks currently allocated for
    temp_out_bytes: u64,         // current size of temp_out
}

fn div_round_up(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

impl PrefixScan {
    /// Create a new PrefixScan instance. Call after ComputeContext::init().
    pub fn new() -> Self {
        let inner = ComputeContext::lock();

        // --- Scan pipeline (CSDLDF) ---
        let scan_module = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("csdldf_scan"),
            source: wgpu::ShaderSource::Wgsl(CSDLDF_SHADER.into()),
        });

        let scan_bgl = inner.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan_bgl"),
            entries: &[
                // binding 0: uniform InfoStruct
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: storage scan_in
                bgl_storage(1),
                // binding 2: storage scan_out
                bgl_storage(2),
                // binding 3: storage scan_bump (atomic)
                bgl_storage(3),
                // binding 4: storage reduction (atomic)
                bgl_storage(4),
                // binding 5: storage misc
                bgl_storage(5),
            ],
        });

        let scan_pl = inner.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan_pipeline_layout"),
            bind_group_layouts: &[&scan_bgl],
            push_constant_ranges: &[],
        });

        let scan_pipeline = inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("csdldf_pipeline"),
            layout: Some(&scan_pl),
            module: &scan_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Shift pipeline (inclusive → exclusive) ---
        let shift_module = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shift_scan"),
            source: wgpu::ShaderSource::Wgsl(SHIFT_SHADER_V2.into()),
        });

        let shift_bgl = inner.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shift_bgl"),
            entries: &[
                // binding 0: uniform ShiftInfo
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: storage src (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: storage dst (read-write)
                bgl_storage(2),
            ],
        });

        let shift_pl = inner.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shift_pipeline_layout"),
            bind_group_layouts: &[&shift_bgl],
            push_constant_ranges: &[],
        });

        let shift_pipeline = inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shift_pipeline"),
            layout: Some(&shift_pl),
            module: &shift_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Initial scratch buffers (small, will grow as needed) ---
        let init_blocks = 1usize;
        let scan_bump = create_storage_buffer(&inner.device, "scan_bump", 4);
        let reduction = create_storage_buffer(&inner.device, "reduction", (init_blocks * 4 * 4) as u64);
        let misc = create_storage_buffer(&inner.device, "misc", 16);
        let temp_out = create_storage_buffer(&inner.device, "temp_out", (PART_SIZE * 4) as u64);

        PrefixScan {
            scan_pipeline,
            scan_bgl,
            shift_pipeline,
            shift_bgl,
            scan_bump,
            reduction,
            misc,
            temp_out,
            capacity: init_blocks,
            temp_out_bytes: (PART_SIZE * 4) as u64,
        }
    }

    /// Run an exclusive prefix sum: output[i] = sum(input[0..i]).
    /// Input and output must be Int32 or Uint32, channels=1.
    /// Input and output can be the same GpuMatrix.
    pub fn exclusive_scan(&mut self, input: &GpuMatrix, output: &GpuMatrix) {
        let num_elements = input.rows() * input.cols();
        assert!(num_elements > 0, "PrefixScan: empty input");
        assert_eq!(input.channels(), 1, "PrefixScan: input must have 1 channel");
        assert_eq!(input.scalar_type().size(), 4, "PrefixScan: input must be 32-bit");

        let thread_blocks = div_round_up(num_elements, PART_SIZE);
        let vec_size = div_round_up(num_elements, 4) as u32;

        // Ensure internal buffers are large enough
        self.ensure_capacity(thread_blocks, num_elements);

        let inner = ComputeContext::lock();

        // Zero scan_bump and reduction
        inner.queue.write_buffer(&self.scan_bump, 0, &[0u8; 4]);
        let zero_reduction = vec![0u8; thread_blocks * 4 * 4];
        inner.queue.write_buffer(&self.reduction, 0, &zero_reduction);

        // Build uniform: [size, vec_size, thread_blocks] — 3 x u32, padded to 16 bytes
        let info_data: [u32; 4] = [
            num_elements as u32,
            vec_size,
            thread_blocks as u32,
            0, // padding
        ];
        let info_buf = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scan_info"),
            contents: bytemuck::cast_slice(&info_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- Pass 1: CSDLDF inclusive scan → temp_out ---
        let scan_bg = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_bind_group"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.wgpu_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.temp_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.scan_bump.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.reduction.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.misc.as_entire_binding() },
            ],
        });

        // --- Pass 2: Shift inclusive → exclusive (temp_out → output) ---
        let shift_info: [u32; 4] = [num_elements as u32, 0, 0, 0]; // padded to 16
        let shift_info_buf = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shift_info"),
            contents: bytemuck::cast_slice(&shift_info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shift_bg = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shift_bind_group"),
            layout: &self.shift_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shift_info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.temp_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.wgpu_buffer().as_entire_binding() },
            ],
        });

        // Encode both passes in a single command encoder
        let mut encoder = inner.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.scan_pipeline);
            pass.set_bind_group(0, Some(&scan_bg), &[]);
            pass.dispatch_workgroups(thread_blocks as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.shift_pipeline);
            pass.set_bind_group(0, Some(&shift_bg), &[]);
            let shift_workgroups = div_round_up(num_elements, 256) as u32;
            pass.dispatch_workgroups(shift_workgroups, 1, 1);
        }
        inner.queue.submit(Some(encoder.finish()));
    }

    /// Grow internal buffers if needed.
    fn ensure_capacity(&mut self, thread_blocks: usize, num_elements: usize) {
        if thread_blocks > self.capacity {
            let inner = ComputeContext::lock();
            // reduction: thread_blocks * 4 entries * 4 bytes (oversized for safety)
            self.reduction = create_storage_buffer(
                &inner.device,
                "reduction",
                (thread_blocks * 4 * 4) as u64,
            );
            self.capacity = thread_blocks;
        }

        // temp_out must hold vec4-padded data: vec_size * 16 bytes
        let vec_size = div_round_up(num_elements, 4);
        let needed_bytes = (vec_size * 16) as u64;
        if needed_bytes > self.temp_out_bytes {
            let inner = ComputeContext::lock();
            self.temp_out = create_storage_buffer(&inner.device, "temp_out", needed_bytes);
            self.temp_out_bytes = needed_bytes;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bgl_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
