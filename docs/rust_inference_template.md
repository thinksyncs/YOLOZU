# Rust Inference Template (Production-Ready)

YOLOZU provides a **minimal Rust inference template** for production deployments requiring memory safety, zero-cost abstractions, and predictable performance.

## Status

**Not yet implemented** — This document serves as a specification and guide for creating a Rust-based inference core when needed.

## Rationale

Rust is a good fit for production inference when:
- Memory safety guarantees are required (no segfaults, data races, or buffer overflows)
- Predictable performance is critical (no GC pauses)
- Cross-compilation and embedded targets are needed
- Integration with C/C++ libraries (ONNXRuntime, TensorRT) is required

## Recommended Approach

When implementing a Rust inference core for YOLOZU:

### 1. Use the Same Output Schema

All runners (Python/C++/Rust) must emit the **same predictions JSON schema**:

```rust
// Minimal structure (see schemas/predictions.schema.json for full spec)
#[derive(Serialize)]
struct PredictionsOutput {
    predictions: Vec<ImagePrediction>,
    meta: Meta,
}

#[derive(Serialize)]
struct ImagePrediction {
    image: String,
    detections: Vec<Detection>,
}

#[derive(Serialize)]
struct Detection {
    class_id: u32,
    score: f32,
    bbox: BBox,
    // Optional: depth, pose, uncertainty, etc.
}

#[derive(Serialize)]
struct BBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}
```

### 2. Backend Integration

Choose one or more inference backends:

#### Option A: ONNXRuntime (via `ort` crate)

```toml
[dependencies]
ort = "2.0"  # ONNXRuntime bindings
```

```rust
use ort::{Environment, Session, Value};

let environment = Environment::default();
let session = Session::builder()?.with_model_from_file("model.onnx")?;
let input_tensor = Value::from_array(environment.allocator(), &input_data)?;
let outputs = session.run(vec![input_tensor])?;
```

**Pros**: Cross-platform, CPU/GPU, easy to use  
**Cons**: Requires ONNXRuntime C++ library (dynamic linking or static build)

#### Option B: TensorRT (via `tensorrt-rs` or custom bindings)

```toml
[dependencies]
tensorrt-sys = "0.x"  # Low-level TensorRT bindings
```

**Pros**: Best performance on NVIDIA GPUs  
**Cons**: Linux + NVIDIA only, complex API

#### Option C: Pure Rust (tract, burn, candle)

```toml
[dependencies]
tract-onnx = "0.21"  # Pure Rust ONNX runtime
# or
burn = "0.13"        # Pure Rust deep learning framework
# or
candle-core = "0.x"  # Lightweight ML framework from HuggingFace
```

**Pros**: No C++ dependencies, easy cross-compilation  
**Cons**: Less mature, may lack operator support or optimizations

### 3. Image I/O and Preprocessing

```toml
[dependencies]
image = "0.25"
ndarray = "0.15"
```

```rust
use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array4};

fn letterbox(img: &DynamicImage, size: (u32, u32)) -> (Array4<f32>, f32, (u32, u32)) {
    // Implement letterbox padding (keep aspect ratio, center crop)
    // Return: (preprocessed_tensor, scale_factor, padding)
    todo!()
}

fn preprocess(path: &str) -> Array4<f32> {
    let img = image::open(path).expect("Failed to load image");
    let (tensor, _, _) = letterbox(&img, (640, 640));
    // Normalize: RGB -> [0,1] or ImageNet stats
    tensor
}
```

### 4. Postprocessing

```rust
fn postprocess(
    outputs: &[Array],
    score_threshold: f32,
    max_detections: usize,
) -> Vec<Detection> {
    // Decode model outputs (logits, boxes, etc.)
    // Apply score threshold
    // Sort by score, take top-k
    todo!()
}
```

### 5. JSON Serialization

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

```rust
use serde::Serialize;
use std::fs::File;
use std::io::Write;

let predictions = PredictionsOutput { /* ... */ };
let json = serde_json::to_string_pretty(&predictions)?;
let mut file = File::create("predictions.json")?;
file.write_all(json.as_bytes())?;
```

### 6. CLI Interface

```toml
[dependencies]
clap = { version = "4.5", features = ["derive"] }
```

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "yolozu-infer-rust")]
struct Args {
    /// Input image path
    #[arg(long)]
    image: String,

    /// ONNX model path
    #[arg(long)]
    model: String,

    /// Output predictions JSON path
    #[arg(long)]
    output: String,

    /// Score threshold (default: 0.01)
    #[arg(long, default_value = "0.01")]
    score_threshold: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load model, run inference, write JSON
    Ok(())
}
```

---

## Project Structure

If you create a separate Rust crate (recommended for clean separation):

```
yolozu-infer-rust/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI entry point
│   ├── inference.rs     # Core inference logic
│   ├── preprocess.rs    # Image preprocessing
│   ├── postprocess.rs   # Output decoding
│   └── schema.rs        # YOLOZU predictions schema
├── README.md
└── LICENSE              # Apache-2.0
```

### Cargo.toml (minimal example)

```toml
[package]
name = "yolozu-infer-rust"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[dependencies]
ort = "2.0"
image = "0.25"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
```

---

## Integration with YOLOZU

Once you have a working Rust inference core:

1. **Submodule**: Add it as a git submodule under `external/yolozu-infer-rust`
2. **Validation**: Run YOLOZU's schema validator:
   ```bash
   yolozu-infer-rust --image test.jpg --model model.onnx --output predictions.json
   python3 tools/validate_predictions.py predictions.json --strict
   ```
3. **Evaluation**: Score predictions in YOLOZU:
   ```bash
   python3 tools/eval_coco.py --predictions predictions.json --dataset data/coco-yolo
   ```

---

## Example: Minimal Stub

```rust
use serde::Serialize;
use std::fs;

#[derive(Serialize)]
struct Predictions {
    predictions: Vec<ImagePred>,
    meta: Meta,
}

#[derive(Serialize)]
struct ImagePred {
    image: String,
    detections: Vec<Detection>,
}

#[derive(Serialize)]
struct Detection {
    class_id: u32,
    score: f32,
    bbox: BBox,
}

#[derive(Serialize)]
struct BBox { cx: f32, cy: f32, w: f32, h: f32 }

#[derive(Serialize)]
struct Meta {
    schema_version: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = Predictions {
        predictions: vec![
            ImagePred {
                image: "test.jpg".into(),
                detections: vec![],  // Empty for stub
            }
        ],
        meta: Meta {
            schema_version: "v1".into(),
        },
    };
    let json = serde_json::to_string_pretty(&output)?;
    fs::write("predictions.json", json)?;
    println!("Wrote predictions.json");
    Ok(())
}
```

Compile and run:
```bash
cargo build --release
./target/release/yolozu-infer-rust
python3 tools/validate_predictions.py predictions.json
```

---

## Performance Considerations

- **Zero-copy**: Use `ndarray` views and avoid unnecessary clones
- **SIMD**: Enable target-specific optimizations (`RUSTFLAGS="-C target-cpu=native"`)
- **Profiling**: Use `cargo flamegraph` or `perf` to identify bottlenecks
- **Async I/O**: For batch processing, use `tokio` or `async-std`

---

## Deployment

Rust binaries are static (if built with `musl` on Linux):

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

Resulting binary is standalone (no runtime dependencies except libc).

---

## References

- **ort (ONNXRuntime)**: https://github.com/pykeio/ort
- **tract (pure Rust ONNX)**: https://github.com/sonos/tract
- **candle (Rust ML)**: https://github.com/huggingface/candle
- **image (Rust image I/O)**: https://github.com/image-rs/image
- **YOLOZU predictions schema**: [`schemas/predictions.schema.json`](../schemas/predictions.schema.json)

---

## When to Implement

Implement a Rust core when:
1. **Safety requirements**: Regulatory/compliance demands memory safety
2. **Embedded targets**: Deploying to ARM/RISC-V/custom hardware
3. **Latency-critical**: Sub-millisecond p99 latency requirements
4. **Existing Rust ecosystem**: Team already uses Rust, wants unified stack

Otherwise, the **C++ template** (`examples/infer_cpp/`) is production-ready and more mature.

---

## Status: Specification Only

This document serves as a **guide** for when a Rust inference core is needed. The C++ template (`examples/infer_cpp/`) is the current production-ready reference implementation.

If you implement a Rust core following this spec, please:
1. Ensure it passes YOLOZU schema validation
2. Benchmark against C++/Python baselines
3. Document any platform-specific quirks
4. Share results back with the community
