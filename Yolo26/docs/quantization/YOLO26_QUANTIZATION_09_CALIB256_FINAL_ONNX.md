# YOLO26 Advanced PTQ: Calibration 256 Final ONNX

## Scope

- Source model: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt`
- Dataset: `/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml`
- Input size: 1024
- Quantization: W8A8
- PTQ calibration: 256 train images, hybrid selection, seed 0
- Hybrid selection: 50% seeded random and 50% class/object-size coverage
- Reconstruction: 20,000 updates per block or layer
- ORT runtime calibration: 512 train images, hybrid selection, seed 0
- Test-set validation: not run; the user will validate the final files

The 256-image reconstruction setting follows the object-detection calibration size used by BRECQ and QDrop experiments. The earlier 64-image outputs remain comparison artifacts.

## Memory-safe reconstruction

At 1024 resolution, keeping 256 inputs and all captured intermediate features on a 16 GB V100 is unsafe. The AdaRound, BRECQ, and QDrop runners were changed to:

1. Keep calibration images and captured features in CPU memory.
2. Transfer only the randomly selected reconstruction sample to the GPU.
3. Preserve all 256 samples as reconstruction candidates.
4. Keep the original 20,000 optimization updates.

This changes storage and transfer behavior only; it does not reduce the calibration set or optimization count.

## Algorithm runs

### BRECQ

- GPU: physical GPU 2
- Reconstruction: 22 YOLO backbone/neck blocks
- Updates: 20,000 per block
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint: `best_1024_brecq_w8a8_calib256_iter20000_mqbench.pt`

### QDrop

- GPU: physical GPU 3
- Reconstruction: 22 YOLO backbone/neck blocks
- Updates: 20,000 per block
- Drop probability: 0.5
- Activation scale learning rate: `4e-5`
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint: `best_1024_qdrop_w8a8_calib256_iter20000_mqbench.pt`

### AdaRound

- GPU: physical GPU 4
- Reconstruction: 175 Conv layers
- Updates: 20,000 per layer
- Detect-head quantizers excluded: 64
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint metadata verified: calibration 256, updates 20,000, layers 175
- MQBench checkpoint: `best_1024_adaround_w8a8_calib256_iter20000_mqbench.pt`

## Final CPU INT8 ONNX files

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_calib256_iter20000_fullops.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_calib256_iter20000_fullops.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_calib256_iter20000_fullops.onnx
```

Each file is approximately 21 MB. The ORT conversion reported the following graph structure for every file:

- QLinearConv: 175
- FP32 Conv retained: 32
- QLinearAdd: 40
- QLinearConcat: 28
- QLinearMul: 123
- QLinearSigmoid: 110
- QuantizeLinear: 4
- DequantizeLinear: 8

All three final files successfully created an ONNX Runtime CPU session and completed one smoke inference with one output tensor.

## QDQ GPU/CPU variants

QOperator models were slow on CUDA because ONNX Runtime partitioned unsupported `QLinear` operators and inserted host/device copies. The same reconstructed FP32 ONNX inputs were therefore converted again with ORT static QDQ while keeping the Detect head in FP32.

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_calib256_iter20000_qdq_nohead.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_calib256_iter20000_qdq_nohead.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_calib256_iter20000_qdq_nohead.onnx
```

Common graph audit:

- File size: about 22.38 MB
- Conv: 207
- QLinearConv: 0
- QuantizeLinear: 516
- DequantizeLinear: 866
- Detect-head Conv kept in FP32: 32
- ONNX checker: passed
- CPUExecutionProvider smoke inference: passed
- CUDAExecutionProvider smoke inference on physical GPU 7: passed

The exporter now accepts `--quant-format qoperator|qdq`; its default remains `qoperator` for backward compatibility.

## FP16 GPU variants

QOperator and QDQ are two ways to represent INT8 quantization in an ONNX graph. FP16 is not a third INT8 representation. It is a floating-point data type and a separate reduced-precision deployment path.

The FP16 models were created from the reconstructed `rounded_fp32.onnx` files, not from the QDQ files:

```text
MQBench reconstruction
  -> rounded FP32 ONNX
  -> internal weights and operators converted to FP16
  -> FP32 model input/output retained for Ultralytics compatibility
```

Final FP16 files:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_calib256_iter20000_fp16.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_calib256_iter20000_fp16.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_calib256_iter20000_fp16.onnx
```

Common graph audit:

- File size: about 38.12 MB
- Conv: 207
- QuantizeLinear: 0
- DequantizeLinear: 0
- QLinearConv: 0
- FP16 initializers: 414
- FP32 initializers: 0
- Model input/output: FP32
- ONNX checker: passed
- CUDAExecutionProvider smoke inference on physical GPU 7: passed
- Output shape: `(1, 300, 6)`

Only the reconstructed weight result remains relevant in FP16. INT8 activation scales and QDrop activation-quantization behavior are not used by the final FP16 graph.

## Measured deployment results

The following values are user-recorded end-to-end results from `MODEL_GPU-CPU_CHECK_REPEAT.py`. They include model loading behavior, image preprocessing, inference, and postprocessing characteristics of the selected runtime; they are not isolated kernel benchmarks.

| Deployment model | GPU FPS | CPU FPS | Accuracy/notes |
|---|---:|---:|---|
| Original FP32 `.pt` | 49.98 | 6.18 | Performance metric 0.777, recorded on the user's local system |
| QOperator INT8 ONNX | about 6-8 in controlled server checks | about 8-10 | CUDA `QLinear` support was poor; host/device copies dominated |
| QDQ INT8 ONNX | about 33 for the new MQBench variants | about 8-10 | Faster than QOperator, but Q/DQ and CUDA memory-copy overhead remained |
| Reconstructed FP16 ONNX | about 60 | about 8.6-9.4 | Best observed GPU throughput |

Exact FPS depends on the GPU, GPU load, ONNX Runtime version, provider selection, warm-up, and benchmark image set. Comparisons should use the same machine, provider, script, warm-up, and dataset split.

## Why the FP16 model was fastest on GPU

1. Tesla V100 has an efficient FP16 execution path.
2. FP16 weights require about half the storage and memory bandwidth of FP32 weights.
3. The FP16 graph contains no Q/DQ or `QLinear` boundaries.
4. CUDA can execute the 207 Conv nodes as a continuous floating-point graph.
5. QDQ and QOperator models introduced provider partitioning and memory-transfer overhead in ONNX Runtime CUDA.

INT8 is not inherently slower than FP16. It was slower in this experiment because ONNX Runtime CUDA did not lower the tested INT8 graphs into an efficient end-to-end GPU INT8 execution path. A runtime with native INT8 graph fusion, such as TensorRT, may produce a different result.

## Concept summary

| Category | Examples | Meaning |
|---|---|---|
| Quantization/reconstruction algorithm | AdaRound, BRECQ, QDrop | Chooses weights/scales that reduce low-precision accuracy loss |
| Numeric data type | FP32, FP16, INT8 | Determines how numbers are represented |
| INT8 ONNX representation | QDQ, QOperator | Determines how INT8 quantization is encoded in the graph |
| Execution runtime | PyTorch, ONNX Runtime CPU/CUDA, TensorRT | Determines which hardware kernels actually run |

QOperator uses explicit integer operators such as `QLinearConv`. It was the slowest CUDA path because unsupported or inefficient `QLinear` operators caused graph partitioning and memory copies.

QDQ keeps standard Conv operators and adds `QuantizeLinear`/`DequantizeLinear` boundaries. CUDA handled it more flexibly than QOperator, but hundreds of Q/DQ boundaries and added memory copies still prevented FP32-level GPU throughput.

FP16 uses ordinary floating-point Conv operators with FP16 internal weights. It is neither QDQ nor QOperator. Its high GPU speed came from the V100 FP16 execution path and the absence of integer quantization boundaries.

## Presentation narrative

1. Apply AdaRound, BRECQ, and QDrop reconstruction to preserve low-precision accuracy.
2. Export QOperator INT8 ONNX and identify severe CUDA slowdown from `QLinear` support and memory copies.
3. Export QDQ INT8 ONNX and recover substantial GPU throughput, while observing remaining Q/DQ overhead.
4. Evaluate a separate FP16 reduced-precision ONNX path from the reconstructed FP32 weights.
5. Obtain the best GPU result with FP16 while retaining CPU results around the measured INT8 range.

Recommended conclusion:

> The initial GPU slowdown was not caused by the quantization algorithm alone. It was primarily caused by how the INT8 graph was represented and lowered by ONNX Runtime CUDA. QDQ improved runtime compatibility over QOperator, but FP16 produced the best GPU result because it used the V100's efficient floating-point path without Q/DQ or QLinear overhead.
