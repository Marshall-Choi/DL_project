# INT8 rounding contract (Muryeol / NumPy ref / C++ engine)

All paths share one rule:

1. **Quantize (symmetric INT8):** `q = round(x / scale)` then `clip(q, -128, 127)`.
2. **Rounding:** `numpy.round` / PyTorch `torch.round` — **half to even** (IEEE 754 default).
3. **Dequantize:** `x̂ = q * scale` (FP32).
4. **Conv (weight-only PTQ path):** activations stay FP32; weights use fake-quant `ŵ = dequantize(quantize(w))`.
5. **Conv (full INT8 path):** activations are per-channel. Do not use one `scale_in` for a whole layer.  
   `y_fp = sum_c(x_q[c] * w_q[oc, c] * scale_in[c] * scale_w[oc]) + bias_fp`, then ReLU and output requant.  
   The scales that reach 98.72% are `runs/full_int8_recovered/best_ptq_scales.pt` (percentile 99.9).  
   `runs/quant_chen/int8_package/manifest.json` still has the old max-abs scales. Those are not the accuracy result.

C++ must implement step 1 identically; layer-wise dumps live under `runs/quant_chen/layer_dumps/`.

**Source of truth for full INT8:** `quantize/int_ref.py::forward_full_int8` (integer conv path). PyTorch `PTQAConvNet` with `quantize_activations=True` is a fake-quant FP32 simulator and may disagree slightly; benchmarks and bit-exact checks use NumPy + C++.