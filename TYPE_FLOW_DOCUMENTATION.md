# Type Flow Documentation for Graph and Vector Data

## Data Flow Overview

### 1. UnifiedTorchModel.default_generator()
**Input:**
- `dataset`: DeepChem Dataset
- For graph data: Dataset.X contains `GraphData` objects (as numpy array with object dtype)

**Output:**
- Generator yielding: `(inputs, outputs, weights)`
  - `inputs`: `[BatchMolGraph]` (list with single BatchMolGraph) for graph data
  - `outputs`: `[y_b]` (numpy array or list)
  - `weights`: `[w_b]` (numpy array or list)

---

### 2. UnifiedTorchModel._prepare_batch()
**Input:**
- `batch`: `Tuple[List, List, List]`
  - For graph: `([BatchMolGraph], [y_b], [w_b])`
  - For vector: `([X_tensor], [y_b], [w_b])` (from parent)

**Output:**
- For `encoder_type="dmpnn"`: 
  - Returns: `(BatchMolGraph, List[torch.Tensor], List[torch.Tensor])`
  - `BatchMolGraph` is moved to device
- For `encoder_type="identity"`: 
  - Returns: `(torch.Tensor, List[torch.Tensor], List[torch.Tensor])`
  - Uses parent's `_prepare_batch()`

---

### 3. UnifiedModel.forward()
**Input:**
- `x`: 
  - For `encoder_type="identity"`: `torch.Tensor` of shape `(batch_size, n_features)`
  - For `encoder_type="dmpnn"`: `BatchMolGraph` object
- `V_d`: `Optional[torch.Tensor]` (for graph encoder, currently unused)
- `X_d`: `Optional[torch.Tensor]` (for graph encoder, currently unused)

**Processing:**
1. **Encoding:**
   - If `encoder_type="dmpnn"`: `encoded = self.encoder(x)` where `x` is `BatchMolGraph`
   - If `encoder_type="identity"`: `encoded = self.encoder(x)` where `x` is `torch.Tensor`
   
2. **After encoding:**
   - `encoded`: `torch.Tensor` of shape `(batch_size, encoder_dim)`
   - **IMPORTANT**: After this point, `x` is still the original input (BatchMolGraph or tensor)
   - Use `encoded.shape[0]` or `output.shape[0]` to get batch size, NOT `x.shape[0]`

3. **FFN:**
   - `output = self.ffn(encoded)` → `torch.Tensor` of shape `(batch_size, output_dim)`

**Output:**
- Varies by `model_type`:
  - `"baseline"`: `torch.Tensor` of shape `(batch_size, n_tasks)`
  - `"mc_dropout"`: `(mean, var, packed)` where each is `torch.Tensor`
  - `"evidential"`: 
    - Classification: `(prob, alpha, aleatoric, epistemic)` where each is `torch.Tensor`
    - Regression: `(mu, v, alpha, beta)` where each is `torch.Tensor`

---

### 4. DMPNNEncoder.forward()
**Input:**
- `bmg`: `BatchMolGraph` object
- `V_d`: `Optional[torch.Tensor]` (currently unused)
- `X_d`: `Optional[torch.Tensor]` (currently unused)

**Output:**
- `torch.Tensor` of shape `(batch_size, hidden_dim)`
- Where `batch_size` = number of graphs in the batch

---

## Problem Identified

**Location:** `nn_baseline.py` line 812 (and similar lines 814-815)

**Issue:** 
```python
prob = prob.view(x.shape[0], -1)  # ERROR: x is BatchMolGraph, has no .shape
```

**Root Cause:**
- After encoding, `x` is still the original input (BatchMolGraph for graph data)
- The code tries to use `x.shape[0]` to get batch size
- `BatchMolGraph` objects don't have a `.shape` attribute

**Solution:**
- Use `encoded.shape[0]` or `output.shape[0]` instead of `x.shape[0]`
- After encoding, `encoded` is always a `torch.Tensor` with shape `(batch_size, encoder_dim)`

---

## Type Summary Table

| Component | Input Type | Output Type |
|-----------|-----------|-------------|
| `UnifiedTorchModel.default_generator()` (graph) | Dataset with GraphData | Generator: `([BatchMolGraph], [y], [w])` |
| `UnifiedTorchModel._prepare_batch()` (graph) | `([BatchMolGraph], [y], [w])` | `(BatchMolGraph, List[Tensor], List[Tensor])` |
| `UnifiedModel.forward()` (graph) | `BatchMolGraph` | `torch.Tensor` (varies by model_type) |
| `DMPNNEncoder.forward()` | `BatchMolGraph` | `torch.Tensor` `(batch_size, hidden_dim)` |
| `UnifiedModel.forward()` (vector) | `torch.Tensor` `(batch_size, n_features)` | `torch.Tensor` (varies by model_type) |

---

## Key Points

1. **After encoding, always use `encoded` or `output` tensors for shape operations, NOT the original input `x`**
2. **`BatchMolGraph` objects don't have `.shape` attribute** - they contain tensors (V, E, edge_index, etc.)
3. **Batch size can be obtained from:**
   - `encoded.shape[0]` (after encoding)
   - `output.shape[0]` (after FFN)
   - `BatchMolGraph.batch.max().item() + 1` (for graph data, but not recommended)
