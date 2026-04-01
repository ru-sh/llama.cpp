# MoE Expert Invocation Distribution — Analysis Plan

Measure how Mixture-of-Experts routing differs between **code** and **natural language** domains using the Qwen3.5-35B-A3B model.

Model: `unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS`
Primary domain of interest: **Coding via LLM**

---

## Architecture anchors

| Component | File | Detail |
|---|---|---|
| Expert selection tensor | `src/llama-graph.cpp:1358` | `ggml_argsort_top_k`, named `"ffn_moe_argsort"` |
| Public eval callback | `include/llama.h:352-353` | `cb_eval` / `cb_eval_user_data` |
| Graph named-tensor callbacks | `src/llama-graph.h:523` | `llm_graph_cb` fires `cb(tensor, "ffn_moe_argsort", layer)` |
| Qwen3 MoE forward pass | `src/models/qwen3moe.cpp:88-105` | softmax gating, weight normalisation |
| MoE hparams | `src/llama-hparams.h:47-48` | `n_expert`, `n_expert_used` |
| Callback wired in | `src/llama-context.cpp:1196` | `ggml_backend_sched_set_eval_callback` |

The `cb_eval` hook intercepts every evaluated ggml tensor node. Filtering on tensor name `"ffn_moe_argsort*"` gives the selected expert indices for each token × layer with **no changes to core inference code**.

---

## Phases

### Phase 1 — Instrumentation

**New files:**
```
tools/moe-inspector/
  main.cpp          — CLI entry, arg parsing, llama init, callback wiring
  moe_capture.h     — capture_state struct, callback signature
  moe_capture.cpp   — ask/observe callback, D2H sync (CUDA), JSONL writer
  CMakeLists.txt    — build target llama-moe-inspector
```

**Changes to existing files:**
```
common/arg.cpp      — add --moe-log <file>, --moe-domain <code|nl>
CMakeLists.txt      — add_subdirectory(tools/moe-inspector)
```

**Callback contract:**
- `ask=true` pass: return `true` only for tensors whose name starts with `"ffn_moe_argsort"`
- `ask=false` pass: memcpy `tensor->data` (shape `[n_expert_used, n_tokens_in_batch]`) to host; append records

**Output format** — JSONL, one record per token per layer:
```json
{"token_id": 1234, "pos": 7, "domain": "code", "layer": 3, "experts": [5, 11]}
```

**CUDA caveat:** `ask=false` is the only safe read point; requires `ggml_backend_synchronize` before memcpy when running on a GPU backend.

---

### Phase 2 — Test instrumentation

**New file:** `tools/moe-inspector/tests/test_capture.py`

Test cases:
1. Feed a deterministic 20-token prompt; assert `record_count == n_tokens × n_moe_layers`
2. Assert all expert indices ∈ `[0, n_expert)` (128 for this model)
3. Assert `len(record["experts"]) == n_expert_used` matches model hparams
4. Feed same prompt twice; assert records are bit-identical (determinism check, `--temp 0 --seed 42`)
5. *(CUDA only)* Assert CPU and CUDA runs produce identical expert selections

---

### Phase 3 — Data collection

Two separate inference runs producing independent JSONL files:

```bash
# Code domain
llama-cli -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS \
  --moe-log data/code_routing.jsonl --moe-domain code \
  -f data/code_prompts.txt --temp 0 --seed 42 --no-stream -n 512

# Natural language domain
llama-cli -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS \
  --moe-log data/nl_routing.jsonl --moe-domain nl \
  -f data/nl_prompts.txt --temp 0 --seed 42 --no-stream -n 512
```

**Dataset requirements:**
- Code: ~500 prompts — HumanEval problems, Python stdlib docstrings, LeetCode starters
- NL: ~500 prompts — Wikipedia lead paragraphs, news text, conversational QA
- Length-match: equal total token counts between domains so histograms are comparable
- Store raw prompt files at `tools/moe-inspector/data/`

---

### Phase 4 — Analysis

**New file:** `tools/moe-inspector/analyze.py`
**New file:** `tools/moe-inspector/requirements.txt` — `numpy scipy matplotlib pandas`

#### Steps

1. **Build frequency matrices**
   ```
   hist[domain][layer, expert] = selection count
   ```
   Normalise each row → probability distribution `p[layer, expert]`.

2. **Per-layer JSD**
   ```
   M = 0.5 * (p_code + p_nl)
   JSD[layer] = 0.5 * KL(p_code ∥ M) + 0.5 * KL(p_nl ∥ M)
   ```

3. **Unused expert threshold**
   Expert is "unused" in a domain if selected in < 0.1 % of tokens for that domain.

4. **Outputs**

   | File | Content |
   |---|---|
   | `out/heatmap_code.png` | expert × layer heatmap, code domain |
   | `out/heatmap_nl.png` | expert × layer heatmap, NL domain |
   | `out/heatmap_diff.png` | diverging heatmap: `freq_code − freq_nl` |
   | `out/jsd_by_layer.png` | JSD scalar per layer, line chart |
   | `out/unused_experts.csv` | layer, domain, unused expert ids |
   | `out/report.md` | summary statistics + embedded plots |

---

## File map

```
tools/moe-inspector/
  PLAN.md               ← this file
  CMakeLists.txt
  main.cpp
  moe_capture.h
  moe_capture.cpp
  analyze.py
  requirements.txt
  data/
    code_prompts.txt
    nl_prompts.txt
  tests/
    test_capture.py
  out/                  ← generated, git-ignored
```
