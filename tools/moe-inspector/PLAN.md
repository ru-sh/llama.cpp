# MoE Expert Invocation Distribution — Analysis Plan

Measure how Mixture-of-Experts routing differs between **code** and **natural language** domains
using the Qwen3.5-35B-A3B model.

Model: `unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS`
Primary domain of interest: **Coding via LLM**

---

## Architecture anchors

| Component | File | Detail |
|---|---|---|
| Expert selection tensor | `src/llama-graph.cpp:1358` | `ggml_argsort_top_k`, named **`"ffn_moe_topk-{layer}"`** via `ggml_format_name` |
| Public eval callback | `include/llama.h:352-353` | `cb_eval` / `cb_eval_user_data` on `llama_context_params` |
| Graph callback sets name | `src/llama-context.cpp:2200` | `ggml_format_name(cur, "%s-%d", name, il)` |
| Qwen3 MoE forward pass | `src/models/qwen3moe.cpp:88-105` | softmax gating, weight normalisation |
| MoE hparams | `src/llama-hparams.h:47-48` | `n_expert`, `n_expert_used` |
| Callback wired in | `src/llama-context.cpp:1196` | `ggml_backend_sched_set_eval_callback` |

The `cb_eval` hook intercepts every evaluated ggml tensor node. Filtering on tensor name prefix
`"ffn_moe_topk-"` gives the selected expert indices for each token × layer with **no changes to
core inference code**.

Tensor shape: `[n_expert_used, n_tokens_in_batch]`, type `int32`.
Element `[e, t]` is at flat index `e + t * n_expert_used`.

D2H copy uses `ggml_backend_tensor_get` — blocking, handles CUDA transparently.

---

## Phases

### Phase 1 — Instrumentation ✅

**New files:**
```
tools/moe-inspector/
  main.cpp       — CLI, strips --moe-log/--moe-domain before common_params_parse,
                   sets params.cb_eval before common_init_from_params
  moe_capture.h  — moe_record / moe_capture_state structs, callback signature
  moe_capture.cpp — ask/observe callback, ggml_backend_tensor_get D2H, JSONL writer
  CMakeLists.txt  — target llama-moe-inspector linked against common+llama+ggml
  collect.sh      — two-domain collection script
```

**Changes to existing files:**
```
tools/CMakeLists.txt — add_subdirectory(moe-inspector)
.gitignore           — tools/moe-inspector/out/, data/*.jsonl
```

**Notes from implementation:**
- `--moe-log` / `--moe-domain` are stripped from argv before forwarding to
  `common_params_parse`; they are not added to `common/arg.cpp`.
- `params.cb_eval` is set before `common_init_from_params` so the context is
  created with the callback already wired.
- Two compile bugs found and fixed:
  - `print_usage` signature must be `void(int, char**)` to match `common_params_parse`
  - `llama_batch_clear` does not exist; correct function is `common_batch_clear`

**Output format** — JSONL, one record per token per layer:
```json
{"token_id": 1234, "pos": 7, "domain": "code", "layer": 3, "experts": [5, 11]}
```

---

### Phase 2 — Test instrumentation ✅ (written, not yet executed)

**File:** `tools/moe-inspector/tests/test_capture.py`

| # | Assertion | Status |
|---|---|---|
| 1 | `record_count % n_moe_layers == 0` | written |
| 2 | All expert indices ∈ `[0, n_expert)` | written |
| 3 | `len(record["experts"]) == n_expert_used` | written |
| 4 | Two runs with `--temp 0 --seed 42` produce bit-identical records | written |
| 5 | CPU and CUDA runs produce identical selections (opt-in via `MOE_TEST_CUDA_PARITY=1`) | written |

**Execution blocked by:** model download requires network access.

Run when network is available:
```bash
cd tools/moe-inspector/tests
MOE_N_EXPERT=128 MOE_N_EXPERT_USED=8 MOE_N_LAYERS=94 pytest test_capture.py -v
```

---

### Phase 3 — Data collection ✅ (scripted, not yet executed)

**File:** `tools/moe-inspector/collect.sh`

Two runs of `llama-moe-inspector` producing independent JSONL files:

```bash
# from tools/moe-inspector/
./collect.sh --ngl 99       # GPU
./collect.sh                # CPU-only
```

Settings used: `--temp 0 --seed 42 -n 128 -c 2048`.

**Dataset contents:**

| Domain | File | Content |
|---|---|---|
| code | `data/code_prompts.txt` | 120 LeetCode/HumanEval-style Python function stubs |
| nl | `data/nl_prompts.txt` | 100 Wikipedia/science paragraph leads |

**Execution blocked by:** model download requires network access.

Expected outputs: `data/code_routing.jsonl`, `data/nl_routing.jsonl` (git-ignored).

---

### Phase 4 — Analysis ✅ (written, not yet executed)

**File:** `tools/moe-inspector/analyze.py`

```bash
pip install -r requirements.txt
python3 analyze.py \
  --code data/code_routing.jsonl \
  --nl   data/nl_routing.jsonl \
  --out  out/
```

#### Steps

1. **Build frequency matrices** — `hist[layer, expert]` counts, normalised per row to `p[layer, expert]`
2. **Per-layer JSD** — `JSD[layer] = jensenshannon(p_code[layer], p_nl[layer]) ** 2`, range [0, 1]
3. **Unused expert detection** — expert flagged if `p[layer, expert] < 0.001` (< 0.1 % of tokens)

#### Outputs

| File | Content |
|---|---|
| `out/heatmap_code.png` | expert × layer selection frequency, code domain |
| `out/heatmap_nl.png` | same for NL domain |
| `out/heatmap_diff.png` | diverging heatmap: `freq_code − freq_nl` |
| `out/jsd_by_layer.png` | JSD per layer, line chart, threshold line at 0.1 |
| `out/unused_experts.csv` | layer, domain, expert_id, freq |
| `out/report.md` | token counts, JSD table, unused % stats, top-5 biased experts |

`analyze.py` verifies all six output files are non-empty before exit.

---

## Execution status

| Phase | Code | Tests run | Data collected |
|---|---|---|---|
| 1 — Instrumentation | ✅ | — | — |
| 2 — Tests | ✅ written | ❌ needs network | — |
| 3 — Data collection | ✅ scripted | ❌ needs network | ❌ needs network |
| 4 — Analysis | ✅ written | ❌ blocked on phase 3 | ❌ blocked on phase 3 |

---

## File map

```
tools/moe-inspector/
  PLAN.md               ← this file
  CMakeLists.txt
  main.cpp
  moe_capture.h
  moe_capture.cpp
  collect.sh
  analyze.py
  requirements.txt
  data/
    code_prompts.txt
    nl_prompts.txt
    code_routing.jsonl  ← generated, git-ignored
    nl_routing.jsonl    ← generated, git-ignored
  tests/
    test_capture.py
  out/                  ← generated, git-ignored
```
