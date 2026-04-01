#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// One routing decision: which experts were selected for one token at one layer.
struct moe_record {
    int32_t              token_id;
    int32_t              pos;
    int32_t              layer;
    std::vector<int32_t> experts;
};

// Shared state between main.cpp and the eval callback.
// Caller must fill batch_token_ids / batch_positions before every llama_decode().
struct moe_capture_state {
    std::string              domain;           // "code" or "nl"
    std::vector<moe_record>  records;          // accumulated routing records
    std::vector<int32_t>     batch_token_ids;  // token ids for current batch  [n_tokens]
    std::vector<int32_t>     batch_positions;  // positions for current batch  [n_tokens]
};

// ggml_backend_sched_eval_callback compatible function.
// Intercepts tensors named "ffn_moe_topk-{layer}" and appends moe_record entries
// to state->records.  Calls ggml_backend_tensor_get for transparent D2H on CUDA.
bool moe_capture_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Append all records in state to a JSONL file at path.
// Each line: {"token_id":N,"pos":N,"domain":"...","layer":N,"experts":[...]}
void moe_capture_write_jsonl(const moe_capture_state & state, const std::string & path);
