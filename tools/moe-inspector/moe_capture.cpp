#include "moe_capture.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

// Tensor of interest: result of ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)
// Named by llama_context::graph_get_cb() as "ffn_moe_topk-{layer}" via ggml_format_name.
// Shape: [n_expert_used, n_tokens]  type: GGML_TYPE_I32
// Element [e, t] = index e + t*n_expert_used in the flat buffer.

static const char * MOE_TOPK_PREFIX    = "ffn_moe_topk-";
static const size_t MOE_TOPK_PREFIX_LEN = 13; // strlen("ffn_moe_topk-")

static bool is_moe_topk(const struct ggml_tensor * t) {
    return strncmp(t->name, MOE_TOPK_PREFIX, MOE_TOPK_PREFIX_LEN) == 0;
}

// Parse layer index from "ffn_moe_topk-{N}".
static int layer_from_name(const char * name) {
    const char * p = name + MOE_TOPK_PREFIX_LEN;
    if (*p == '\0') return -1;
    return atoi(p);
}

bool moe_capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // Return true to opt-in to the observe pass for this tensor.
        return is_moe_topk(t);
    }

    // ask == false: tensor is fully evaluated on its backend, safe to read.
    auto * state = reinterpret_cast<moe_capture_state *>(user_data);

    const int layer      = layer_from_name(t->name);
    const int n_exp_used = static_cast<int>(t->ne[0]);
    const int n_tokens   = static_cast<int>(t->ne[1]);

    if (layer < 0 || n_exp_used <= 0 || n_tokens <= 0) {
        return true;
    }

    // ggml_backend_tensor_get is a synchronous blocking call that handles
    // D2H copy transparently for any backend including CUDA.
    const size_t nbytes = static_cast<size_t>(n_exp_used) * n_tokens * sizeof(int32_t);
    std::vector<int32_t> buf(static_cast<size_t>(n_exp_used) * n_tokens);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    const int n_batch = static_cast<int>(state->batch_token_ids.size());

    for (int tok = 0; tok < n_tokens && tok < n_batch; ++tok) {
        moe_record rec;
        rec.token_id = state->batch_token_ids[tok];
        rec.pos      = state->batch_positions[tok];
        rec.layer    = layer;
        rec.experts.resize(n_exp_used);

        // Flat layout: element [e, tok] is at index e + tok*n_exp_used.
        for (int e = 0; e < n_exp_used; ++e) {
            rec.experts[e] = buf[e + tok * n_exp_used];
        }
        state->records.push_back(std::move(rec));
    }

    return true; // returning false would cancel graph execution
}

void moe_capture_write_jsonl(const moe_capture_state & state, const std::string & path) {
    std::ofstream f(path, std::ios::app);
    if (!f) {
        throw std::runtime_error("moe_capture: cannot open output file: " + path);
    }

    for (const auto & rec : state.records) {
        f << "{\"token_id\":"  << rec.token_id
          << ",\"pos\":"       << rec.pos
          << ",\"domain\":\""  << state.domain << "\""
          << ",\"layer\":"     << rec.layer
          << ",\"experts\":[";
        for (size_t i = 0; i < rec.experts.size(); ++i) {
            if (i > 0) f << ',';
            f << rec.experts[i];
        }
        f << "]}\n";
    }
}
