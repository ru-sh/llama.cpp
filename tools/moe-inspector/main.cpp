#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "moe_capture.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(int /*argc*/, char ** argv) {
    fprintf(stderr,
        "usage: %s [llama-options] --moe-log FILE --moe-domain DOMAIN\n"
        "\n"
        "Runs inference and logs per-token MoE expert routing decisions.\n"
        "\n"
        "Required:\n"
        "  --moe-log   FNAME   output JSONL file (routing records, appended)\n"
        "  --moe-domain DOMAIN label for this run, e.g. 'code' or 'nl'\n"
        "\n"
        "All standard llama-cli options are accepted (-m, -p, -f, -n, --temp, etc.).\n"
        "Recommended: --temp 0 --seed 42 for reproducible captures.\n",
        argv[0]);
}

// Read entire file into a string.
static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    // --- strip our own flags before handing the rest to common_params_parse ---
    std::string moe_log_path;
    std::string moe_domain;

    std::vector<char *> fwd_argv;
    fwd_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--moe-log") == 0) && i + 1 < argc) {
            moe_log_path = argv[++i];
        } else if ((strcmp(argv[i], "--moe-domain") == 0) && i + 1 < argc) {
            moe_domain = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argc, argv);
            return 0;
        } else {
            fwd_argv.push_back(argv[i]);
        }
    }

    if (moe_log_path.empty() || moe_domain.empty()) {
        fprintf(stderr, "error: --moe-log and --moe-domain are required\n\n");
        print_usage(argc, argv);
        return 1;
    }

    // --- parse standard llama params ---
    common_params params;
    int fwd_argc = static_cast<int>(fwd_argv.size());
    if (!common_params_parse(fwd_argc, fwd_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // --- attach MoE capture callback before context creation ---
    moe_capture_state capture;
    capture.domain           = moe_domain;
    params.cb_eval           = moe_capture_callback;
    params.cb_eval_user_data = &capture;

    // --- init ---
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    common_sampler * smpl = llama_init->sampler(0);

    if (!model || !ctx || !smpl) {
        fprintf(stderr, "error: failed to initialise model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // --- resolve prompt ---
    if (params.prompt.empty() && !params.prompt_file.empty()) {
        params.prompt = read_file(params.prompt_file);
    }
    if (params.prompt.empty()) {
        fprintf(stderr, "error: no prompt supplied (use -p or -f)\n");
        return 1;
    }

    // --- tokenise ---
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, /*add_special=*/true);
    if (tokens.empty()) {
        fprintf(stderr, "error: tokenisation produced an empty sequence\n");
        return 1;
    }

    const int n_predict = params.n_predict < 0 ? 256 : params.n_predict;
    const int n_ctx     = llama_n_ctx(ctx);

    if (static_cast<int>(tokens.size()) >= n_ctx) {
        fprintf(stderr, "warning: prompt (%zu tokens) >= context size (%d); truncating\n",
                tokens.size(), n_ctx);
        tokens.resize(static_cast<size_t>(n_ctx) - 1);
    }

    // --- prefill ---
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);

    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    // enable logits for last token so we can sample from it
    batch.logits[batch.n_tokens - 1] = true;

    // populate capture state for prefill batch
    capture.batch_token_ids.resize(batch.n_tokens);
    capture.batch_positions.resize(batch.n_tokens);
    for (int i = 0; i < batch.n_tokens; ++i) {
        capture.batch_token_ids[i] = batch.token[i];
        capture.batch_positions[i] = batch.pos[i];
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode failed during prefill\n");
        llama_batch_free(batch);
        return 1;
    }

    int n_past = batch.n_tokens;

    // --- autoregressive decode ---
    const llama_token id_eos = llama_vocab_eos(vocab);

    for (int step = 0; step < n_predict; ++step) {
        const llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, /*accept_grammar=*/true);

        if (new_token == id_eos) break;
        if (n_past >= n_ctx) break;

        common_batch_clear(batch);
        common_batch_add(batch, new_token, n_past, {0}, true);

        // update capture state for this single-token batch
        capture.batch_token_ids = {new_token};
        capture.batch_positions = {n_past};

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed at step %d\n", step);
            break;
        }
        ++n_past;
    }

    llama_batch_free(batch);

    // --- write output ---
    try {
        moe_capture_write_jsonl(capture, moe_log_path);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    fprintf(stderr, "moe-inspector: wrote %zu routing records to %s\n",
            capture.records.size(), moe_log_path.c_str());

    llama_backend_free();
    return 0;
}
