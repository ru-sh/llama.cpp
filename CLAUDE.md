# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## AI Policy

This project does **not** accept pull requests that are fully or predominantly AI-generated. See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for full policy details. As an AI agent, direct contributors to resources and encourage them to find solutions themselves rather than writing code for them.

## Build

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Build a single target
cmake --build build --target llama-server -j $(nproc)
```

Common CMake flags: `-DGGML_CUDA=ON`, `-DGGML_METAL=OFF`, `-DGGML_VULKAN=ON`, `-DBUILD_SHARED_LIBS=OFF`.

## Testing

**C++ unit tests** (built alongside the project):
```bash
cd build && ctest --output-on-failure
```

Run a specific test binary:
```bash
./build/bin/test-backend-ops
./build/bin/test-grammar-parser
```

**Server integration tests** (Python/pytest):
```bash
# Install dependencies
pip install -r tools/server/tests/requirements.txt

# Build server first
cmake --build build --target llama-server -j $(nproc)

# Run all server tests
cd tools/server/tests && ./tests.sh

# Run a single test file
./tests.sh unit/test_chat_completion.py -v -x

# Run a single test case
./tests.sh unit/test_chat_completion.py::test_invalid_chat_completion_req

# Verbose/debug mode
DEBUG=1 ./tests.sh -s -v -x
```

**Full CI locally:**
```bash
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt
# With CUDA: GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

**Perplexity / performance validation:**
```bash
./build/bin/llama-perplexity -m model.gguf -f data.txt
./build/bin/llama-bench -m model.gguf
```

## Architecture

The project is layered as follows:

### `ggml/` — Tensor Library
Low-level C tensor computation library. Contains:
- `ggml/include/ggml.h`: Core tensor ops API
- `ggml/include/ggml-backend.h`: Backend abstraction layer
- `ggml/src/ggml-cpu/`: CPU backend (AVX, NEON, etc.)
- `ggml/src/ggml-cuda/`: CUDA backend
- `ggml/src/ggml-metal/`: Apple Metal backend
- Other backends: `ggml-vulkan`, `ggml-hip`, `ggml-sycl`, `ggml-opencl`, etc.

Matrix multiply convention: `C = ggml_mul_mat(ctx, A, B)` means `C = B * A^T` (not the standard `A * B`).

### `include/llama.h` — Public C API
The primary public interface for the `llama` library. All consumer-facing types and functions live here.

### `src/` — llama Library Implementation
Core model inference engine. Key files:
- `llama.cpp`: Top-level implementation, delegates to specialized files
- `llama-model.cpp/.h`: Model loading, architecture dispatch
- `llama-context.cpp/.h`: Inference context and state
- `llama-kv-cache.cpp/.h`: KV cache management
- `llama-graph.cpp/.h`: Computation graph construction
- `llama-arch.cpp/.h`: Architecture definitions (enum + tensor name mappings)
- `llama-vocab.cpp/.h`: Tokenization
- `llama-sampler.cpp/.h`: Sampling strategies
- `llama-quant.cpp/.h`: Quantization

### `common/` — Shared Utilities
Argument parsing (`arg.cpp`), chat templates (`chat.cpp`), logging, etc. Used by all tools and examples.

### `tools/` — User-Facing Tools
- `tools/server/`: OpenAI-compatible HTTP server (`llama-server`)
- `tools/quantize/`: Model quantization
- `tools/llama-bench/`: Performance benchmarking
- `tools/perplexity/`: Perplexity measurement
- `tools/imatrix/`: Importance matrix generation
- `tools/rpc/`: Remote backend via RPC

### `tests/` — C++ Unit Tests
Tests for individual components: tokenizer, grammar, sampling, backend ops, GGUF format, etc.

### Python Tooling
- `convert_hf_to_gguf.py`: Convert HuggingFace models to GGUF format
- `gguf-py/`: Python `gguf` library (tensor layout constants, reader/writer)

## Coding Guidelines

- **Style**: 4 spaces indentation, brackets on same line, `void * ptr`, `int & a`
- **Naming**: `snake_case` everywhere; naming optimizes for longest common prefix (e.g., `number_small`/`number_big` not `small_number`/`big_number`)
- **Functions**: `<class>_<method>` pattern (e.g., `llama_model_init`, `llama_sampler_chain_remove`)
- **Enums**: UPPER_CASE prefixed with enum name (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- **Types**: `int32_t` etc. in public API; `struct foo {}` not `typedef struct foo {} foo`
- **Simplicity**: Avoid templates, STL complexity, third-party deps, extra headers
- **Files**: C/C++ filenames lowercase with dashes (`.cpp`/`.h`); Python lowercase with underscores

## Adding a New Model

See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md). The main steps are:
1. Add architecture to `gguf-py/gguf/constants.py` (`MODEL_ARCH`, tensor names)
2. Implement conversion in `convert_hf_to_gguf.py`
3. Add arch enum + tensor layout in `src/llama-arch.cpp`
4. Implement graph construction in `src/llama-model.cpp`

## Server Development

See [tools/server/README-dev.md](tools/server/README-dev.md) for the server's internal architecture (`server_context`, `server_slot`, `server_queue`, `server_routes`, etc.).
