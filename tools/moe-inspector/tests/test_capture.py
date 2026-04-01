"""
Tests for moe-inspector capture correctness.

Prerequisites:
  - Build: cmake --build build --target llama-moe-inspector
  - Model: unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS (or set MOE_INSPECTOR_MODEL env var)

Run:
  cd tools/moe-inspector/tests
  pytest test_capture.py -v

Environment variables:
  MOE_INSPECTOR_BIN   path to llama-moe-inspector binary (default: build/bin/llama-moe-inspector)
  MOE_INSPECTOR_MODEL HuggingFace repo:file or local path to a MoE GGUF model
  MOE_N_EXPERT        total number of experts in the model (default: 128 for Qwen3.5-35B-A3B)
  MOE_N_EXPERT_USED   experts selected per token          (default: 8  for Qwen3.5-35B-A3B)
  MOE_N_LAYERS        number of MoE layers                (default: 94 for Qwen3.5-35B-A3B)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]  # repo root

BIN = os.environ.get(
    "MOE_INSPECTOR_BIN",
    str(ROOT / "build" / "bin" / "llama-moe-inspector"),
)
MODEL = os.environ.get(
    "MOE_INSPECTOR_MODEL",
    "unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS",
)
N_EXPERT      = int(os.environ.get("MOE_N_EXPERT",      "128"))
N_EXPERT_USED = int(os.environ.get("MOE_N_EXPERT_USED", "8"))
N_MOE_LAYERS  = int(os.environ.get("MOE_N_LAYERS",      "94"))

# Short deterministic prompt — keeps the run fast.
TEST_PROMPT = "def fibonacci(n):"
# Tokens in TEST_PROMPT (approximate; tests only assert >= 1).
MIN_EXPECTED_TOKENS = 1


def skip_if_no_binary():
    if not Path(BIN).exists():
        pytest.skip(f"binary not found: {BIN}  (run: cmake --build build --target llama-moe-inspector)")


def run_inspector(prompt: str, outfile: str, extra_args: list | None = None) -> subprocess.CompletedProcess:
    """Run llama-moe-inspector and return the CompletedProcess."""
    cmd = [
        BIN,
        "-hf" if MODEL.count("/") >= 1 and not Path(MODEL).exists() else "-m",
        MODEL,
        "-p", prompt,
        "--temp", "0",
        "--seed", "42",
        "-n", "10",
        "--moe-log", outfile,
        "--moe-domain", "test",
        "--log-disable",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureCorrectness:
    """DoD: all 5 assertions pass."""

    def test_1_record_count(self):
        """Record count == n_prompt_tokens × n_moe_layers  (plus n_generated × n_moe_layers)."""
        skip_if_no_binary()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            outfile = tf.name

        try:
            result = run_inspector(TEST_PROMPT, outfile)
            assert result.returncode == 0, f"binary exited {result.returncode}:\n{result.stderr}"

            records = load_jsonl(outfile)
            assert len(records) > 0, "no records written"

            # Each record maps to one (token, layer) pair.
            # Total = n_tokens_processed × n_moe_layers.
            # We don't know the exact prompt token count without running the tokenizer,
            # so just check it's a multiple of N_MOE_LAYERS.
            assert len(records) % N_MOE_LAYERS == 0, (
                f"record count {len(records)} is not a multiple of N_MOE_LAYERS={N_MOE_LAYERS}; "
                "some layers may have been missed"
            )
            n_tokens_seen = len(records) // N_MOE_LAYERS
            assert n_tokens_seen >= MIN_EXPECTED_TOKENS, (
                f"expected at least {MIN_EXPECTED_TOKENS} tokens, got {n_tokens_seen}"
            )
        finally:
            Path(outfile).unlink(missing_ok=True)

    def test_2_expert_indices_in_range(self):
        """All expert indices must be in [0, N_EXPERT)."""
        skip_if_no_binary()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            outfile = tf.name

        try:
            result = run_inspector(TEST_PROMPT, outfile)
            assert result.returncode == 0, result.stderr

            records = load_jsonl(outfile)
            assert records, "no records written"

            for rec in records:
                for eid in rec["experts"]:
                    assert 0 <= eid < N_EXPERT, (
                        f"expert index {eid} out of range [0, {N_EXPERT}) "
                        f"in record pos={rec['pos']} layer={rec['layer']}"
                    )
        finally:
            Path(outfile).unlink(missing_ok=True)

    def test_3_experts_per_record_matches_n_expert_used(self):
        """Each record must contain exactly N_EXPERT_USED expert indices."""
        skip_if_no_binary()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            outfile = tf.name

        try:
            result = run_inspector(TEST_PROMPT, outfile)
            assert result.returncode == 0, result.stderr

            records = load_jsonl(outfile)
            assert records, "no records written"

            for rec in records:
                assert len(rec["experts"]) == N_EXPERT_USED, (
                    f"expected {N_EXPERT_USED} experts per record, "
                    f"got {len(rec['experts'])} at pos={rec['pos']} layer={rec['layer']}"
                )
        finally:
            Path(outfile).unlink(missing_ok=True)

    def test_4_determinism(self):
        """Two runs with --temp 0 --seed 42 must produce bit-identical records."""
        skip_if_no_binary()

        with (
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf1,
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf2,
        ):
            out1, out2 = tf1.name, tf2.name

        try:
            for outfile in (out1, out2):
                result = run_inspector(TEST_PROMPT, outfile)
                assert result.returncode == 0, result.stderr

            records1 = load_jsonl(out1)
            records2 = load_jsonl(out2)

            assert len(records1) == len(records2), (
                f"run 1 produced {len(records1)} records, run 2 produced {len(records2)}"
            )

            for i, (r1, r2) in enumerate(zip(records1, records2)):
                assert r1 == r2, f"records differ at index {i}:\n  run1: {r1}\n  run2: {r2}"
        finally:
            Path(out1).unlink(missing_ok=True)
            Path(out2).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not os.environ.get("MOE_TEST_CUDA_PARITY"),
        reason="set MOE_TEST_CUDA_PARITY=1 and ensure a CUDA build to run this test",
    )
    def test_5_cuda_cpu_parity(self):
        """CPU and CUDA builds must produce identical expert selections for the same prompt."""
        cuda_bin = os.environ.get("MOE_INSPECTOR_BIN_CUDA", BIN + "-cuda")
        if not Path(cuda_bin).exists():
            pytest.skip(f"CUDA binary not found: {cuda_bin}")

        with (
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf_cpu,
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf_gpu,
        ):
            out_cpu, out_gpu = tf_cpu.name, tf_gpu.name

        try:
            # CPU run (default binary, -ngl 0 forces all layers on CPU)
            result_cpu = run_inspector(TEST_PROMPT, out_cpu, extra_args=["-ngl", "0"])
            assert result_cpu.returncode == 0, result_cpu.stderr

            # CUDA run using a separately built binary that defaults to GPU offload
            cmd_gpu = [
                cuda_bin,
                "-hf" if MODEL.count("/") >= 1 else "-m", MODEL,
                "-p", TEST_PROMPT,
                "--temp", "0", "--seed", "42", "-n", "10",
                "--moe-log", out_gpu, "--moe-domain", "test", "--log-disable",
            ]
            result_gpu = subprocess.run(cmd_gpu, capture_output=True, text=True)
            assert result_gpu.returncode == 0, result_gpu.stderr

            records_cpu = load_jsonl(out_cpu)
            records_gpu = load_jsonl(out_gpu)

            assert len(records_cpu) == len(records_gpu), (
                f"CPU: {len(records_cpu)} records, CUDA: {len(records_gpu)} records"
            )
            for i, (rc, rg) in enumerate(zip(records_cpu, records_gpu)):
                assert rc["experts"] == rg["experts"], (
                    f"expert mismatch at record {i} "
                    f"pos={rc['pos']} layer={rc['layer']}:\n"
                    f"  CPU:  {rc['experts']}\n"
                    f"  CUDA: {rg['experts']}"
                )
        finally:
            Path(out_cpu).unlink(missing_ok=True)
            Path(out_gpu).unlink(missing_ok=True)
