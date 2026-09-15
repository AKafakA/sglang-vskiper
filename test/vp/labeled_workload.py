"""Labeled real-workload construction and scoring for VP serving tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Optional


DECODE_DATASETS = ("gsm8k", "coqa", "humaneval")
# Broader-decode workloads (v1.3): standalone named workloads only — they are
# deliberately NOT part of DECODE_DATASETS so the decode_mix/mixed composition
# contracts stay byte-stable.
EXTENDED_DECODE_DATASETS = (
    "gsm8k_cot",
    # Owner ruling 2026-09-05 : the two long-OUTPUT math rows, served
    # UNCAPPED in the perf lane (the lm-eval 256-token cap is a quality-lane
    # constant only); the 8-shot gsm8k_cot row is cancelled (short shots gave
    # 110-token answers).
    "gsm8k_cot_zeroshot",
    "minerva_math",
    # Long-context INSTRUCTION QA (owner 2026-09-05): LongBench v1 QA tasks, window-filtered.
    "longbench_qasper",
    "longbench_multifieldqa_en",
    "longbench_hotpotqa",
    "longbench_2wikimqa",
    "longbench_musique",
    "longbench_narrativeqa",
    # Long-context CODING row (owner 2026-09-05): LongBench repo-level completion.
    "longbench_lcc",
    "longbench_repobench-p",
    "ifeval",
    "mmlu_pro",
    "mmlu_pro_cot",
    # Long-context writing row (sealed axis A1-A6, 2026-09-04). Phase is
    # "decode" because these generate 1024 tokens -- ~55% of cell time is
    # decode -- unlike LongBench's gov_report/multi_news in PREFILL_DATASETS,
    # whose 512-token outputs over the same documents are prefill-dominated.
    "scrolls_gov_report",
    "scrolls_summ_screen_fd",
    "scrolls_qmsum",
    # Balanced long-OUTPUT rows (owner, 2026-09-06): the decode
    # mechanism needs decode-heavy rows; context grows with the generation.
    "longbench_write",
    "lca_libgen",
    "lca_libgen_noapi",
    "lca_libgen_official",
    "longwriter6k",
    "livecodebench",
    # Owner (2026-09-08): BBH chain-of-thought (lm-eval bbh_cot_fewshot,
    # 3 fixed CoT shots per task, exact match) = the second win-regime row.
    "bbh_cot",
)
PREFILL_DATASETS = (
    "mmlu",
    "hellaswag",
    "winogrande",
    "gov_report",
    "multi_news",
)
ALL_DATASETS = DECODE_DATASETS + EXTENDED_DECODE_DATASETS + PREFILL_DATASETS
WORKLOAD_DATASETS = {
    **{name: (name,) for name in ALL_DATASETS},
    "decode_mix": DECODE_DATASETS,
    "mixed": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "prefill_core": PREFILL_DATASETS[:3],
    "prefill_long": PREFILL_DATASETS[3:],
    # Sealed writing row: SCROLLS trio, window-FILTERED (never truncated).
    "longctx_writing": (
        "scrolls_gov_report",
        "scrolls_summ_screen_fd",
        "scrolls_qmsum",
    ),
    # Long-context INSTRUCTION QA row: LongBench v1 QA tasks with >=29 window-fit
    # survivors at 8192 (454 unique requests, ~2.3 M prompt tokens, p50 ~5k).
    "longbench_qa": (
        "longbench_qasper",
        "longbench_2wikimqa",
        "longbench_multifieldqa_en",
        "longbench_hotpotqa",
    ),
    # Long-context CODING row: lcc + repobench-p (weights ∝ window-fit survivors, set after measurement).
    "longbench_code": ("longbench_lcc", "longbench_repobench-p"),
    # Balanced long-output rows : one dataset each, window-filtered.
    "longbench_write": ("longbench_write",),
    "lca_libgen": ("lca_libgen",),
    "lca_libgen_noapi": ("lca_libgen_noapi",),
    "lca_libgen_official": ("lca_libgen_official",),
    "longwriter6k": ("longwriter6k",),
    "livecodebench": ("livecodebench",),
    # Quality-line writing row (owner 2026-09-06): the official LongBench-Write
    # test prompts embedded in the LongWriter-6k serving stream, so the test
    # items are served at the row's target QPS inside the real load (a 106-
    # request burst alone never engages the decode band). Weights come
    # from the build config (longbench_write small, its pool exhausts -> every
    # window survivor is included exactly once, no replication).
    "writing_row": ("longwriter6k", "longbench_write"),
    "mixed_25": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_50": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_75": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_all8": DECODE_DATASETS + PREFILL_DATASETS,
}
WORKLOAD_PHASE = {
    **{name: "decode" for name in DECODE_DATASETS},
    **{name: "decode" for name in EXTENDED_DECODE_DATASETS},
    **{name: "prefill" for name in PREFILL_DATASETS},
    "decode_mix": "decode",
    "mixed": "mixed",
    "prefill_core": "prefill",
    "prefill_long": "prefill",
    "longctx_writing": "decode",
    "longbench_qa": "decode",
    "longbench_code": "decode",
    "longbench_write": "decode",
    "lca_libgen": "decode",
    "lca_libgen_noapi": "decode",
    "lca_libgen_official": "decode",
    "longwriter6k": "decode",
    "livecodebench": "decode",
    "writing_row": "decode",
    "mixed_25": "mixed",
    "mixed_50": "mixed",
    "mixed_75": "mixed",
    "mixed_all8": "mixed",
}
DEFAULT_DECODE_WEIGHTS = {
    # Proportional to the MEASURED window-fit survivors (330/145/38 of
    # 972/338/272) so the three pools drain together under sample_items'
    # weighted draw; equal weights would exhaust qmsum's 38 rows first.
    "scrolls_gov_report": 0.643,
    "scrolls_summ_screen_fd": 0.283,
    "scrolls_qmsum": 0.074,
    "gsm8k": 0.45,
    "coqa": 0.45,
    "humaneval": 0.10,
    # longbench_qa row (owner 2026-09-05): proportional to the MEASURED window-fit
    # survivors at context 8192 (qasper 186, 2wikimqa 147, multifieldqa_en 92,
    # hotpotqa 29 of 200/200/150/200) so the four pools drain together;
    # musique (3) and narrativeqa (7) survivors are too few for a row.
    "longbench_qasper": 0.410,
    "longbench_2wikimqa": 0.324,
    "longbench_multifieldqa_en": 0.203,
    "longbench_hotpotqa": 0.064,
    # longbench_code row: ∝ measured window-fit survivors at 8192 (lcc 478/500, repobench-p 217/500).
    "longbench_lcc": 0.688,
    "longbench_repobench-p": 0.312,
    # single-dataset rows
    "longbench_write": 1.0,
    "lca_libgen": 1.0,
    "lca_libgen_noapi": 1.0,
    "lca_libgen_official": 1.0,
    "longwriter6k": 1.0,
    "livecodebench": 1.0,
    "bbh_cot": 1.0,
}
DEFAULT_PREFILL_WEIGHTS = {
    "mmlu": 1.0 / 3.0,
    "hellaswag": 1.0 / 3.0,
    "winogrande": 1.0 / 3.0,
    "gov_report": 0.0,
    "multi_news": 0.0,
}
DEFAULT_DATASET_WEIGHTS = {**DEFAULT_DECODE_WEIGHTS, **DEFAULT_PREFILL_WEIGHTS}

PROTOCOL_SCHEMA_VERSION = 4
PROTOCOL_SUITE_ID = "flexidepth-lm-eval-0.4.9.1-serving-v1"
# Uniform generation penalty applied to every request body of a suite.
#
# The default leaves every request body and every metadata row byte-identical to
# a suite built before the penalty became a builder input, so the frozen payload
# does not move. The value itself is ALWAYS declared in the suite summary — an
# absent declaration is an error rather than an implicit 0.0 — and it always
# binds the suite identity hashes, so a rebuild that drops the policy changes the
# hashes instead of silently producing a penalty-free suite.
DEFAULT_FREQUENCY_PENALTY = 0.0
FREQUENCY_PENALTY_FIELD = "frequency_penalty"
DATASET_PROTOCOLS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "id": "lm-eval-0.4.9.1:gsm8k-v3:5shot-multiturn",
        "source": "openai/gsm8k:main",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "gsm8k_cot": {
        "id": "lm-eval-0.4.9.1:gsm8k_cot-v3:8shot-fixed-multiturn",
        "source": "openai/gsm8k:main",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "gsm8k_cot_zeroshot": {
        # lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml (v3): doc_to_text
        # "Q: {{question}}\nA: Let's think step by step.", 0-shot, until
        # ["Q:", "</s>", "<|im_end|>"], strict-match "The answer is (...)."
        # max_gen_toks is the harness default (256) -- quality lane only.
        "id": "lm-eval-0.4.9.1:gsm8k_cot_zeroshot-v3:0shot-chat",
        "source": "openai/gsm8k:main",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "minerva_math": {
        # lm_eval/tasks/minerva_math/*.yaml (v2.0, 7 subjects): 4 FIXED
        # Minerva shots (utils.list_fewshot_samples, sampler first_n),
        # doc_to_text "Problem:\n{problem}\n\nSolution:", until ["Problem:"],
        # metrics exact_match (sympy is_equiv) + math_verify -- both scored
        # ONLY by the third-party harness offline. max_gen_toks = harness
        # default (256), quality lane only.
        "id": "lm-eval-0.4.9.1:minerva_math-v2:4shot-fixed-multiturn",
        "source": "EleutherAI/hendrycks_math:7-subjects",
        "evaluation_split": "test",
        "quality_semantics": "minerva_math_offline",
        "task_reference_max_output_len": 256,
    },
    # LongBench (v1) QA rows — lm-eval 0.4.9.1 longbench_<task> (v3.0): official
    # doc_to_text, official qa_f1_score (offline), max_gen_toks per task.
    # DECLARED DEVIATION ("newlines-real"): the harness yamls are single-quoted
    # YAML, so lm-eval renders "\n" as the two characters backslash+n; the
    # LongBench authors' prompts use real line breaks, which is what we render
    # (verified 2026-09-05: templates are otherwise byte-identical). Both arms
    # see the same prompt; the F1 scorer is unaffected.
    **{
        f"longbench_{task}": {
            "id": f"lm-eval-0.4.9.1:longbench_{task}-v3:zero-shot-chat:newlines-real",
            "source": "zai-org/LongBench:data.zip",
            "evaluation_split": "test",
            "quality_semantics": "longbench_qa_f1_offline",
            "task_reference_max_output_len": genlen,
        }
        for task, genlen in (
            ("qasper", 128),
            ("multifieldqa_en", 64),
            ("hotpotqa", 32),
            ("2wikimqa", 32),
            ("musique", 32),
            ("narrativeqa", 128),
        )
    },
    **{
        f"longbench_{task}": {
            "id": f"lm-eval-0.4.9.1:longbench_{task}-v3:zero-shot-chat:newlines-real",
            "source": "zai-org/LongBench:data.zip",
            "evaluation_split": "test",
            "quality_semantics": "longbench_code_sim_offline",
            "task_reference_max_output_len": 64,
        }
        for task in ("lcc", "repobench-p")
    },
    # Balanced long-output rows (owner, 2026-09-06). Staged locally
    # (`newrows_local.py`), window-FILTERED, natural (uncapped) generation.
    "longbench_write": {
        # LongWriter (THUDM 2024) LongBench-Write: 120 prompts with a required
        # length in words; the model's chat template renders the prompt as in
        # the authors' pred.py. Quality = the authors' length score S_l
        # (offline formula) and quality score S_q (GPT-4o judge, only with the
        # owner's approval) via score_longwrite_offline.py.
        "id": "longwriter:longbench_write-v1:zero-shot-chat",
        "source": "THUDM/LongWriter:evaluation/longbench_write.jsonl",
        "evaluation_split": "test",
        "quality_semantics": "longwrite_offline",
        # window-filter budget only (required words x 1.35 tokens/word); the
        # served generation is natural. Set per row from the record.
        "task_reference_max_output_len": None,
    },
    "lca_libgen": {
        # Long Code Arena library-based code generation (JetBrains Research
        # 2024): instruction + the library's defined elements (the benchmark's
        # "with API list" context). Quality = ChrF + API recall from the
        # lca-baselines metrics via score_lca_offline.py.
        "id": "lca-libgen-v1:instruction+api-list-chat",
        "source": "JetBrains-Research/lca-library-based-code-generation:test",
        "evaluation_split": "test",
        "quality_semantics": "lca_offline",
        "task_reference_max_output_len": None,
    },
    "lca_libgen_noapi": {
        #: the benchmark's instruction-only setting (no API list) -- the
        # full API list drove Llama-3-8B into repetition loops on 55-61 % of
        # prompts in both arms. Same reference, same scorer.
        "id": "lca-libgen-v1:instruction-only-chat",
        "source": "JetBrains-Research/lca-library-based-code-generation:test",
        "evaluation_split": "test",
        "quality_semantics": "lca_offline",
        "task_reference_max_output_len": None,
    },
    "lca_libgen_official": {
        # Owner ruling (2026-09-06): the benchmark's OWN protocol
        # (lca-baselines library_based_code_generation/src/models): the
        # no-context prompt template verbatim as the user message, greedy
        # (temperature 0.0) and max_tokens = 2048. The cap is the benchmark's
        # generation limit and is applied ONLY on this row because
        # Llama-3-8B-Instruct runs into repetition loops on 14-15 % of these
        # prompts in BOTH arms under natural generation ; suites built
        # from this dataset use decode output policy `task_reference_limit`.
        "id": "lca-libgen-v1:official-nocontext-chat:cap2048",
        "source": "JetBrains-Research/lca-library-based-code-generation:test",
        "evaluation_split": "test",
        "quality_semantics": "lca_offline",
        "task_reference_max_output_len": 2048,
    },
    "longwriter6k": {
        # Owner 2026-09-06 (serving-scale writing row): the LongWriter-6k
        # prompts (THUDM), each a single user instruction with a required
        # length; the reference response is used ONLY by the window filter
        # (prompt + reference <= context). Natural, uncapped generation.
        # Quality: LongWriter's length score S_l (offline formula) where the
        # prompt states a required length; S_q needs the owner's API approval.
        "id": "longwriter:longwriter6k-v1:zero-shot-chat",
        "source": "THUDM/LongWriter-6k:train",
        "evaluation_split": "train",
        "quality_semantics": "longwrite_offline",
        "task_reference_max_output_len": None,
    },
    "livecodebench": {
        # Owner 2026-09-06 (serving-scale code row): LiveCodeBench
        # code_generation_lite, every release, unique question_id, under the
        # benchmark's OWN chat prompt (lcb_runner/prompts/code_generation.py:
        # SYSTEM_MESSAGE_GENERIC + '### Question / ### Format / ### Answer'
        # template, the LLaMa3 style). Natural, uncapped generation. No
        # reference response exists (tests only), so the window filter is
        # prompt-only. Quality: the benchmark's own test harness, offline.
        "id": "livecodebench:code_generation_lite-all:official-chat",
        "source": "livecodebench/code_generation_lite:test.jsonl..test6.jsonl",
        "evaluation_split": "test",
        "quality_semantics": "lcb_offline",
        "task_reference_max_output_len": None,
    },
    "mmlu_pro": {
        "id": "lm-eval-0.4.9.1:mmlu_pro-native:5shot-letter-chat",
        "source": "TIGER-Lab/MMLU-Pro",
        "evaluation_split": "test",
        "quality_semantics": "declared_native_variant",
        "task_reference_max_output_len": 16,
    },
    "mmlu_pro_cot": {
        "id": "lm-eval-0.4.9.1:mmlu_pro-cot:5shot-cot-chat",
        "source": "TIGER-Lab/MMLU-Pro",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1024,
    },
    "bbh_cot": {
        # lm_eval/tasks/bbh/cot_fewshot/*.yaml (v4.0, 27 tasks; the task files
        # are shipped verbatim under test/vp/bbh_cot_fewshot/): description +
        # 3 FIXED CoT shots (fewshot_config.samples, sampler first_n) rendered
        # as user/assistant turns, doc_to_text "Q: {{input}}\nA: Let's think
        # step by step.\n", until ["</s>", "Q", "\n\n"], filter regex
        # "(?<=the answer is )(.*)(?=.)" take_first, exact_match on the
        # target; max_gen_toks 1024.
        "id": "lm-eval-0.4.9.1:bbh_cot_fewshot-v4:3shot-fixed-raw",
        "source": "SaylorTwift/bbh:27-tasks",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1024,
    },
    "ifeval": {
        "id": "lm-eval-0.4.9.1:ifeval-v4:zero-shot-chat",
        "source": "google/IFEval",
        "evaluation_split": "train",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1280,
    },
    "coqa": {
        "id": "lm-eval-0.4.9.1:coqa-v3:zero-shot-raw",
        "source": "EleutherAI/coqa",
        "evaluation_split": "validation",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "scrolls_gov_report": {
        "id": "lm-eval:longbench_gov_report:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "scrolls_summ_screen_fd": {
        "id": "lm-eval:scrolls_summscreenfd:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "scrolls_qmsum": {
        "id": "lm-eval:scrolls_qmsum:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "humaneval": {
        "id": "lm-eval-0.4.9.1:humaneval-v1:zero-shot-raw",
        "source": "openai/openai_humaneval",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1024,
    },
    "mmlu": {
        "id": "lm-eval-0.4.9.1:mmlu-v1:5shot-multiturn-prefill-probe",
        "source": "cais/mmlu",
        "evaluation_split": "test",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "hellaswag": {
        "id": "lm-eval-0.4.9.1:hellaswag-v1:5shot-multiturn-prefill-probe",
        "source": "Rowan/hellaswag",
        "evaluation_split": "validation",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "winogrande": {
        "id": "lm-eval-0.4.9.1:winogrande-v1:5shot-multiturn-prefill-probe",
        "source": "allenai/winogrande:winogrande_xl",
        "evaluation_split": "validation",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "gov_report": {
        "id": "longbench-v2:gov_report:zero-shot",
        "source": "THUDM/LongBench",
        "evaluation_split": "test",
        "quality_semantics": "longbench_rouge_l",
        "task_reference_max_output_len": 512,
    },
    "multi_news": {
        "id": "longbench-v2:multi_news:zero-shot",
        "source": "THUDM/LongBench",
        "evaluation_split": "test",
        "quality_semantics": "longbench_rouge_l",
        "task_reference_max_output_len": 512,
    },
}


@dataclass
class WorkloadItem:
    dataset: str
    item_id: str
    phase: str
    prompt: str | list[dict[str, str]]
    prompt_kind: str
    reference_output_len: Optional[int]
    metric: str
    gold: Any
    protocol_id: str
    quality_semantics: str
    task_reference_max_output_len: Optional[int]
    stop: list[str] = field(default_factory=list)
    evaluator_data: dict[str, Any] = field(default_factory=dict)
    source_item_id: Optional[str] = None
    replica_index: int = 0


def _load_module(name: str, filename: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _gsm8k_example(row: dict[str, Any], include_answer: bool) -> str:
    text = f"Question: {row['question'].strip()}\nAnswer:"
    if include_answer:
        text += " " + row["answer"].strip()
    return text


def _gsm8k_chat_messages(
    train: Any, row: dict[str, Any], fewshot_indices: list[int]
) -> list[dict[str, str]]:
    messages = []
    for shot_index in fewshot_indices:
        shot = train[shot_index]
        messages.extend(
            [
                {"role": "user", "content": _gsm8k_example(shot, False)},
                {"role": "assistant", "content": shot["answer"]},
            ]
        )
    messages.append({"role": "user", "content": _gsm8k_example(row, False)})
    return messages


def load_gsm8k(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    import datasets

    train = datasets.load_dataset("openai/gsm8k", "main", split="train")
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    protocol = DATASET_PROTOCOLS["gsm8k"]
    fewshot_rng = random.Random(1234)
    items: list[WorkloadItem] = []

    def _emit(split_name: str, rows: Any, forbid_self: bool) -> None:
        # test is emitted first and unchanged (same order + same few-shot draws) so
        # eval-split items stay byte-identical when include_train_pool is False.
        for index, row in enumerate(rows):
            if len(items) >= limit:
                return
            fewshot_indices = fewshot_rng.sample(range(len(train)), 5)
            while forbid_self and index in fewshot_indices:
                fewshot_indices = fewshot_rng.sample(range(len(train)), 5)
            messages = _gsm8k_chat_messages(train, row, fewshot_indices)
            items.append(
                WorkloadItem(
                    dataset="gsm8k",
                    item_id=f"{split_name}:{index}",
                    phase="decode",
                    prompt=messages,
                    prompt_kind="chat_messages",
                    reference_output_len=None,
                    metric="gsm8k_strict_match",
                    gold=_gsm8k_answer(row["answer"]),
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["Question:", "</s>", "<|im_end|>"],
                    evaluator_data={
                        "source_split": split_name,
                        "fewshot_indices": fewshot_indices,
                        "fewshot_seed": 1234,
                    },
                )
            )

    _emit("test", test, forbid_self=False)
    if include_train_pool:
        # Perf/throughput lane only: train rows join the SERVING load (~8.8k total).
        # The quality lane scores the test split only, so no leak.
        _emit("train", train, forbid_self=True)
    return items


# The 8 fixed chain-of-thought exemplars from lm-eval 0.4.9.1
# lm_eval/tasks/gsm8k/gsm8k-cot.yaml (fewshot_config.samples, sampler
# first_n) — reproduced VERBATIM; any wording drift breaks protocol parity.
GSM8K_COT_FEWSHOT: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in"
        " the grove today. After they are done, there will be 21 trees. How"
        " many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some"
        " more were planted. So there must have been 21 - 15 = 6. The answer"
        " is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how"
        " many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The"
        " answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how"
        " many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total"
        " they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The"
        " answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has"
        " 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some"
        " to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his"
        " mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and"
        " dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers"
        " were installed each day, from monday to thursday. How many"
        " computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more"
        " computers were added. So 5 * 4 = 20 computers were added. 9 + 20"
        " is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On"
        " wednesday, he lost 2 more. How many golf balls did he have at the"
        " end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he"
        " had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf"
        " balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money"
        " does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 ="
        " 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The"
        " answer is 8.",
    ),
)


def _gsm8k_cot_question(question: str) -> str:
    # doc_to_text of gsm8k-cot.yaml: "Q: {{question}}\n\nA:"
    return f"Q: {question.strip()}\n\nA:"


def load_gsm8k_cot(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    import datasets

    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    protocol = DATASET_PROTOCOLS["gsm8k_cot"]
    shot_messages = []
    shot_questions: set[str] = set()
    for shot_question, shot_target in GSM8K_COT_FEWSHOT:
        shot_questions.add(shot_question.strip())
        shot_messages.extend(
            [
                {"role": "user", "content": _gsm8k_cot_question(shot_question)},
                {"role": "assistant", "content": shot_target},
            ]
        )
    items: list[WorkloadItem] = []

    def _emit(split_name: str, rows: Any) -> None:
        # test is emitted first and unchanged, so eval-split items stay byte-identical
        # when include_train_pool is False (same contract as load_gsm8k / load_coqa).
        for index, row in enumerate(rows):
            if len(items) >= limit:
                return
            if split_name == "train" and row["question"].strip() in shot_questions:
                # a train row that IS one of the eight fixed CoT shots would put
                # its own solution in its prompt; skip it (perf lane only).
                continue
            messages = [
                *shot_messages,
                {"role": "user", "content": _gsm8k_cot_question(row["question"])},
            ]
            items.append(
                WorkloadItem(
                    dataset="gsm8k_cot",
                    item_id=f"{split_name}:{index}",
                    phase="decode",
                    prompt=messages,
                    prompt_kind="chat_messages",
                    reference_output_len=None,
                    metric="gsm8k_cot_strict_match",
                    gold=_gsm8k_answer(row["answer"]),
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["Q:", "</s>", "<|im_end|>"],
                    evaluator_data={
                        "source_split": split_name,
                        "fewshot": "gsm8k_cot_v3_first8_fixed",
                    },
                )
            )

    _emit("test", test)
    if include_train_pool:
        # Perf/throughput lane only: train questions join the SERVING load (same
        # 8 fixed shots). The quality lane scores the test split only, so no leak.
        _emit("train", datasets.load_dataset("openai/gsm8k", "main", split="train"))
    return items


def load_gsm8k_cot_zeroshot(
    limit: int, include_train_pool: bool = False
) -> list[WorkloadItem]:
    """lm-eval `gsm8k_cot_zeroshot` (v3) under the chat template: ONE user turn
    "Q: {question}\\nA: Let's think step by step." (doc_to_text verbatim), no
    shots, stop on the yaml's `until` list; strict-match answer regex."""

    import datasets

    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    protocol = DATASET_PROTOCOLS["gsm8k_cot_zeroshot"]
    items: list[WorkloadItem] = []

    def _emit(split_name: str, rows: Any) -> None:
        # test first and unchanged (same contract as load_gsm8k / load_gsm8k_cot).
        for index, row in enumerate(rows):
            if len(items) >= limit:
                return
            # doc_to_text verbatim: the question is NOT stripped (byte parity with
            # the harness, which renders {{question}} raw).
            prompt = f"Q: {row['question']}\nA: Let's think step by step."
            items.append(
                WorkloadItem(
                    dataset="gsm8k_cot_zeroshot",
                    item_id=f"{split_name}:{index}",
                    phase="decode",
                    prompt=[{"role": "user", "content": prompt}],
                    prompt_kind="chat_messages",
                    reference_output_len=None,
                    metric="gsm8k_cot_zeroshot_strict_match",
                    gold=_gsm8k_answer(row["answer"]),
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["Q:", "</s>", "<|im_end|>"],
                    evaluator_data={"source_split": split_name, "fewshot": "none"},
                )
            )

    _emit("test", test)
    if include_train_pool:
        # Perf/throughput lane only; the quality lane scores the test split.
        _emit("train", datasets.load_dataset("openai/gsm8k", "main", split="train"))
    return items


# The 4 fixed Minerva exemplars from lm-eval 0.4.9.1
# lm_eval/tasks/minerva_math/utils.py:list_fewshot_samples (sampler first_n),
# reproduced VERBATIM (including the stray "}" closing the first problem).
MINERVA_MATH_FEWSHOT: tuple[tuple[str, str], ...] = (
    (
        "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "The expressions inside each square root must be non-negative. "
        "Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. "
        "Also, the denominator cannot be equal to zero, so $5-x>0$, which gives "
        "$x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
        "Final Answer: The final answer is $[2,5)$. I hope it is correct.",
    ),
    (
        "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
        "$\\det (\\mathbf{A} \\mathbf{B}).$",
        "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})"
        "(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\n"
        "Final Answer: The final answer is $24$. I hope it is correct.",
    ),
    (
        "Terrell usually lifts two 20-pound weights 12 times. If he uses two "
        "15-pound weights instead, how many times must Terrell lift them in order "
        "to lift the same total weight?",
        "If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
        "$2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound "
        "weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot "
        "n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for "
        "$n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30="
        "\\boxed{16}\n\\end{align*}\n"
        "Final Answer: The final answer is $16$. I hope it is correct.",
    ),
    (
        "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b."
        "\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both "
        "nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n"
        "$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n"
        "$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\n"
        "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    ),
)
MINERVA_MATH_SUBJECTS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _minerva_math_problem(problem: str) -> str:
    # doc_to_text of minerva_math/utils.py: "Problem:" + "\n" + problem + "\n\n" + "Solution:"
    return "Problem:" + "\n" + problem + "\n\n" + "Solution:"


def _minerva_last_boxed_only_string(string: str) -> Optional[str]:
    # lm_eval/tasks/minerva_math/utils.py:last_boxed_only_string, verbatim.
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _minerva_remove_boxed(s: str) -> str:
    # lm_eval/tasks/minerva_math/utils.py:remove_boxed, verbatim.
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def load_minerva_math(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    """lm-eval `minerva_math` (v2.0, 7 subjects) under the chat template: the 4
    fixed Minerva shots as user/assistant turns, then the problem. Quality is
    scored ONLY by the third-party harness (sympy `is_equiv` + `math_verify`);
    the gold here is the raw boxed answer string kept for the audit trail."""

    import datasets

    protocol = DATASET_PROTOCOLS["minerva_math"]
    shot_messages = []
    for shot_problem, shot_solution in MINERVA_MATH_FEWSHOT:
        shot_messages.extend(
            [
                {"role": "user", "content": _minerva_math_problem(shot_problem)},
                {"role": "assistant", "content": shot_solution},
            ]
        )
    items: list[WorkloadItem] = []

    def _emit(split_name: str) -> None:
        # test first (subjects in the harness's fixed order), then the train pool.
        for subject in MINERVA_MATH_SUBJECTS:
            rows = datasets.load_dataset(
                "EleutherAI/hendrycks_math", subject, split=split_name
            )
            for index, row in enumerate(rows):
                if len(items) >= limit:
                    return
                boxed = _minerva_last_boxed_only_string(row["solution"])
                if boxed is None:
                    # the harness would fail on such a row; none exist in the
                    # released splits, but never emit an item without a gold.
                    continue
                try:
                    gold = _minerva_remove_boxed(boxed)
                except AssertionError:
                    # a `\fbox{...}` answer (or a malformed box): lm-eval's own
                    # process_docs asserts on exactly this, so such a row cannot
                    # be in a split the harness evaluates; skip rather than abort
                    # the whole suite build (Codex review).
                    continue
                messages = [
                    *shot_messages,
                    {"role": "user", "content": _minerva_math_problem(row["problem"])},
                ]
                items.append(
                    WorkloadItem(
                        dataset="minerva_math",
                        item_id=f"{split_name}:{subject}:{index}",
                        phase="decode",
                        prompt=messages,
                        prompt_kind="chat_messages",
                        reference_output_len=None,
                        metric="math_verify_offline",
                        gold=gold,
                        protocol_id=protocol["id"],
                        quality_semantics=protocol["quality_semantics"],
                        task_reference_max_output_len=protocol[
                            "task_reference_max_output_len"
                        ],
                        # exact parity with minerva_math's generation until-list
                        # (["Problem:"] only; the chat template's own EOS ends turns)
                        stop=["Problem:"],
                        evaluator_data={
                            "source_split": split_name,
                            "subject": subject,
                            "level": row["level"],
                            "fewshot": "minerva_math_v2_first4_fixed",
                        },
                    )
                )

    _emit("test")
    if include_train_pool and len(items) < limit:
        # Perf/throughput lane only (7,500 train problems); quality = test split.
        # Guarded so a limit already met by the test split loads no train shard.
        _emit("train")
    return items


_MMLU_PRO_LETTERS = "ABCDEFGHIJ"


def _mmlu_pro_question(row: dict) -> str:
    # lm-eval mmlu_pro doc_to_text shape: Question / Options / Answer
    lines = ["Question:", row["question"].strip(), "Options:"]
    # The published dataset pads every options list to 10 with "N/A"; the
    # official MMLU-Pro loader removes those placeholders before lettering.
    options = [o for o in row["options"] if o != "N/A"]
    for letter, option in zip(_MMLU_PRO_LETTERS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def _load_mmlu_pro_shots(cot: bool) -> dict[str, list[dict]]:
    import datasets

    validation = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    by_category: dict[str, list[dict]] = {}
    for row in validation:
        by_category.setdefault(row["category"], []).append(row)
    shots: dict[str, list[dict]] = {}
    for category, rows in by_category.items():
        messages = []
        for row in rows[:5]:
            if cot:
                # validation cot_content begins "A: Let's think step by
                # step." — the user turn already carries that cue, so strip
                # the FULL fixed prefix (review P2: avoid duplicating it).
                target = row["cot_content"].strip()
                for prefix in ("A: Let's think step by step.", "A:"):
                    if target.startswith(prefix):
                        target = target[len(prefix):].strip()
                        break
                question = _mmlu_pro_question(row) + "\nAnswer: Let's think step by step."
            else:
                target = f"The answer is ({row['answer']})."
                question = _mmlu_pro_question(row) + "\nAnswer:"
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": target},
                ]
            )
        shots[category] = messages
    return shots


def load_mmlu_pro(limit: int, cot: bool) -> list[WorkloadItem]:
    import datasets

    dataset_name = "mmlu_pro_cot" if cot else "mmlu_pro"
    protocol = DATASET_PROTOCOLS[dataset_name]
    test = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    shots = _load_mmlu_pro_shots(cot)
    missing = sorted({row["category"] for row in test} - set(shots))
    if missing:
        raise ValueError(
            f"MMLU-Pro test categories missing validation shots: {missing}"
        )
    items = []
    for index, row in enumerate(test):
        if len(items) >= limit:
            break
        suffix = "\nAnswer: Let's think step by step." if cot else "\nAnswer:"
        messages = [
            *shots[row["category"]],
            {"role": "user", "content": _mmlu_pro_question(row) + suffix},
        ]
        items.append(
            WorkloadItem(
                dataset=dataset_name,
                item_id=f"test:{index}",
                phase="decode",
                prompt=messages,
                prompt_kind="chat_messages",
                reference_output_len=None,
                metric="mmlu_pro_letter_match",
                gold=str(row["answer"]),
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=["Question:", "</s>", "<|im_end|>"],
                evaluator_data={
                    "source_split": "test",
                    "category": row["category"],
                    "fewshot": "mmlu_pro_validation_first5_per_category",
                },
            )
        )
    return items


def mmlu_pro_letter_match(prediction: str, gold: str) -> float:
    # lm-eval mmlu_pro filters, patterns VERBATIM: primary
    # 'answer is \(?([ABCDEFGHIJ])\)?', fallback r'.*[aA]nswer:\s*\(?([ABCDEFGHIJ])\)?'
    match = re.search(r"answer is \(?([ABCDEFGHIJ])\)?", prediction)
    if match is None:
        match = re.search(r".*[aA]nswer:\s*\(?([ABCDEFGHIJ])\)?", prediction)
    if match is None:
        return 0.0
    return float(match.group(1) == str(gold).strip())


def load_ifeval(limit: int) -> list[WorkloadItem]:
    import datasets

    rows = datasets.load_dataset("google/IFEval", split="train")
    protocol = DATASET_PROTOCOLS["ifeval"]
    items = []
    for index, row in enumerate(rows.select(range(min(limit, len(rows))))):
        items.append(
            WorkloadItem(
                dataset="ifeval",
                item_id=f"train:{index}",
                phase="decode",
                prompt=[{"role": "user", "content": row["prompt"]}],
                prompt_kind="chat_messages",
                reference_output_len=None,
                # Scored ONLY by the third-party harness offline
                # (score_lmeval_offline.py); no in-repo reimplementation of
                # the IFEval instruction registry.
                metric="ifeval_offline",
                gold=None,
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=[],
                evaluator_data={
                    "source_split": "train",
                    "key": row["key"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                },
            )
        )
    return items


BBH_COT_TASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
)


def _bbh_cot_task_spec(task: str) -> dict[str, Any]:
    """The lm-eval cot_fewshot task file, verbatim (description, doc_to_text,
    the three fixed CoT shots)."""
    import yaml

    path = Path(__file__).resolve().parent / "bbh_cot_fewshot" / f"{task}.yaml"
    spec = yaml.safe_load(path.read_text())
    shots = spec["fewshot_config"]["samples"]
    if spec["fewshot_config"].get("sampler") != "first_n" or len(shots) != 3:
        raise ValueError(f"bbh_cot task {task}: expected 3 first_n CoT shots")
    if spec["doc_to_text"] != "Q: {{input}}\nA: Let's think step by step.\n":
        raise ValueError(f"bbh_cot task {task}: unexpected doc_to_text")
    return spec


def _bbh_cot_question(text: str) -> str:
    # doc_to_text of every cot_fewshot task (the trailing newline included)
    return f"Q: {text}\nA: Let's think step by step.\n"


def load_bbh_cot(limit: int) -> list[WorkloadItem]:
    """BBH chain-of-thought: description + three fixed lm-eval CoT shots as a RAW
    CONTINUATION, then the question; tasks interleaved round-robin so any prefix of the
    pool is task-balanced.

    RAW, not chat (D-643). `doc_to_text` is "Q: {{input}}\nA: Let's think step by step.\n"
    -- a continuation prompt -- every exemplar ends "So the answer is X.", so the answer
    marker the sole `get-answer` filter needs is PROMPT-INDUCED, and `until`'s "\n\n" is
    the few-shot separator. Under a chat template the model stops continuing and starts
    answering: it writes its own preamble, emits a paragraph break, and generation dies at
    that "\n\n" before any reasoning exists.

    Measured cost of getting this wrong: stock's marker rate was 76.7% under chat vs 98.1%
    under raw, and (D - C) INVERTED from +10.83 pp to -6.54 pp -- a 17 pp swing in the
    direction that flattered us. Raw does not collapse this checkpoint on BBH (0.0% empty
    both arms), unlike raw gsm8k's 55% (D-632), so the risk that forces chat on gsm8k does
    not apply here."""
    import datasets

    protocol = DATASET_PROTOCOLS["bbh_cot"]
    per_task: list[list[WorkloadItem]] = []
    for task in BBH_COT_TASKS:
        spec = _bbh_cot_task_spec(task)
        rows = datasets.load_dataset("SaylorTwift/bbh", task, split="test")
        # lm-eval's own assembly, verified byte-identical against a served raw run:
        #   <description>\n\n  then each  "Q: <in>\nA: Let's think step by step.\n<target>"
        #   joined by "\n\n" (the few-shot separator), then the question with no target.
        # target_delimiter is "", so the CoT target follows doc_to_text's trailing newline
        # directly.
        shot_block = "\n\n".join(
            _bbh_cot_question(shot["input"]) + shot["target"]
            for shot in spec["fewshot_config"]["samples"]
        )
        # NOTE spec["description"] ALREADY ends with "\n\n" -- adding another produced a
        # 2-byte divergence from lm-eval, caught by the byte-equality check against a real
        # served run rather than by reading the yaml.
        prefix = spec["description"] + shot_block + "\n\n"
        items: list[WorkloadItem] = []
        for index, row in enumerate(rows):
            items.append(
                WorkloadItem(
                    dataset="bbh_cot",
                    item_id=f"{task}:test:{index}",
                    phase="decode",
                    prompt=prefix + _bbh_cot_question(row["input"]),
                    prompt_kind="raw_completion",
                    reference_output_len=None,
                    metric="bbh_cot_exact_match",
                    gold=row["target"],
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    # lm-eval generation_kwargs.until, verbatim. NO chat EOS: this is
                    # the raw protocol, and "\n\n" is meaningful here (it separates
                    # exemplars) precisely because the prompt is a raw continuation.
                    stop=["</s>", "Q", "\n\n"],
                    evaluator_data={"task": task, "source_split": "test"},
                )
            )
        per_task.append(items)
    ordered: list[WorkloadItem] = []
    for position in range(max(len(items) for items in per_task)):
        for items in per_task:
            if position < len(items):
                ordered.append(items[position])
    return ordered[:limit]


def _coqa_questions(row: dict[str, Any]) -> list[str]:
    questions = row["questions"]
    if isinstance(questions, dict):
        values = questions.get("input_text") or questions.get("question") or []
        return [str(value) for value in values]
    return [str(value) for value in questions]


def _coqa_gold_answers(row: dict[str, Any], turn: int) -> list[str]:
    answers = row["answers"]
    if isinstance(answers, dict):
        values = []
        for key in ("input_text", "answer"):
            candidate = answers.get(key)
            if isinstance(candidate, list) and turn < len(candidate):
                values.append(str(candidate[turn]))
            elif isinstance(candidate, str) and turn == 0:
                values.append(candidate)
        primary = [value for value in values if value]
    elif isinstance(answers, list) and turn < len(answers):
        answer = answers[turn]
        if isinstance(answer, dict):
            primary = [
                str(answer[key])
                for key in ("input_text", "answer")
                if key in answer
            ]
        else:
            primary = [str(answer)]
    else:
        primary = []

    additional = row.get("additional_answers") or {}
    for annotator in additional.values() if isinstance(additional, dict) else []:
        if not isinstance(annotator, dict):
            continue
        values = annotator.get("input_text") or []
        if isinstance(values, list) and turn < len(values):
            primary.append(str(values[turn]))
    deduplicated = []
    seen = set()
    for value in primary:
        normalized = value.lower()
        if value and normalized not in seen:
            seen.add(normalized)
            deduplicated.append(value)
    return deduplicated


def _coqa_official_prompt(
    row: dict[str, Any],
) -> tuple[str, list[str], int]:
    questions = _coqa_questions(row)
    if not questions:
        return "", [], -1
    target_turn = len(questions) - 1
    gold = _coqa_gold_answers(row, target_turn)
    history_answers = [
        (_coqa_gold_answers(row, turn) or [""])[0]
        for turn in range(target_turn)
    ]
    prompt = (row.get("story") or row.get("passage") or "") + "\n\n"
    for question, answer in zip_longest(questions, history_answers):
        if question is None:
            break
        prompt += f"Q: {question}\n\n"
        prompt += f"A: {answer}\n\n" if answer is not None else "A:"
    return prompt, gold, target_turn


def load_coqa(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    import datasets

    protocol = DATASET_PROTOCOLS["coqa"]

    def _load(split: str) -> Any:
        for source in ("EleutherAI/coqa", "stanfordnlp/coqa"):
            try:
                return datasets.load_dataset(source, split=split)
            except Exception:
                continue
        return datasets.load_dataset("coqa", split=split, trust_remote_code=True)

    items: list[WorkloadItem] = []

    def _emit(split_name: str, data: Any) -> None:
        # validation is emitted first and unchanged, so eval-split items stay
        # byte-identical when include_train_pool is False.
        for story_index, row in enumerate(data):
            if len(items) >= limit:
                return
            prompt, gold, target_turn = _coqa_official_prompt(row)
            if not gold:
                continue
            items.append(
                WorkloadItem(
                    dataset="coqa",
                    item_id=f"{split_name}:{story_index}:{target_turn}",
                    phase="decode",
                    prompt=prompt,
                    prompt_kind="raw_completion",
                    reference_output_len=None,
                    metric="coqa_f1",
                    gold=gold,
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["\nQ:"],
                    evaluator_data={
                        "source_split": split_name,
                        "target_turn": target_turn,
                    },
                )
            )

    _emit("validation", _load("validation"))
    if include_train_pool:
        # Perf/throughput lane only: train stories join the SERVING load (~7.7k total).
        # The quality lane scores the validation split only, so no leak.
        _emit("train", _load("train"))
    return items


def load_humaneval(limit: int) -> list[WorkloadItem]:
    import datasets

    data = datasets.load_dataset("openai/openai_humaneval", split="test")
    protocol = DATASET_PROTOCOLS["humaneval"]
    items = []
    for index, row in enumerate(data):
        task_id = str(row.get("task_id", index))
        items.append(
            WorkloadItem(
                dataset="humaneval",
                item_id=task_id,
                phase="decode",
                prompt=row["prompt"],
                prompt_kind="raw_completion",
                reference_output_len=None,
                metric="humaneval_pass_at_1",
                gold=None,
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                evaluator_data={
                    "source_split": "test",
                    "task_id": task_id,
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "canonical_solution": row.get("canonical_solution", ""),
                },
            )
        )
        if len(items) >= limit:
            break
    return items


def load_selection_dataset(dataset: str, limit: int) -> list[WorkloadItem]:
    module = _load_module("vp_flexidepth_select_eval", "flexidepth_select_eval.py")
    _few_shot, rows = module._load_task(dataset, limit, 5)
    source_split = {
        "mmlu": "test",
        "hellaswag": "validation",
        "winogrande": "validation",
    }[dataset]
    items = []
    protocol = DATASET_PROTOCOLS[dataset]
    for row in rows:
        label = int(row["label"])
        if row["selection_mode"] == "multiple_input":
            prompt_variants = [
                (
                    f"{row['id']}:candidate:{candidate_index}",
                    [
                        *row["fewshot_messages"],
                        {"role": "user", "content": candidate},
                    ],
                    candidate_index,
                )
                for candidate_index, candidate in enumerate(row["choices"])
            ]
        else:
            prompt_variants = [
                (
                    str(row["id"]),
                    list(row["messages"]),
                    None,
                )
            ]
        for item_id, prompt, candidate_index in prompt_variants:
            items.append(
                WorkloadItem(
                    dataset=dataset,
                    item_id=item_id,
                    phase="prefill",
                    prompt=prompt,
                    prompt_kind="chat_messages",
                    reference_output_len=1,
                    metric="selection_prefill_probe",
                    gold={
                        "index": label,
                        "choice": str(row["choices"][label]),
                    },
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["\n"],
                    evaluator_data={
                        "source_split": source_split,
                        "fewshot_indices": row.get("fewshot_indices") or [],
                        "selection_metric": row["selection_metric"],
                        "selection_mode": row["selection_mode"],
                        "candidate_index": candidate_index,
                        "continuation": row.get("continuation"),
                    },
                )
            )
    return items


def load_scrolls_summary(
    dataset: str,
    limit: int,
    tokenizer: Any,
    context_length: int,
) -> list[WorkloadItem]:
    """SCROLLS long-context summarisation from locally staged jsonl.

    Rows whose prompt + reference output would exceed the model window are
    DROPPED, not truncated (owner ruling A1, 2026-09-04): truncating would score
    ROUGE against a summary of a document the model only half saw, making the
    figure incomparable to any published SCROLLS number -- which is the reason
    the axis uses SCROLLS validation splits at all. Filtering happens here
    rather than in `render_workload_rows` because the reference output length is
    a per-dataset constant, so both sides of the window test are known at load
    time; doing it later would break that function's gap-free `index` contract.

    Quality is NOT scored in-repo. The metric defers to the third-party harness
    (lm-eval `longbench_gov_report` / `scrolls_*`), matching the `ifeval_offline`
    precedent -- an in-repo ROUGE reimplementation would violate the standing
    rule that quality comes only from third-party harnesses.
    """

    scrolls = _load_module("vp_scrolls_local", "scrolls_local.py")
    protocol = DATASET_PROTOCOLS[dataset]
    reference_output_len = int(scrolls.GENLEN[dataset])
    items: list[WorkloadItem] = []
    considered = 0
    for record in scrolls.iter_records(dataset):
        if len(items) >= limit:
            break
        considered += 1
        prompt = scrolls.render_prompt(dataset, record)
        candidate = WorkloadItem(
            dataset=dataset,
            item_id=str(record["id"]),
            phase="decode",
            prompt=[{"role": "user", "content": prompt}],
            prompt_kind="chat_messages",
            reference_output_len=reference_output_len,
            metric="rouge_offline",
            gold=list(record["answers"]),
            protocol_id=protocol["id"],
            quality_semantics=protocol["quality_semantics"],
            task_reference_max_output_len=protocol[
                "task_reference_max_output_len"
            ],
            stop=[],
            evaluator_data={},
        )
        # Measure with the SAME function that records prompt_len, not a
        # convenient approximation of it. An earlier version filtered on
        # `len(tokenizer(prompt)["input_ids"])` -- the bare text -- while the
        # suite records the CHAT-TEMPLATED length from `_render_prompt_ids`.
        # The template's header and generation prompt make the recorded length
        # larger, so two gov_report rows passed the filter and were then granted
        # 1,022 and 1,023 output tokens against a 1,024-token reference: under
        # the window by the filter's arithmetic, over it by the server's, and
        # silently quality-ineligible. Sharing the function makes the filter and
        # the record unable to disagree.
        prompt_tokens = len(_render_prompt_ids(tokenizer, candidate))
        if not scrolls.fits_window(prompt_tokens, dataset, context_length):
            continue
        items.append(
            replace(
                candidate,
                evaluator_data={
                    "source_split": protocol["evaluation_split"],
                    "prompt_tokens": prompt_tokens,
                    "window_filtered": True,
                    "context_length": context_length,
                },
            )
        )
    if not items:
        raise ValueError(
            f"{dataset}: no staged row fits prompt + {reference_output_len} "
            f"<= {context_length} (considered {considered})"
        )
    return items


# LongBench (v1) QA family via lm-eval 0.4.9.1 `longbench_<task>` tasks: doc_to_text
# VERBATIM from lm_eval/tasks/longbench/<task>.yaml, quality = the harness's own
# `metrics.get_qa_f1_score` (scored offline, third-party), max_gen_toks per task.
# Owner 2026-09-05: "instruction means the current flexidepth with tulu finetuning
# working, and the long context means we can show the win" — real documents,
# instruction prompts, 4-8k tokens; rows that do not fit the window are DROPPED
# (ruling A1: filter, never truncate).
LONGBENCH_QA_TASKS: dict[str, dict[str, Any]] = {
    "qasper": {
        "template": (
            "You are given a scientific article and a question. Answer the question as"
            " concisely as you can, using a single phrase or sentence if possible. If"
            " the question cannot be answered based on the information in the article,"
            ' write "unanswerable". If the question is a yes/no question, answer "yes",'
            ' "no", or "unanswerable". Do not provide any explanation.\n\nArticle:'
            " {context}\n\n Answer the question based on the above article as concisely"
            " as you can, using a single phrase or sentence if possible. If the question"
            " cannot be answered based on the information in the article, write"
            ' "unanswerable". If the question is a yes/no question, answer "yes", "no",'
            ' or "unanswerable". Do not provide any explanation.\n\nQuestion:'
            " {input}\n\nAnswer:"
        ),
        "max_gen_toks": 128,
    },
    "multifieldqa_en": {
        "template": (
            "Read the following text and answer briefly.\n\n{context}\n\nNow, answer"
            " the following question based on the above text, only give me the answer"
            " and do not output any other words.\n\nQuestion: {input}\nAnswer:"
        ),
        "max_gen_toks": 64,
    },
    "hotpotqa": {
        "template": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion:"
            " {input}\nAnswer:"
        ),
        "max_gen_toks": 32,
    },
    "2wikimqa": {
        "template": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion:"
            " {input}\nAnswer:"
        ),
        "max_gen_toks": 32,
    },
    "musique": {
        "template": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion:"
            " {input}\nAnswer:"
        ),
        "max_gen_toks": 32,
    },
    "narrativeqa": {
        "template": (
            "You are given a story, which can be either a novel or a movie script, and"
            " a question. Answer the question asconcisely as you can, using a single"
            " phrase if possible. Do not provide any explanation.\n\nStory:"
            " {context}\n\nNow, answer the question based on the story asconcisely as"
            " you can, using a single phrase if possible. Do not provide any"
            " explanation.\n\nQuestion: {input}\n\nAnswer:"
        ),
        "max_gen_toks": 128,
    },
    # Coding row (owner 2026-09-05,/): LongBench repo-level code completion,
    # lm-eval `longbench_lcc` / `longbench_repobench-p` (doc_to_text verbatim, until [],
    # max_gen_toks 64), quality = LongBench `code_sim_score` (edit similarity, offline).
    "lcc": {
        "template": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen_toks": 64,
        "metric": "longbench_code_sim_offline",
    },
    "repobench-p": {
        "template": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen_toks": 64,
        "metric": "longbench_code_sim_offline",
    },
}


def _longbench_qa_prompt(task: str, record: dict[str, Any]) -> str:
    # {{context}} / {{input}} substituted verbatim (no strip: the harness renders
    # raw) in ONE pass, so a document that itself contains the literal text
    # "{input}" cannot be rewritten by a second replacement (Codex review).
    template = LONGBENCH_QA_TASKS[task]["template"]
    return re.sub(
        r"\{(context|input)\}", lambda m: record[m.group(1)], template
    )


def load_longbench_qa(
    dataset: str, limit: int, tokenizer: Any, context_length: int
) -> list[WorkloadItem]:
    """LongBench QA row (`longbench_<task>`): instruction prompt over a real long
    document, official lm-eval template, official F1 scored offline. Rows whose
    chat-templated prompt + the task's max_gen_toks exceed the window are
    dropped (ruling A1). Stable ids from the dataset's `_id`."""

    import json as _json
    import zipfile

    from huggingface_hub import hf_hub_download

    task = dataset[len("longbench_") :]
    if task not in LONGBENCH_QA_TASKS:
        raise ValueError(f"unsupported LongBench QA task {task!r}")
    protocol = DATASET_PROTOCOLS[dataset]
    genlen = int(LONGBENCH_QA_TASKS[task]["max_gen_toks"])
    # The Hub repo ships a dataset SCRIPT (LongBench.py) + data.zip; `datasets`>=4
    # refuses scripts, so read the jsonl member directly (same access path as
    # longbench_eval.load_records; file order is the seal, `_id` is the stable id).
    zip_path = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")

    def _rows():
        with zipfile.ZipFile(zip_path) as zf, zf.open(f"data/{task}.jsonl") as f:
            for line in f:
                yield _json.loads(line.decode("utf-8"))

    items: list[WorkloadItem] = []
    considered = 0
    for record in _rows():
        if len(items) >= limit:
            break
        considered += 1
        candidate = WorkloadItem(
            dataset=dataset,
            item_id=str(record["_id"]),
            phase="decode",
            prompt=[{"role": "user", "content": _longbench_qa_prompt(task, record)}],
            prompt_kind="chat_messages",
            reference_output_len=None,
            metric=LONGBENCH_QA_TASKS[task].get("metric", "longbench_qa_f1_offline"),
            gold=list(record["answers"]),
            protocol_id=protocol["id"],
            quality_semantics=protocol["quality_semantics"],
            task_reference_max_output_len=genlen,
            stop=[],  # the task's generation until-list is empty
            evaluator_data={},
        )
        prompt_tokens = len(_render_prompt_ids(tokenizer, candidate))
        if prompt_tokens + genlen > context_length:
            continue
        items.append(
            replace(
                candidate,
                evaluator_data={
                    "source_split": "test",
                    "prompt_tokens": prompt_tokens,
                    "window_filtered": True,
                    "context_length": context_length,
                    "longbench_length_words": int(record["length"]),
                },
            )
        )
    if not items:
        raise ValueError(
            f"{dataset}: no row fits prompt + {genlen} <= {context_length} "
            f"(considered {considered})"
        )
    return items


def load_longbench_summary(
    dataset: str, limit: int, tokenizer: Any, max_length: int = 7000
) -> list[WorkloadItem]:
    exporter = _load_module("vp_longbench_export", "export_longbench_autobench.py")
    longbench = exporter.load_longbench_module()
    records = longbench.load_records(dataset, limit)
    protocol = DATASET_PROTOCOLS[dataset]
    items = []
    for index, record in enumerate(records):
        reference_output_len = int(longbench.GENLEN[dataset])
        row = exporter.render_row(
            longbench,
            tokenizer,
            dataset,
            record,
            max_length,
            reference_output_len,
        )
        items.append(
            WorkloadItem(
                dataset=dataset,
                item_id=str(index),
                phase="prefill",
                prompt=row["messages"],
                prompt_kind="chat_messages",
                reference_output_len=reference_output_len,
                metric="rouge_l",
                gold=row["answers"],
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=reference_output_len,
            )
        )
    return items


def load_newrow(
    dataset: str, limit: int, tokenizer: Any, context_length: int
) -> list[WorkloadItem]:
    """Balanced long-output rows (D-519): LongBench-Write / LCA library-based
    code generation from locally staged jsonl (`newrows_local.py`).

    Window filter (ruling A1): prompt (chat-templated, measured with the SAME
    function that records prompt_len) + the row's reference output budget must
    fit the context; rows that do not are DROPPED. Quality is scored only by
    the third-party scorers offline (metric names end in `_offline`).
    """

    newrows = _load_module("vp_newrows_local", "newrows_local.py")
    protocol = DATASET_PROTOCOLS[dataset]
    items: list[WorkloadItem] = []
    considered = 0
    for record in newrows.iter_records(dataset):
        if len(items) >= limit:
            break
        considered += 1
        budget = int(newrows.output_budget_tokens(dataset, record, tokenizer))
        if dataset == "longbench_write":
            metric = "longwrite_offline"
            gold = [int(record["length"])]  # required length in words
            extra = {"required_words": int(record["length"]), "type": record.get("type")}
        elif dataset == "longwriter6k":
            metric = "longwrite_offline"
            required = newrows.longwriter_required_words(record)  # parsed from the prompt; None when unstated
            gold = [required] if required else []
            extra = {"required_words": required, "reference_tokens": budget}
        elif dataset == "livecodebench":
            metric = "lcb_offline"
            gold = {"question_id": str(record["question_id"]), "platform": record.get("platform"), "difficulty": record.get("difficulty")}
            extra = {"has_starter_code": bool(record.get("starter_code")), "contest_date": record.get("contest_date"), "release_file": record.get("release_file")}
        else:
            metric = "lca_offline"
            gold = {
                "reference": str(record["clean_reference"]),
                "unique_apis": [str(a) for a in record["unique_apis"]],
            }
            extra = {
                "repo_full_name": record.get("repo_full_name"),
                "n_unique_apis": int(record.get("n_unique_apis", len(record["unique_apis"]))),
                # the reference program's own token count (information only);
                # for `lca_libgen_official` the budget is the benchmark cap.
                "reference_tokens": int(newrows.reference_tokens(record, tokenizer)),
            }
        candidate = WorkloadItem(
            dataset=dataset,
            item_id=str(record["_id"]),
            phase="decode",
            prompt=newrows.render_messages(dataset, record),
            prompt_kind="chat_messages",
            reference_output_len=None,
            metric=metric,
            gold=gold,
            protocol_id=protocol["id"],
            quality_semantics=protocol["quality_semantics"],
            task_reference_max_output_len=budget,
            stop=[],
            evaluator_data={},
        )
        prompt_tokens = len(_render_prompt_ids(tokenizer, candidate))
        if not newrows.fits_window(prompt_tokens, budget, context_length):
            continue
        items.append(
            replace(
                candidate,
                evaluator_data={
                    "source_split": protocol["evaluation_split"],
                    "prompt_tokens": prompt_tokens,
                    "window_filtered": True,
                    "context_length": context_length,
                    "output_budget_tokens": budget,
                    **extra,
                },
            )
        )
    if not items:
        raise ValueError(
            f"{dataset}: no row fits prompt + budget <= {context_length} "
            f"(considered {considered})"
        )
    return items


def load_dataset_items(
    dataset: str,
    limit: int,
    tokenizer: Optional[Any] = None,
    include_train_pool: bool = False,
    context_length: Optional[int] = None,
) -> list[WorkloadItem]:
    if dataset in {"longbench_write", "lca_libgen", "lca_libgen_noapi", "lca_libgen_official", "longwriter6k", "livecodebench"}:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer for the window filter")
        if context_length is None:
            raise ValueError(
                f"{dataset} requires context_length: the window filter is the "
                "sealed alternative to truncation (ruling A1) and must not "
                "silently default"
            )
        return load_newrow(dataset, limit, tokenizer, context_length)
    if dataset in {
        "scrolls_gov_report",
        "scrolls_summ_screen_fd",
        "scrolls_qmsum",
    }:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer for the window filter")
        if context_length is None:
            raise ValueError(
                f"{dataset} requires context_length: the window filter is the "
                "sealed alternative to truncation (ruling A1) and must not "
                "silently default"
            )
        return load_scrolls_summary(dataset, limit, tokenizer, context_length)
    if dataset == "gsm8k":
        return load_gsm8k(limit, include_train_pool)
    if dataset == "gsm8k_cot":
        return load_gsm8k_cot(limit, include_train_pool)
    if dataset == "gsm8k_cot_zeroshot":
        return load_gsm8k_cot_zeroshot(limit, include_train_pool)
    if dataset == "minerva_math":
        return load_minerva_math(limit, include_train_pool)
    if dataset.startswith("longbench_") and dataset[len("longbench_"):] in LONGBENCH_QA_TASKS:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer for the window filter")
        if context_length is None:
            raise ValueError(
                f"{dataset} requires context_length: the window filter is the "
                "sealed alternative to truncation (ruling A1) and must not "
                "silently default"
            )
        return load_longbench_qa(dataset, limit, tokenizer, context_length)
    if dataset == "ifeval":
        return load_ifeval(limit)
    if dataset == "bbh_cot":
        return load_bbh_cot(limit)
    if dataset == "mmlu_pro":
        return load_mmlu_pro(limit, cot=False)
    if dataset == "mmlu_pro_cot":
        return load_mmlu_pro(limit, cot=True)
    if dataset == "coqa":
        return load_coqa(limit, include_train_pool)
    if dataset == "humaneval":
        return load_humaneval(limit)
    if dataset in {"mmlu", "hellaswag", "winogrande"}:
        return load_selection_dataset(dataset, limit)
    if dataset in {"gov_report", "multi_news"}:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer")
        return load_longbench_summary(dataset, limit, tokenizer)
    raise ValueError(f"unsupported dataset: {dataset}")


def parse_weights(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    result = {}
    for item in raw.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator:
            raise ValueError(f"invalid dataset weight {item!r}; expected name=value")
        result[name] = float(value)
    return result


def _weighted_dataset_choice(
    rng: random.Random,
    available: list[str],
    weights: dict[str, float],
) -> str:
    values = [max(0.0, weights.get(name, 1.0)) for name in available]
    if not any(values):
        values = [1.0] * len(available)
    return rng.choices(available, weights=values, k=1)[0]


def sample_items(
    pools: dict[str, list[WorkloadItem]],
    num_requests: int,
    phase_mode: str,
    decode_ratio: float,
    seed: int,
    weights: Optional[dict[str, float]] = None,
    repeat_exhausted: bool = False,
) -> list[WorkloadItem]:
    if phase_mode not in {"decode", "prefill", "mixed"}:
        raise ValueError(f"invalid phase mode: {phase_mode}")
    if not 0.0 <= decode_ratio <= 1.0:
        raise ValueError("decode_ratio must be in [0, 1]")
    supplied_weights = dict(weights or {})
    weights = {**DEFAULT_DATASET_WEIGHTS, **supplied_weights}
    if len(pools) == 1:
        only_dataset = next(iter(pools))
        if only_dataset not in supplied_weights:
            weights[only_dataset] = 1.0
    originals = {name: list(items) for name, items in pools.items()}
    working = {name: list(items) for name, items in originals.items()}
    for name, items in working.items():
        random.Random(_stable_seed(seed, name)).shuffle(items)
    refill_epochs = Counter()
    occurrences = Counter()
    rng = random.Random(seed)
    sampled = []
    while len(sampled) < num_requests:
        source = originals if repeat_exhausted else working
        available = [name for name, items in source.items() if items]
        if not available:
            break
        if phase_mode == "mixed":
            target_phase = "decode" if rng.random() < decode_ratio else "prefill"
            phase_available = [
                name
                for name in available
                if source[name][-1].phase == target_phase
            ]
            if phase_available:
                available = phase_available
        else:
            available = [
                name for name in available if source[name][-1].phase == phase_mode
            ]
            if not available:
                break
        available = [name for name in available if weights.get(name, 1.0) > 0]
        if not available:
            break
        selected = _weighted_dataset_choice(rng, available, weights)
        if not working[selected]:
            refill_epochs[selected] += 1
            working[selected] = list(originals[selected])
            random.Random(
                _stable_seed(seed, f"{selected}:refill:{refill_epochs[selected]}")
            ).shuffle(working[selected])
        item = working[selected].pop()
        source_item_id = item.source_item_id or item.item_id
        occurrence_key = (item.dataset, source_item_id)
        replica_index = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        if replica_index:
            item = replace(
                item,
                item_id=f"{source_item_id}:replica:{replica_index}",
                source_item_id=source_item_id,
                replica_index=replica_index,
            )
        sampled.append(item)
    if len(sampled) < num_requests:
        raise ValueError(
            f"requested {num_requests} unique rows but only sampled {len(sampled)} "
            f"for phase_mode={phase_mode}"
        )
    return sampled


def validate_frequency_penalty(value: Any) -> float:
    """Normalize the uniform generation penalty into a recordable float."""

    penalty = float(value)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError(
            f"frequency_penalty must be finite and non-negative, got {value!r}"
        )
    return penalty


def _prompt_hash(prompt_ids: list[int]) -> str:
    payload = json.dumps(prompt_ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_list(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    shape = getattr(tokenized, "shape", None)
    if shape is not None:
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, (list, tuple))
        and len(tokenized) == 1
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = tokenized[0]
    return [int(token) for token in tokenized]


def _render_prompt_ids(tokenizer: Any, item: WorkloadItem) -> list[int]:
    if item.prompt_kind == "chat_messages":
        if not isinstance(item.prompt, list):
            raise ValueError(f"{item.dataset}:{item.item_id} has invalid chat prompt")
        tokenized = tokenizer.apply_chat_template(
            item.prompt, tokenize=True, add_generation_prompt=True
        )
        return _token_list(tokenized)
    if item.prompt_kind == "raw_completion":
        if not isinstance(item.prompt, str):
            raise ValueError(f"{item.dataset}:{item.item_id} has invalid raw prompt")
        return _token_list(
            tokenizer.encode(item.prompt, add_special_tokens=False)
        )
    raise ValueError(
        f"{item.dataset}:{item.item_id} has unknown prompt kind {item.prompt_kind!r}"
    )


def render_workload_rows(
    items: list[WorkloadItem],
    phase_mode: str,
    tokenizer: Any,
    context_length: int,
    decode_output_policy: str = "remaining_model_context",
    fixed_decode_output_len: Optional[int] = None,
    sampling_overrides: Optional[dict[str, dict[str, Any]]] = None,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tokenizer is None:
        raise ValueError("a tokenizer is required for context-derived output budgets")
    if context_length <= 1:
        raise ValueError("context_length must be greater than one")
    penalty = validate_frequency_penalty(frequency_penalty)
    requests = []
    metadata = []
    for index, item in enumerate(items):
        request_id = f"{item.dataset}:{item.item_id}"
        source_item_id = item.source_item_id or item.item_id
        source_request_id = f"{item.dataset}:{source_item_id}"
        prompt_ids = _render_prompt_ids(tokenizer, item)
        prompt_len = len(prompt_ids)
        if prompt_len >= context_length:
            raise ValueError(
                f"{item.dataset}:{item.item_id} prompt has {prompt_len} tokens, "
                f"which does not fit context length {context_length}"
            )
        is_prefill_probe = phase_mode == "prefill" or (
            phase_mode == "mixed" and item.phase == "prefill"
        )
        if is_prefill_probe:
            output_len = 1
            output_policy = "prefill_probe_one_token"
        elif decode_output_policy == "remaining_model_context":
            output_len = context_length - prompt_len
            output_policy = "remaining_model_context"
        elif decode_output_policy == "task_reference_limit":
            if not item.task_reference_max_output_len:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} has no task reference limit"
                )
            output_len = min(
                context_length - prompt_len,
                int(item.task_reference_max_output_len),
            )
            output_policy = "task_reference_limit"
        elif decode_output_policy == "fixed_token_capacity":
            if fixed_decode_output_len is None or fixed_decode_output_len <= 0:
                raise ValueError(
                    "fixed_token_capacity requires a positive fixed output length"
                )
            if prompt_len + fixed_decode_output_len > context_length:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} has only "
                    f"{context_length - prompt_len} output positions, fewer than "
                    f"the fixed capacity length {fixed_decode_output_len}"
                )
            output_len = fixed_decode_output_len
            output_policy = "fixed_token_capacity"
        else:
            raise ValueError(
                f"unsupported decode output policy {decode_output_policy!r}"
            )
        extra_request_body: dict[str, Any] = {
            "temperature": 0.0,
            "ignore_eos": output_policy == "fixed_token_capacity",
        }
        if item.stop and output_policy != "fixed_token_capacity":
            extra_request_body["stop"] = item.stop
        sampling_policy = dict((sampling_overrides or {}).get(item.dataset, {}))
        if sampling_policy and output_policy == "fixed_token_capacity":
            raise ValueError(
                f"{item.dataset} sampling overrides cannot change fixed-token capacity"
            )
        extra_request_body.update(sampling_policy)
        if penalty != DEFAULT_FREQUENCY_PENALTY:
            if FREQUENCY_PENALTY_FIELD in extra_request_body:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} already carries a "
                    f"{FREQUENCY_PENALTY_FIELD} sampling override; the uniform "
                    "generation penalty must not be applied on top of it"
                )
            extra_request_body[FREQUENCY_PENALTY_FIELD] = penalty
        requests.append(
            {
                "request_id": request_id,
                "prompt": prompt_ids,
                "prompt_len": prompt_len,
                "output_len": output_len,
                "extra_request_body": extra_request_body,
            }
        )
        quality_eligible = (
            output_policy != "fixed_token_capacity"
            and item.metric != "selection_prefill_probe"
            and not (
                item.reference_output_len is not None
                and output_len < item.reference_output_len
                and item.metric == "rouge_l"
            )
        )
        metadata.append(
            {
                "index": index,
                "request_id": request_id,
                "source_request_id": source_request_id,
                "replica_index": item.replica_index,
                "dataset": item.dataset,
                "source_split": item.evaluator_data.get("source_split"),
                "phase": item.phase,
                "metric": item.metric,
                "gold": item.gold,
                "reference_output_len": item.reference_output_len,
                "requested_output_len": output_len,
                "prompt_len": prompt_len,
                "context_length": context_length,
                "output_policy": output_policy,
                "fixed_output_tokens": (
                    fixed_decode_output_len
                    if output_policy == "fixed_token_capacity"
                    else None
                ),
                "task_reference_max_output_len": (
                    item.task_reference_max_output_len
                ),
                "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
                "protocol_suite_id": PROTOCOL_SUITE_ID,
                "protocol_id": item.protocol_id,
                "quality_semantics": item.quality_semantics,
                "prompt_kind": item.prompt_kind,
                "quality_eligible": quality_eligible,
                "sampling_policy": sampling_policy,
                "prompt_sha256": _prompt_hash(prompt_ids),
                "evaluator_data": item.evaluator_data,
            }
        )
    return requests, metadata


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_workload(
    workload: str,
    num_requests: int,
    seed: int,
    tokenizer: Optional[Any],
    context_length: int,
    datasets_filter: Optional[list[str]] = None,
    phase_mode: Optional[str] = None,
    decode_ratio: float = 0.5,
    weights: Optional[dict[str, float]] = None,
    repeat_exhausted: bool = False,
    decode_output_policy: str = "remaining_model_context",
    fixed_decode_output_len: Optional[int] = None,
    sampling_overrides: Optional[dict[str, dict[str, Any]]] = None,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    include_train_pool: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if workload not in WORKLOAD_DATASETS:
        raise ValueError(f"unsupported workload {workload!r}")
    datasets_selected = tuple(datasets_filter or WORKLOAD_DATASETS[workload])
    unknown = sorted(set(datasets_selected) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    selected_phase = phase_mode or WORKLOAD_PHASE[workload]
    per_dataset_limit = num_requests
    pools = {
        name: load_dataset_items(
            name,
            per_dataset_limit,
            tokenizer,
            include_train_pool,
            context_length=context_length,
        )
        for name in datasets_selected
    }
    items = sample_items(
        pools,
        num_requests,
        selected_phase,
        decode_ratio,
        seed,
        weights,
        repeat_exhausted,
    )
    requests, metadata = render_workload_rows(
        items,
        selected_phase,
        tokenizer,
        context_length,
        decode_output_policy=decode_output_policy,
        fixed_decode_output_len=fixed_decode_output_len,
        sampling_overrides=sampling_overrides,
        frequency_penalty=frequency_penalty,
    )
    counts = Counter(row["dataset"] for row in metadata)
    phase_counts = Counter(row["phase"] for row in metadata)
    replica_counts = Counter(
        row["dataset"] for row in metadata if int(row["replica_index"]) > 0
    )
    source_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_ids: dict[str, set[str]] = defaultdict(set)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        dataset = str(row["dataset"])
        source_ids[dataset].add(str(row["source_request_id"]))
        prompt_hashes[dataset].add(str(row["prompt_sha256"]))
        if row.get("source_split"):
            source_split_counts[dataset][str(row["source_split"])] += 1
    summary = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_suite_id": PROTOCOL_SUITE_ID,
        "dataset_protocols": {
            name: DATASET_PROTOCOLS[name] for name in datasets_selected
        },
        "output_policy": {
            "decode": decode_output_policy,
            "fixed_output_tokens": fixed_decode_output_len,
            "prefill": "one_token_probe",
            "context_length": context_length,
        },
        "sampling_overrides": sampling_overrides or {},
        FREQUENCY_PENALTY_FIELD: validate_frequency_penalty(frequency_penalty),
        "workload": workload,
        "phase_mode": selected_phase,
        "decode_ratio_requested": decode_ratio,
        "seed": seed,
        "num_requests": len(requests),
        "datasets": dict(sorted(counts.items())),
        "phases": dict(sorted(phase_counts.items())),
        "quality_eligible": sum(bool(row["quality_eligible"]) for row in metadata),
        "repeat_exhausted": repeat_exhausted,
        "replicated_requests": sum(
            int(row["replica_index"] > 0) for row in metadata
        ),
        "replicated_requests_by_dataset": dict(sorted(replica_counts.items())),
        "source_splits": {
            dataset: dict(sorted(splits.items()))
            for dataset, splits in sorted(source_split_counts.items())
        },
        "unique_source_requests": len(
            {str(row["source_request_id"]) for row in metadata}
        ),
        "unique_source_requests_by_dataset": {
            dataset: len(values) for dataset, values in sorted(source_ids.items())
        },
        "unique_prompts": len({str(row["prompt_sha256"]) for row in metadata}),
        "unique_prompts_by_dataset": {
            dataset: len(values)
            for dataset, values in sorted(prompt_hashes.items())
        },
    }
    return requests, metadata, summary


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def numeric_exact_match(prediction: str, gold: str) -> float:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", prediction)
    predicted = numbers[-1].replace(",", "") if numbers else ""
    expected = str(gold).replace(",", "").strip()
    try:
        return float(abs(float(predicted) - float(expected)) < 1e-4)
    except Exception:
        return float(predicted == expected)


def gsm8k_strict_match(prediction: str, gold: str) -> float:
    match = re.search(r"#### (-?[0-9.,]+)", prediction)
    if match is None:
        return 0.0
    predicted = match.group(1).replace(",", "").rstrip(".")
    expected = str(gold).replace(",", "").rstrip(".").strip()
    return float(predicted == expected)


def gsm8k_cot_strict_match(prediction: str, gold: str) -> float:
    # lm-eval gsm8k-cot strict-match filter, pattern VERBATIM (the trailing
    # unescaped dot is part of the upstream pattern), take_first. The
    # regexes_to_ignore normalization removes exactly ONE trailing period
    # (\.$) — never rstrip, which would over-normalize "5.." to "5" and
    # score answers upstream rejects.
    match = re.search(r"The answer is (\-?[0-9\.\,]+).", prediction)
    if match is None:
        return 0.0
    predicted = re.sub(
        r"\.$", "", match.group(1).replace(",", "").replace("$", "")
    )
    expected = re.sub(r"\.$", "", str(gold).replace(",", "").strip())
    return float(predicted == expected)


def bbh_cot_exact_match(prediction: str, gold: str) -> float:
    # lm-eval bbh cot_fewshot filter "get-answer": regex
    # "(?<=the answer is )(.*)(?=.)" (greedy; the lookahead drops the final
    # character, i.e. the closing period), take_first, then exact_match on the
    # raw target (no case/punctuation normalisation in the task file).
    match = re.search(r"(?<=the answer is )(.*)(?=.)", prediction)
    if match is None:
        return 0.0
    return float(match.group(1) == str(gold))


def coqa_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    if len(gold_answers) <= 1:
        return max(token_f1(prediction, answer) for answer in gold_answers)
    total = 0.0
    for held_out in range(len(gold_answers)):
        references = gold_answers[:held_out] + gold_answers[held_out + 1 :]
        total += max(token_f1(prediction, answer) for answer in references)
    return total / len(gold_answers)


def _lcs_length(left: list[str], right: list[str]) -> int:
    state = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, 1):
            current = state[index]
            state[index] = (
                previous + 1
                if left_token == right_token
                else max(state[index], state[index - 1])
            )
            previous = current
    return state[-1]


def rouge_l(prediction: str, reference: str) -> float:
    predicted = prediction.split()
    expected = reference.split()
    if not predicted or not expected:
        return 0.0
    common = _lcs_length(predicted, expected)
    precision = common / len(predicted)
    recall = common / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def choice_accuracy(prediction: str, gold: dict[str, Any]) -> float:
    match = re.search(r"\b([A-Z])\b", prediction.upper())
    if match:
        return float(match.group(1) == str(gold["letter"]).upper())
    return float(_normalize_text(str(gold["text"])) in _normalize_text(prediction))


def humaneval_pass_at_1(
    prediction: str, evaluator_data: dict[str, Any], allow_code_execution: bool
) -> tuple[Optional[float], str]:
    if not allow_code_execution:
        return None, "code_execution_disabled"
    try:
        from humaneval_execution import check_correctness
    except (ImportError, ModuleNotFoundError):
        return None, "humaneval_executor_unavailable"
    problem = {
        "task_id": evaluator_data["task_id"],
        "prompt": evaluator_data["prompt"],
        "test": evaluator_data["test"],
        "entry_point": evaluator_data["entry_point"],
        "canonical_solution": evaluator_data.get("canonical_solution", ""),
    }
    result = check_correctness(problem, prediction, 3.0)
    return float(bool(result.get("passed"))), "scored"


def score_prediction(
    metadata: dict[str, Any], prediction: str, allow_code_execution: bool = False
) -> tuple[Optional[float], str]:
    if not metadata.get("quality_eligible", True):
        if metadata.get("metric") == "selection_prefill_probe":
            return None, "quality_scored_by_selection_evaluator"
        if metadata.get("output_policy") in {
            "fixed_token_capacity",
            "production_max_equal_work",
        }:
            return None, "capacity_workload_not_quality_evidence"
        return None, "quality_not_defined_for_prefill_truncation"
    metric = metadata["metric"]
    gold = metadata.get("gold")
    if metric == "gsm8k_strict_match":
        return gsm8k_strict_match(prediction, str(gold)), "scored"
    if metric == "gsm8k_cot_strict_match":
        return gsm8k_cot_strict_match(prediction, str(gold)), "scored"
    if metric == "bbh_cot_exact_match":
        return bbh_cot_exact_match(prediction, str(gold)), "scored"
    if metric == "gsm8k_cot_zeroshot_strict_match":
        # gsm8k-cot-zeroshot.yaml carries the SAME strict-match filter and
        # regexes_to_ignore as gsm8k-cot.yaml; distinct name for traceability.
        return gsm8k_cot_strict_match(prediction, str(gold)), "scored"
    if metric == "math_verify_offline":
        # MATH correctness (sympy is_equiv + math_verify) comes only from the
        # third-party harness offline; no in-repo reimplementation.
        return None, "offline_third_party_scoring"
    if metric in ("longbench_qa_f1_offline", "longbench_code_sim_offline"):
        # LongBench QA F1 / code edit-similarity come only from lm-eval's longbench
        # metrics offline (score_longbench_offline.py).
        return None, "offline_third_party_scoring"
    if metric in ("longwrite_offline", "lca_offline", "lcb_offline"):
        # LongBench-Write / LongWriter-6k S_l/S_q (LongWriter evaluation), LCA
        # ChrF/API recall (lca-baselines metrics) and LiveCodeBench pass@1 (the
        # benchmark's test harness) come only from the offline scorers.
        return None, "offline_third_party_scoring"
    if metric == "mmlu_pro_letter_match":
        return mmlu_pro_letter_match(prediction, str(gold)), "scored"
    if metric == "ifeval_offline":
        # IFEval verdicts come only from the third-party harness offline;
        # serving-side scoring is intentionally not defined.
        return None, "offline_third_party_scoring"
    if metric == "rouge_offline":
        # SCROLLS/LongBench ROUGE comes only from the third-party harness
        # (lm-eval `metrics.get_rouge_score`); no in-repo reimplementation.
        return None, "offline_third_party_scoring"
    if metric == "numeric_exact_match":
        return numeric_exact_match(prediction, str(gold)), "scored"
    if metric == "coqa_f1":
        first_line = prediction.strip().split("\n")[0]
        return coqa_f1(first_line, [str(answer) for answer in gold]), "scored"
    if metric == "token_f1":
        first_line = prediction.strip().split("\n")[0]
        return max(token_f1(first_line, str(answer)) for answer in gold), "scored"
    if metric == "choice_accuracy":
        return choice_accuracy(prediction, gold), "scored"
    if metric == "rouge_l":
        return max(rouge_l(prediction, str(reference)) for reference in gold), "scored"
    if metric == "humaneval_pass_at_1":
        return humaneval_pass_at_1(
            prediction, metadata.get("evaluator_data") or {}, allow_code_execution
        )
    raise ValueError(f"unsupported metric: {metric}")


def score_benchmark_record(
    record: dict[str, Any],
    metadata: list[dict[str, Any]],
    allow_code_execution: bool = False,
) -> dict[str, Any]:
    predictions = record.get("generated_texts") or []
    request_ids = record.get("request_ids") or []
    errors = record.get("errors") or []
    requested_output_lens = record.get("requested_output_lens") or []
    server_output_lens = record.get("server_reported_output_lens") or []
    retokenized_output_lens = record.get("retokenized_output_lens") or []
    effective_output_lens = record.get("output_lens") or []
    output_len_sources = record.get("output_len_sources") or []
    finish_reasons = record.get("finish_reasons") or []
    e2e_latencies = record.get("e2e_latencies") or []
    if request_ids:
        if len(request_ids) != len(predictions):
            raise ValueError(
                f"request_ids has {len(request_ids)} rows but predictions has "
                f"{len(predictions)}"
            )
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("benchmark request_ids are not unique")
        metadata_by_id = {row["request_id"]: row for row in metadata}
        if len(metadata_by_id) != len(metadata):
            raise ValueError("metadata request_ids are not unique")
        unknown = [
            request_id
            for request_id in request_ids
            if request_id not in metadata_by_id
        ]
        if unknown:
            raise ValueError(f"benchmark contains unknown request_ids: {unknown[:5]}")
        aligned_metadata = [metadata_by_id[request_id] for request_id in request_ids]
        alignment_source = "request_id"
    else:
        aligned_metadata = metadata
        alignment_source = "legacy_position"
    count = min(len(predictions), len(aligned_metadata))
    per_dataset: dict[str, list[float]] = defaultdict(list)
    per_source_split: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_request = []
    statuses = Counter()
    cap_hits = Counter()
    for index in range(count):
        meta = aligned_metadata[index]
        error = errors[index] if index < len(errors) else ""
        if error:
            score, status = None, "request_error"
        else:
            score, status = score_prediction(
                meta, str(predictions[index]), allow_code_execution
            )
        statuses[status] += 1
        if score is not None:
            per_dataset[meta["dataset"]].append(score)
            source_split = meta.get("source_split")
            if source_split:
                per_source_split[meta["dataset"]][str(source_split)].append(score)
        finish_reason = (
            finish_reasons[index] if index < len(finish_reasons) else None
        )
        if finish_reason == "length":
            cap_hits[meta["dataset"]] += 1
        per_request.append(
            {
                "index": index,
                "request_id": meta["request_id"],
                "source_request_id": meta.get("source_request_id"),
                "replica_index": meta.get("replica_index", 0),
                "dataset": meta["dataset"],
                "source_split": meta.get("source_split"),
                "metric": meta["metric"],
                "score": score,
                "status": status,
                "error": error,
                "prediction": predictions[index],
                "requested_output_len": (
                    requested_output_lens[index]
                    if index < len(requested_output_lens)
                    else None
                ),
                "server_output_len": (
                    server_output_lens[index]
                    if index < len(server_output_lens)
                    else None
                ),
                "retokenized_output_len": (
                    retokenized_output_lens[index]
                    if index < len(retokenized_output_lens)
                    else None
                ),
                "effective_output_len": (
                    effective_output_lens[index]
                    if index < len(effective_output_lens)
                    else None
                ),
                "output_len_source": (
                    output_len_sources[index]
                    if index < len(output_len_sources)
                    else None
                ),
                "finish_reason": finish_reason,
                "cap_hit": finish_reason == "length",
                "e2e_latency_s": (
                    e2e_latencies[index] if index < len(e2e_latencies) else None
                ),
            }
        )
    performance_fields = (
        "metric_accounting_version",
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "total_output_tokens_retokenized",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "output_throughput_retokenized",
        "total_throughput",
        "total_throughput_retokenized",
        "mean_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "concurrency",
        "max_concurrent_requests",
        "request_rate",
        "server_input_usage_reported",
        "declared_input_fallbacks",
        "server_usage_reported",
        "retokenized_fallbacks",
        "missing_finish_reasons",
        "cap_hit_count",
    )
    return {
        "tag": record.get("tag"),
        "dataset_name": record.get("dataset_name"),
        "alignment_source": alignment_source,
        "scored_requests": count,
        "status_counts": dict(sorted(statuses.items())),
        "cap_hits_by_dataset": dict(sorted(cap_hits.items())),
        "quality": {
            dataset: {
                "n": len(scores),
                "mean": sum(scores) / len(scores) if scores else math.nan,
            }
            for dataset, scores in sorted(per_dataset.items())
        },
        "quality_by_source_split": {
            dataset: {
                source_split: {
                    "n": len(scores),
                    "mean": sum(scores) / len(scores) if scores else math.nan,
                }
                for source_split, scores in sorted(splits.items())
            }
            for dataset, splits in sorted(per_source_split.items())
        },
        "performance": {
            field: record.get(field)
            for field in performance_fields
            if field in record
        },
        "per_request": per_request,
    }
