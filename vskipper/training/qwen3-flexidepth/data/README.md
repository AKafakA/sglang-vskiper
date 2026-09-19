# Training data stage: the tokenized alignment dataset

`prepare_qwen3_training_data.py` is the preprocessing script from the sealed FlexiDepth source archive
`Flexidepth-1ec394d-qwen3-data-rc6.tar.gz` (revision `1ec394d2`, sha256 `61e77a3c…`) that built the tokenized alignment dataset both
Qwen3 arms train on, `flexidepth-qwen3-alignment-b14afda6-d117af2f-seq2048-r2`: `allenai/tulu-3-sft-mixture` @ `b14afda6`, tokenized
with the Qwen3-8B tokenizer (`d117af2f`), max length 2,048, minimum length 15, `test_size` 0.01, seed 42 (939,343 source examples;
45,807 discarded over length; 893,536 retained = 884,600 train + 8,936 eval; 513,695,928 retained tokens). `full_stage_r2.sbatch`
(`STAGE=alignment`) is the wrapper as run on the cluster, `preflight_r2.sbatch` its dry run and `campaign_manifest_r2.json` the campaign
record. The dataset directory carries `DATASET_SHA256SUMS` (its sha256 `7fd1a8f2…` is pinned by `bin/run_arm.sh`) and
`PREPROCESSING_MANIFEST.json` with the counts above. The annealing dataset (`STAGE=annealing`, `af60f3c1`) comes from the same wrapper
and is not used by the paper's checkpoints (both arms are alignment-only).
