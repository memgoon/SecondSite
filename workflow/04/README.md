# 04. Sensitivity analyses and uncertainty

Evaluate scaffold exclusion and canonical-sequence representation separately. The standalone pooled bootstrap is included here. Report pooled and conditional metrics with their own denominators.

Original implementation names are retained for source provenance. These files are provided for methodological inspection, not as a self-contained execution environment.

- [murcko_holdout_extension/scripts/aggregate.py](../../source/analysis/murcko_holdout_extension/scripts/aggregate.py)
- [murcko_holdout_extension/scripts/base_trainer.py](../../source/analysis/murcko_holdout_extension/scripts/base_trainer.py)
- [murcko_holdout_extension/scripts/build_contract.py](../../source/analysis/murcko_holdout_extension/scripts/build_contract.py)
- [murcko_holdout_extension/scripts/check_contract.py](../../source/analysis/murcko_holdout_extension/scripts/check_contract.py)
- [murcko_holdout_extension/scripts/common.py](../../source/analysis/murcko_holdout_extension/scripts/common.py)
- [murcko_holdout_extension/scripts/extract_return.py](../../source/analysis/murcko_holdout_extension/scripts/extract_return.py)
- [murcko_holdout_extension/scripts/gpu_preflight.py](../../source/analysis/murcko_holdout_extension/scripts/gpu_preflight.py)
- [murcko_holdout_extension/scripts/model_definitions.py](../../source/analysis/murcko_holdout_extension/scripts/model_definitions.py)
- [murcko_holdout_extension/scripts/run_gpu.sh](../../source/analysis/murcko_holdout_extension/scripts/run_gpu.sh)
- [murcko_holdout_extension/scripts/status.py](../../source/analysis/murcko_holdout_extension/scripts/status.py)
- [murcko_holdout_extension/scripts/train_gpu.py](../../source/analysis/murcko_holdout_extension/scripts/train_gpu.py)
- [murcko_holdout_extension/scripts/trainer_core.py](../../source/analysis/murcko_holdout_extension/scripts/trainer_core.py)
- [murcko_holdout_extension/scripts/validate_fits.py](../../source/analysis/murcko_holdout_extension/scripts/validate_fits.py)
- [murcko_holdout_extension/README.md](../../source/analysis/murcko_holdout_extension/README.md)
- [murcko_holdout_extension/requirements-cpu.txt](../../source/analysis/murcko_holdout_extension/requirements-cpu.txt)
- [murcko_holdout_extension/requirements-gpu.txt](../../source/analysis/murcko_holdout_extension/requirements-gpu.txt)
- [full_sequence_representation_pilot/scripts/aggregate_pilot.py](../../source/analysis/full_sequence_representation_pilot/scripts/aggregate_pilot.py)
- [full_sequence_representation_pilot/scripts/build_cpu_contract.py](../../source/analysis/full_sequence_representation_pilot/scripts/build_cpu_contract.py)
- [full_sequence_representation_pilot/scripts/embed_full_sequences_gpu.sh](../../source/analysis/full_sequence_representation_pilot/scripts/embed_full_sequences_gpu.sh)
- [full_sequence_representation_pilot/scripts/gpu_preflight.py](../../source/analysis/full_sequence_representation_pilot/scripts/gpu_preflight.py)
- [full_sequence_representation_pilot/scripts/run_gpu.sh](../../source/analysis/full_sequence_representation_pilot/scripts/run_gpu.sh)
- [full_sequence_representation_pilot/scripts/stitch_full_sequence_embeddings.py](../../source/analysis/full_sequence_representation_pilot/scripts/stitch_full_sequence_embeddings.py)
- [full_sequence_representation_pilot/scripts/train_full_sequence.py](../../source/analysis/full_sequence_representation_pilot/scripts/train_full_sequence.py)
- [full_sequence_representation_pilot/README.md](../../source/analysis/full_sequence_representation_pilot/README.md)
- [full_sequence_representation_pilot/EXPERIMENT_CONTRACT.md](../../source/analysis/full_sequence_representation_pilot/EXPERIMENT_CONTRACT.md)
- [double_anchored_pooled_bootstrap/bootstrap_pooled.py](../../source/analysis/double_anchored_pooled_bootstrap/bootstrap_pooled.py)
