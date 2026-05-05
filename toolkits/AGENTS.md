# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-26
**Scope:** Pai-Megatron-Patch toolkits for data preprocessing, checkpoint conversion, and auto-configuration

## OVERVIEW
Training infrastructure utilities for data preparation, checkpoint format conversion, and automatic hyperparameter configuration.

## STRUCTURE
```
toolkits/
├── model_checkpoints_convertor/     # HF <-> Megatron weight conversion per model
├── pretrain_data_preprocessing/     # JSON to indexed binary dataset conversion
├── sft_data_preprocessing/          # Supervised fine-tuning data preprocessing
├── multimodal_data_preprocessing/   # Vision-language data preprocessing
├── distributed_checkpoints_convertor/  # Distributed checkpoint format conversion
└── auto_configurator/               # Automatic training hyperparameter configuration
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Convert HF weights for pretraining | `model_checkpoints_convertor/convert_<model>_hf_to_megatron.py` | Supports qwen, llama, deepseek, and 40+ others |
| Export Megatron weights for HF inference | `model_checkpoints_convertor/convert_<model>_megatron_to_hf.py` | Bidirectional conversion available |
| Preprocess pretraining data | `pretrain_data_preprocessing/preprocess_data.py` | JSON to indexed binary for streaming |
| Prepare SFT datasets | `sft_data_preprocessing/` | Conversation format templates and preprocessing |
| Multimodal training data | `multimodal_data_preprocessing/` | Image-text preprocessing for VL models |
| Migrate to distributed checkpoints | `distributed_checkpoints_convertor/` | Legacy to FSDP/TP format conversion |
| Auto-configure hyperparameters | `auto_configurator/` | Config generation based on model size and hardware |

## ANTI-PATTERNS (THIS TOOLKIT)
| Forbidden | Location | Reason |
|-----------|----------|--------|
| Manual tensor mapping in conversion scripts | Multiple converter files | Use model-specific mapping configs instead |
| Hardcoded output paths | All preprocessing scripts | Use command-line arguments for flexibility |
| Skipping validation steps | All converters | Always verify checkpoint integrity after conversion |