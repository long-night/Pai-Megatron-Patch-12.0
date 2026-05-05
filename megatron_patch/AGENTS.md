# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-26
**Scope:** Pai-Megatron-Patch megatron_patch module

## OVERVIEW
Monkey-patch overlay that extends Megatron-LM without invasive modifications.

## STRUCTURE
```
megatron_patch/
├── model/                     # Model implementations (Qwen, LLaMA, DeepSeek, etc.)
├── data/                      # Custom dataset implementations
├── tokenizer/                 # Custom tokenizers
├── generation/                # Text generation utilities
├── template/                  # Model config template system
├── fixes/                     # Bug fixes and compatibility patches
├── arguments.py               # CLI argument patches via get_patch_args()
├── training.py                # Training loop extensions
├── tensor_parallel.py         # Tensor parallelism patches
├── finetune_utils.py          # Fine-tuning utilities
├── lm_evaluate.py             # Language model evaluation
└── initialize.py              # Patch initialization
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model | `model/<model>/` | Follow existing model directory pattern |
| Patch training args | `arguments.py:get_patch_args()` | Extends Megatron's argparse |
| Extend training loop | `training.py` | Overrides/extends core training logic |
| Custom tensor parallel | `tensor_parallel.py` | Patches parallel state management |
| Fine-tuning utilities | `finetune_utils.py` | LoRA, adapter fine-tuning helpers |
| Model config templates | `template/helper.py` | Dynamic config generation |
| LM evaluation | `lm_evaluate.py` | Evaluation harness for patched models |

## ANTI-PATTERNS (THIS MODULE)
| Forbidden | Location | Reason |
|-----------|----------|--------|
| Direct modifications to Megatron-LM source | Any | Breaks monkey-patch architecture |
| Duplicate argument definitions | `arguments.py` | Use `patch_if_not_exist()` helper |
| Hardcoded model paths | `template/` | Use template system for flexibility |
| Bypassing patch initialization | Any | Always import via `initialize.py` |