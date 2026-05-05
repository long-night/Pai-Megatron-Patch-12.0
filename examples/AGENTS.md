# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-26
**Scope:** Model-specific training examples for 25+ LLM families in Pai-Megatron-Patch-12.0

## OVERVIEW
Model-specific training scripts and configs for popular Chinese and open-source LLMs, each organized in its own directory with pretrain/finetune/eval pipelines.

## STRUCTURE
```
Pai-Megatron-Patch-12.0/examples/
├── llama/              # LLaMA-1 (65B)
├── llama2/             # LLaMA-2 (7B, 13B, 70B)
├── llama3/             # LLaMA-3 (8B, 70B)
├── llama3_1/           # LLaMA-3.1 (405B)
├── qwen/               # Qwen-1 (7B, 14B, 72B)
├── qwen1_5/            # Qwen-1.5 (0.5B-72B)
├── qwen2/              # Qwen-2 (0.5B-72B)
├── qwen2_5/            # Qwen-2.5 (0.5B-72B)
├── qwen3/              # Qwen-3 + GRPO RL
├── deepseek/           # DeepSeek-1 (67B)
├── deepseek_v2/        # DeepSeek-V2 MoE (236B)
├── deepseek_v3/        # DeepSeek-V3 MoE (671B) + GRPO RL
├── baichuan/           # Baichuan-1 (7B, 13B)
├── baichuan2/          # Baichuan-2 (7B, 13B)
├── bloom/              # BLOOM (176B)
├── falcon/             # Falcon (40B, 180B)
├── mistral/            # Mistral (7B)
├── chatglm/            # ChatGLM (6B, 6B-V2, 6B-V3)
├── codellama/          # Code LLaMA (7B, 13B, 34B)
├── llava/              # LLaVA vision-language
└── llava_mcore/        # LLaVA with Megatron-Core refactor
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| LLaMA examples | `examples/llama3/` | Standard dense model pattern |
| Qwen examples | `examples/qwen2_5/` | Includes Chinese-specific tokenizers |
| DeepSeek MoE examples | `examples/deepseek_v3/` | Expert routing + load balancing |
| GRPO reinforcement learning | `examples/qwen3/` or `examples/deepseek_v3/` | Reward model training, policy updates |
| Multimodal configs | `examples/llava/` | Vision encoder + cross-attention |

## CONVENTIONS
- **Standard trio per model**: `pretrain.sh`, `finetune.sh`, `evaluate.sh` with model-specific arguments
- **MoE-specific scripts**: DeepSeek-V2/V3 use `--expert-model-parallel-size` and auxiliary loss configs
- **Megatron-Core refactor**: `llava_mcore/` uses new APIs, `llava/` uses legacy interfaces
- **GRPO integration**: Policy gradient scripts in select models (Qwen-3, DeepSeek-V3)

## ANTI-PATTERNS (THIS MODULE)
| Forbidden | Location | Reason |
|-----------|----------|--------|
| Copying scripts between models | Any example directory | Maintain model-specific configs |
| Using dense model scripts for MoE | `deepseek_v2/`, `deepseek_v3/` | MoE requires expert routing args |
| Hardcoding paths in shell scripts | All `*.sh` files | Use `$MEGATRON_ROOT` variables |
| Mixing vision configs for text-only models | `llava/`, `llava_mcore/` | Vision params cause crashes in text models |