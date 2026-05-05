# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
from contextlib import nullcontext
import torch
import torch._dynamo
import inspect

from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron_patch.tokenizer import build_tokenizer
"""
from megatron_patch.model.qwen3_moe.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
"""
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import train_valid_test_datasets_provider
from megatron.training import get_args, pretrain, print_rank_0

torch._dynamo.config.suppress_errors = True


# ModelPerf Hook integration (CPU environment, no GPU required)
def _setup_modelperf_hook():
    import sys
    import os
    modelperf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'ModelPerf'
    )
    if modelperf_path not in sys.path:
        sys.path.insert(0, modelperf_path)
    try:
        from modelperf.framework_adapter import register_pai_patch_hooks
        export_dir = os.path.join(modelperf_path, 'examples', 'output', 'captured_configs')
        os.makedirs(export_dir, exist_ok=True)
        hook_manager = register_pai_patch_hooks(export_dir=export_dir)
        print_rank_0(f'[ModelPerf] Hook registered. Configs will export to {export_dir}')
        return hook_manager
    except Exception as e:
        print_rank_0(f'[ModelPerf] Warning: Failed to register hook: {e}')
        return None


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel]: The returned model
    """
    args = get_args()

    # ModelPerf: extract configs directly from parsed args (most reliable method)
    try:
        import sys
        import os
        modelperf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'ModelPerf'
        )
        if modelperf_path not in sys.path:
            sys.path.insert(0, modelperf_path)
        from modelperf.framework_adapter.config_extractor import ConfigExtractor
        extractor = ConfigExtractor()
        extractor.extract_from_args(args)
        export_dir = os.path.join(modelperf_path, 'examples', 'output', 'captured_configs')
        os.makedirs(export_dir, exist_ok=True)
        extractor.export_json(export_dir)
        print_rank_0(f'[ModelPerf] Configs extracted and exported to {export_dir}')
    except Exception as e:
        print_rank_0(f'[ModelPerf] Warning: Failed to extract configs: {e}')
    build_tokenizer(args)
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building QWen3 model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
        else:
            # Define the decoder layer spec
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                    normalization=args.normalization)
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

    _setup_modelperf_graph_capture(model)

    return model


# ModelPerf: capture computational graph during training
def _setup_modelperf_graph_capture(model):
    import sys
    import os
    modelperf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'ModelPerf'
    )
    if modelperf_path not in sys.path:
        sys.path.insert(0, modelperf_path)
    try:
        from modelperf.capture.module_hook import ModuleCapture
        from modelperf.capture.comm_hook import install_communication_hooks
        from modelperf.capture.graph import ComputationalGraph

        graph = ComputationalGraph()

        module_capture = ModuleCapture(graph)
        module_capture.capture(model)

        comm_capture = install_communication_hooks(graph)

        _modelperf_captures['module'] = module_capture
        _modelperf_captures['comm'] = comm_capture

        module_capture.start()
        comm_capture.start()

        print_rank_0(f'[ModelPerf] Graph capture started: ModuleCapture + CommunicationCapture')
    except Exception as e:
        print_rank_0(f'[ModelPerf] Warning: Failed to start graph capture: {e}')


def _stop_modelperf_graph_capture():
    try:
        if 'module' in _modelperf_captures:
            _modelperf_captures['module'].stop()
        if 'comm' in _modelperf_captures:
            _modelperf_captures['comm'].stop()

        import os
        import json
        modelperf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'ModelPerf'
        )
        export_dir = os.path.join(modelperf_path, 'examples', 'output', 'captured_graph')
        os.makedirs(export_dir, exist_ok=True)

        graph = _modelperf_captures.get('module')
        if graph is not None:
            graph_obj = graph.get_graph()
            graph_path = os.path.join(export_dir, 'computational_graph.json')
            with open(graph_path, 'w') as f:
                json.dump(graph_obj.to_dict(), f, indent=2)
            print_rank_0(f'[ModelPerf] Graph exported to {graph_path}: '
                         f'{len(graph_obj.nodes)} nodes, {len(graph_obj.get_comm_nodes())} comm')
    except Exception as e:
        print_rank_0(f'[ModelPerf] Warning: Failed to export graph: {e}')


_modelperf_captures = {}

if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    _modelperf_hook_manager = _setup_modelperf_hook()

    try:
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            extra_args_provider=get_patch_args,
        )
    finally:
        if _modelperf_hook_manager is not None:
            _modelperf_hook_manager.uninstall()
            print_rank_0('[ModelPerf] Hook uninstalled')
        _stop_modelperf_graph_capture()