# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    `residual_linear_shared` (fc_embedding + shared fc_hidden) instead of
    the `Linear(2H, H)` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

import weakref
from collections.abc import Iterable, Sequence
from typing import Any

import regex as re
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm import envs
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.kernels.triton.fused_pre_fc_norms import fused_pre_fc_norms_split
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    SupportsHCSP,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    get_fused_shared_expert_layers,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_mtp_feedback,
    register_b12x_layer,
    set_b12x_preparation_provider,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..common.hyperconnection import is_hc_sequence_parallel_enabled
from .backend import uses_b12x
from .hyperconnection import (
    B12xGatedResidual,
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
    HyperConnectionWorkspace,
)
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
    _remap_qsa_cache_scale_name,
    get_qsa_topk_buffer_width,
)


def _mtp_api() -> Any:
    api = get_b12x_mtp_feedback()
    if api is None:
        raise ImportError(
            "Qwen4Exp MTP requires b12x.sequence.mtp_feedback; "
            "install the b12x serving extra"
        )
    return api


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""
    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )
        if draft_quant_config.get_name() == "quark":
            draft_quant_config.apply_vllm_mapper(
                WeightsMapper(
                    orig_to_new_prefix={
                        f"mtp.layers.{i}.": f"mtp.layers.{mtp_start_layer_idx + i}."
                        for i in range(num_mtp_layers)
                    }
                )
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self._use_b12x = uses_b12x(vllm_config)
        # MoE SP also enables HC SP, while HC SP can be enabled independently.
        self.use_hc_sequence_parallel = is_hc_sequence_parallel_enabled(vllm_config)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        self.supports_mtp_prefill_compaction = (
            self._use_b12x
            and self.num_mtp_layers == 1
            and not self.use_hc_sequence_parallel
            and envs.VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT
            and vllm_config.scheduler_config.max_num_seqs == 1
        )
        self._prefill_output_indices: torch.Tensor | None = None

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_workspace = (
            HyperConnectionWorkspace(
                hc_config, vllm_config.scheduler_config.max_num_batched_tokens
            )
            if uses_b12x(vllm_config)
            else None
        )
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
            self.num_mtp_layers,
        )
        # The proposer replaces this with the target model's buffer.
        topk_indices_buffer = None
        if getattr(config, "indexer_n_heads", None) is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                get_qsa_topk_buffer_width(config),
                dtype=torch.int32,
            )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            projection_cls = (
                ReplicatedLinear if self._use_b12x else ColumnParallelLinear
            )
            projection_kwargs = {} if self._use_b12x else {"gather_output": True}
            self.fc_embedding = projection_cls(
                self.hidden_size,
                self.hidden_size,
                **projection_kwargs,
                bias=False,
                return_bias=False,
                quant_config=None if self._use_b12x else draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = projection_cls(
                self.hidden_size,
                self.hidden_size,
                **projection_kwargs,
                bias=False,
                return_bias=False,
                quant_config=None if self._use_b12x else draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="qwen_sparse_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                    topk_indices_buffer=topk_indices_buffer,
                    workspace=self.hyper_connection_workspace,
                )
                for idx in range(self.num_mtp_layers)
            )
        self.fused_shared_expert_layers = get_fused_shared_expert_layers(
            self,
            Qwen4ExpSparseMoeBlock,
            enable_eplb=vllm_config.parallel_config.enable_eplb,
        )

        if self._use_b12x:
            self.pre_fc_norm_embedding = GroupedGemmaRMSNorm(
                self.hidden_size,
                eps=config.rms_norm_eps,
                group_size=None,
                dtype=torch.bfloat16,
            )
            self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(
                self.hidden_size * self.hc_count,
                eps=config.rms_norm_eps,
                group_size=None,
                dtype=torch.bfloat16,
            )
        else:
            self.pre_fc_norm_embedding = GemmaRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_fc_norm_hidden = GemmaRMSNorm(
                self.hidden_size * self.hc_count, eps=config.rms_norm_eps
            )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        mixer_cls = (
            B12xGatedResidual
            if self.hyper_connection_workspace is not None
            else GatedResidual
        )
        mixer_kwargs = (
            {"workspace": self.hyper_connection_workspace}
            if self.hyper_connection_workspace is not None
            else {}
        )
        self.hyper_connection_mixer = mixer_cls(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            **mixer_kwargs,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

        if not self._use_b12x:
            return
        device = torch.device(current_platform.current_device())
        if self.supports_mtp_prefill_compaction:
            self.register_buffer(
                "_decode_output_indices",
                torch.zeros(1, dtype=torch.int64, device=device),
                persistent=False,
            )
            self._prefill_output_indices = self._decode_output_indices
        self._feedback_caps = _mtp_api().Caps(
            device=device,
            max_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            hidden_size=self.hidden_size,
            streams=self.hc_count,
            dtype=torch.bfloat16,
        )
        self._plan = None
        self._feedback_scratch_spec: tuple[tuple[int, ...], torch.dtype] | None = None
        self.register_buffer("_feedback_token_embedding", None, persistent=False)
        self.register_buffer("_feedback_multi_state", None, persistent=False)
        self.register_buffer("_feedback_output", None, persistent=False)
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.prefix = prefix
        b12x_name = b12x_layer_prefix(self)
        self._b12x_layer_name = _encode_layer_name(b12x_name)
        register_b12x_layer(b12x_name, self)
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_prefill_output_indices(self, output_indices: torch.Tensor | None) -> None:
        if not self.supports_mtp_prefill_compaction:
            return
        self._prefill_output_indices = (
            self._decode_output_indices if output_indices is None else output_indices
        )

    def snapshot_qsa_interval_starts(self) -> None:
        for layer in self.layers:
            snapshot = getattr(
                getattr(layer, "self_attn", None),
                "snapshot_speculative_interval_starts",
                None,
            )
            if snapshot is not None:
                snapshot()

    def restore_qsa_interval_starts(self) -> None:
        for layer in self.layers:
            restore = getattr(
                getattr(layer, "self_attn", None),
                "restore_speculative_interval_starts",
                None,
            )
            if restore is not None:
                restore()

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""
        for attention in self._iter_qsa_attentions():
            set_skip = getattr(attention, "set_skip_topk", None)
            if set_skip is not None:
                set_skip(skip and self.num_mtp_layers == 1)
            else:
                attention.indexer.skip_topk = skip and self.num_mtp_layers == 1

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""
        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            compact = getattr(attention, "compact_topk_indices", None)
            if compact is not None:
                compact(row_indices)
            else:
                buffer = attention.topk_indices_buffer
                selected = buffer.index_select(0, row_indices)
                buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def _feedback_request_name(self) -> str:
        return f"{self.prefix}.mtp.feedback"

    @staticmethod
    def _alignment(tensor: torch.Tensor) -> int:
        pointer = tensor.data_ptr()
        return min(16, pointer & -pointer) if pointer else 16

    def _feedback_plan(self):
        return _mtp_api().plan(
            self._feedback_caps,
            invocation={
                "input_alignments": (
                    16,
                    16,
                    self._alignment(self.pre_fc_norm_embedding.weight),
                    self._alignment(self.pre_fc_norm_hidden.weight),
                )
            },
        )

    def _feedback_resources_ready(self) -> bool:
        weights = (
            self.pre_fc_norm_embedding.weight,
            self.pre_fc_norm_hidden.weight,
            self.fc_embedding.weight,
            self.fc_hidden.weight,
        )
        return not any(weight.is_meta for weight in weights)

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if layer is not self:
            raise ValueError("MTP preparation owner mismatch")
        if not self._use_b12x or workload.stage != "weights":
            return ()
        if not self._feedback_resources_ready():
            return ()
        if workload.max_tokens > self._feedback_caps.max_tokens:
            raise PreparationResourceUnavailableError(
                f"{self.prefix} MTP feedback capacity "
                f"{self._feedback_caps.max_tokens} cannot serve "
                f"{workload.max_tokens} tokens"
            )
        plan = self._feedback_plan()
        self._plan = plan
        request = plan.request(
            name=self._feedback_request_name,
            prepare_call=self._prepare_feedback_call,
            benchmark_call=self._feedback_benchmark_factory(),
        )
        return (
            B12xPreparationUnit(
                name="MTP_FEEDBACK",
                key=(self.prefix, workload.max_tokens),
                requests=(request,),
                stage="weights",
            ),
        )

    def _publish_feedback_storage(self, state) -> None:
        """Publish fixed feedback storage while keeping arena views transient."""
        (spec,) = state.layout.scratch_specs()
        self._feedback_scratch_spec = (spec.shape, spec.dtype)
        factory = dict(device=spec.device, dtype=torch.bfloat16)
        tokens = state.layout.caps.max_tokens
        self._feedback_token_embedding = torch.empty(
            (tokens, self.hidden_size), **factory
        )
        self._feedback_multi_state = torch.empty(
            (tokens, self.hc_count, self.hidden_size), **factory
        )
        self._feedback_output = torch.empty(
            state.layout.output_storage_shape(), **factory
        )

    def _prepare_feedback_call(self, state):
        self._publish_feedback_storage(state)
        token_embedding = self._feedback_token_embedding
        multi_state = self._feedback_multi_state
        output = self._feedback_output
        assert token_embedding is not None
        assert multi_state is not None
        assert output is not None
        assert self._feedback_scratch_spec is not None
        # Trial scratch belongs to the trial; the serving workspace is drawn
        # from only inside the feedback op body.
        shape, dtype = self._feedback_scratch_spec
        scratch = torch.empty(shape, dtype=dtype, device=token_embedding.device)
        return self._feedback_call(
            state,
            scratch=scratch,
            token_embedding=token_embedding,
            multi_state=multi_state,
            output=output,
        )

    def _feedback_benchmark_factory(self):
        shared = None

        def factory(state):
            nonlocal shared
            tensors = None if shared is None else tuple(ref() for ref in shared)
            if tensors is None or any(tensor is None for tensor in tensors):
                (spec,) = state.layout.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                tensor_factory = dict(device=spec.device, dtype=torch.bfloat16)
                tokens = state.layout.caps.max_tokens
                token_embedding = torch.empty(
                    (tokens, self.hidden_size), **tensor_factory
                )
                multi_state = torch.empty(
                    (tokens, self.hc_count, self.hidden_size), **tensor_factory
                )
                output = torch.empty(
                    state.layout.output_storage_shape(), **tensor_factory
                )
                tensors = scratch, token_embedding, multi_state, output
                shared = tuple(weakref.ref(tensor) for tensor in tensors)
            scratch, token_embedding, multi_state, output = tensors
            return self._feedback_call(
                state,
                scratch=scratch,
                token_embedding=token_embedding,
                multi_state=multi_state,
                output=output,
            )

        return factory

    def _feedback_call(
        self,
        state,
        *,
        scratch,
        token_embedding,
        multi_state,
        output,
    ):
        from b12x.preparation import PreparedCall

        def produce() -> None:
            # Prime the installed fixed route with real-shaped, nonzero
            # activations while borrowing the loaded norm and FC weights.
            token_embedding.fill_(0.125)
            multi_state.fill_(0.25)

        def reset() -> None:
            scratch.zero_()
            output.zero_()

        def run():
            return state.run_tensors(
                token_embedding,
                multi_state,
                self.pre_fc_norm_embedding.weight,
                self.pre_fc_norm_hidden.weight,
                self.fc_embedding.weight,
                self.fc_hidden.weight,
                scratch,
                output,
                eps=self.config.rms_norm_eps,
            )

        return PreparedCall(
            run=run,
            produce=produce,
            reset=reset,
            capture_safe=False,
        )

    def _run_feedback(
        self,
        token_embedding: torch.Tensor,
        multi_state: torch.Tensor,
    ) -> None:
        plan = self._plan
        if plan is None or self._feedback_scratch_spec is None:
            raise PreparationResourceUnavailableError(
                f"{self.prefix} MTP feedback is not prepared"
            )
        from vllm.v1.worker.workspace import current_workspace_manager

        (scratch,) = current_workspace_manager().get_simultaneous(
            self._feedback_scratch_spec
        )
        num_tokens = token_embedding.shape[0]
        if num_tokens > self._feedback_caps.max_tokens:
            raise ValueError(
                "Qwen4Exp MTP feedback capacity exceeded: "
                f"{num_tokens}/{self._feedback_caps.max_tokens} tokens"
            )
        multi_state = multi_state.reshape(num_tokens, self.hc_count, self.hidden_size)
        self._feedback_token_embedding[:num_tokens].copy_(token_embedding)
        self._feedback_multi_state[:num_tokens].copy_(multi_state)
        binding = _mtp_api().bind(
            plan,
            scratch=scratch,
            token_embedding=self._feedback_token_embedding,
            multi_state=self._feedback_multi_state,
            token_norm_weight=self.pre_fc_norm_embedding.weight,
            state_norm_weight=self.pre_fc_norm_hidden.weight,
            embedding_fc_weight=self.fc_embedding.weight.data,
            hidden_fc_weight=self.fc_hidden.weight.data,
            output=self._feedback_output,
            tokens=num_tokens,
        )
        _mtp_api().run(binding, eps=self.config.rms_norm_eps)

    def _prepare_feedback(
        self,
        token_embedding: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_b12x:
            embedding = self.fc_embedding(self.pre_fc_norm_embedding(token_embedding))
            multi_state = self.pre_fc_norm_hidden(hidden_states).reshape(
                -1, self.hc_count, self.hidden_size
            )
            return (self.fc_hidden(multi_state) + embedding.unsqueeze(1)).flatten(-2)
        if self._plan is None or self._feedback_output is None:
            raise PreparationResourceUnavailableError(
                f"{self.prefix} MTP feedback is not prepared"
            )
        if torch.compiler.is_compiling():
            torch.ops.vllm.qwen4_exp_b12x_mtp_feedback(
                token_embedding,
                hidden_states,
                self._feedback_output,
                self._b12x_layer_name,
            )
        else:
            self._run_feedback(token_embedding, hidden_states)
        return self._feedback_output[: token_embedding.shape[0]].flatten(-2)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        assert hidden_states is not None
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        if self._use_b12x:
            hidden_states = self._prepare_feedback(inputs_embeds, hidden_states)
            prev_block_output = None
        else:
            # Both pre-FC norms in one launch; the hidden norm is independent of
            # fc_embedding, so it can be hoisted above it.
            num_tokens = hidden_states.shape[0]
            inputs_embeds, hidden_states = fused_pre_fc_norms_split(
                inputs_embeds,
                hidden_states.view(num_tokens, hc_count * hidden_size),
                self.pre_fc_norm_embedding.weight,
                self.pre_fc_norm_hidden.weight,
                self.pre_fc_norm_embedding.variance_epsilon,
            )
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.fc_hidden(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            prev_block_output = inputs_embeds

        current_step_idx = spec_step_idx % self.num_mtp_layers
        full_num_tokens = positions.shape[-1]
        if self.use_hc_sequence_parallel:
            if (
                self.use_sequence_parallel
                and envs.VLLM_MOE_SKIP_PADDING
                and is_forward_context_available()
            ):
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            # The FC output is already complete and needs only a slice.
            if prev_block_output is not None:
                prev_block_output = sp_shard(prev_block_output)
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=prev_block_output,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
            output_indices=self._prefill_output_indices,
        )

        # Finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        if self.use_hc_sequence_parallel:
            # Gather both outputs together, then restore their contiguous layouts.
            packed_hidden_states = torch.cat(
                [sample_hidden_states, multi_hidden], dim=-1
            )
            packed_hidden_states = sp_all_gather(packed_hidden_states)[:full_num_tokens]
            sample_hidden_states, multi_hidden = packed_hidden_states.split(
                [hidden_size, hc_count * hidden_size], dim=-1
            )
            sample_hidden_states = sample_hidden_states.contiguous()
            multi_hidden = multi_hidden.contiguous()
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            fused_shared_expert_layers=self.fused_shared_expert_layers,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpMTP(
    LocalArgmaxMixin, nn.Module, SupportsHCSP, SupportsPP, Qwen4ExpMixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        if not uses_b12x(vllm_config):
            enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        layer_ids = frozenset(
            range(
                self.model.mtp_start_layer_idx,
                self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
            )
        )

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    if uses_b12x(self.vllm_config):
                        remapped_name = _remap_qsa_cache_scale_name(
                            remapped_name, layer_ids
                        )
                    yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(remap_weight_names(), mapper=mapper)


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]


def _mtp_feedback_op(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    layer._run_feedback(token_embedding, multi_state)


def _mtp_feedback_fake(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_b12x_mtp_feedback",
    op_func=_mtp_feedback_op,
    mutates_args=["output"],
    fake_impl=_mtp_feedback_fake,
)
