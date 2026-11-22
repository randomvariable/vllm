# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 native dense weights with checkpoint-compressed routed scales."""

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.model_loader.mxfp4_csf_loader import read_mxfp4_csf_layer
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_default_torch_num_threads

from ..quant_config import _EXPERTS_LAYER_RE, DeepseekV4FP8Config


class DeepseekV41Mxfp4CsfConfig(DeepseekV4FP8Config):
    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") == cls.get_name()
        ):
            return "mxfp4_csf"
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        result = super().from_config({**config, "quant_method": "fp8"})
        assert isinstance(result, cls)
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            config = get_current_vllm_config()
            match = _EXPERTS_LAYER_RE.search(prefix)
            if (
                match
                and int(match[1]) < config.model_config.hf_text_config.num_hidden_layers
            ):
                p = config.parallel_config
                if (
                    not current_platform.is_cuda()
                    or config.kernel_config.moe_backend != "b12x"
                    or config.load_config.load_format != "mxfp4_csf"
                    or p.pipeline_parallel_size != 1
                    or p.data_parallel_size != 1
                    or p.use_ubatching
                    or p.enable_expert_parallel
                    or p.enable_eplb
                ):
                    raise ValueError(
                        "MXFP4-CSF requires CUDA b12x, the mxfp4_csf loader, and TP "
                        "without PP, DP, EP or microbatching"
                    )
                return Mxfp4CsfMoEMethod(layer.moe_config, self, int(match[1]))
        return super().get_quant_method(layer, prefix)


class Mxfp4CsfMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, owner, layer_index):
        super().__init__(moe)
        self.owner = owner
        self.layer_index = layer_index
        self.a16 = envs.VLLM_B12X_MOE_FP4_FORCE_A16

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16 or self.moe.has_bias:
            raise ValueError("MXFP4-CSF requires bias-free BF16 experts")
        self.geometry = num_experts, hidden_size, intermediate_size_per_partition
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        activation = None if self.a16 else "mxfp8"
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(dtype=activation),
            _a2=FusedMoEQuantDesc(dtype=activation),
            _w1=FusedMoEQuantDesc(dtype="mxfp4"),
            _w2=FusedMoEQuantDesc(dtype="mxfp4"),
        )

    def process_weights_after_loading(self, layer):
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        e, h, n = self.geometry
        device = layer.w13_weight.device
        shapes = ((e, h // 32, 2 * n), (e, n // 32, h))
        if not self.a16 and n % 128 != 64:
            shapes = (
                (e, 2 * round_up(n, 128), round_up(h // 32, 4)),
                (e, round_up(h, 256), round_up(n // 32, 4)),
            )
        scratch = self.owner.scale_scratch
        if scratch is None:
            # TP-only layers execute serially on each rank and share this storage.
            scratch = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )
            self.owner.scale_scratch = scratch
        elif any(
            t.shape != s or t.device != device
            for t, s in zip(scratch, shapes, strict=True)
        ):
            raise ValueError("MXFP4-CSF shared scratch geometry or device changed")
        with set_default_torch_num_threads(1):
            weights = read_mxfp4_csf_layer(
                self.owner.checkpoint_root,
                self.layer_index,
                num_experts=e,
                hidden_size=h,
                intermediate_size=n * tp,
                tp_rank=rank,
                tp_size=tp,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
            )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config, csf_scales=True)
        backend.process_weights_after_loading(layer, checkpoint_weights=weights)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
