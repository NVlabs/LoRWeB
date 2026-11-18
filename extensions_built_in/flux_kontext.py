# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

# Copyright 2023-present the HuggingFace Inc. team.
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


# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling LoRWeB or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions in the LICENSE file located at the root directory.


import os
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Any, Tuple, Callable, Literal
import inspect
import random
import math
import yaml
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxKontextPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor, maybe_allow_in_graph
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.peft_utils import _maybe_warn_for_unhandled_keys
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import PipelineImageInput
try:
    from diffusers.hooks.group_offloading import _maybe_remove_and_reapply_group_offloading
    HIGH_TRANSFORMERS_VERSION = True
except ImportError:
    HIGH_TRANSFORMERS_VERSION = False
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    PREFERRED_KONTEXT_RESOLUTIONS,
    calculate_shift,
    retrieve_timesteps,
    XLA_AVAILABLE
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.loaders.lora_pipeline import _LOW_CPU_MEM_USAGE_DEFAULT_LORA, _fetch_state_dict
from transformers import (
    T5TokenizerFast,
    T5EncoderModel,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    Siglip2ImageProcessor,
    Siglip2VisionModel
    )
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.model import LoraModel
# from peft.utils import register_peft_method
from peft.utils.other import get_pattern_key
from optimum.quanto import freeze, QTensor
from einops import rearrange, repeat
from diffusers.utils import (
    logging,
    USE_PEFT_BACKEND,
    get_adapter_name,
    is_peft_version,
    scale_lora_layers,
    is_torch_version,
    unscale_lora_layers,
    convert_unet_state_dict_to_peft
)

from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.models.flux import bypass_flux_guidance, restore_flux_guidance
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import unwrap_model
from toolkit.util.quantize import quantize, get_qtype


if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DO_ANALOGY = False
DO_BOX_ANALOGY = True


ANALOGY_IDS = {
    "A": 3,
    "A_tag": 2,
    "B": 1
}

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}

import enum
from peft.utils import PeftType
from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )

PeftType = enum.Enum('PeftType', {**{e.name: e.value for e in PeftType}, 'ATT_LORA': 'ATT_LORA'}, type=str)


def register_peft_method(
    *, name: str, config_cls, model_cls, prefix: Optional[str] = None, is_mixed_compatible=False
) -> None:
    """
    Function to register a finetuning method like LoRA to be available in PEFT.

    This method takes care of registering the PEFT method's configuration class, the model class, and optionally the
    prefix.

    Args:
        name (str):
            The name of the PEFT method. It must be unique.
        config_cls:
            The configuration class of the PEFT method.
        model_cls:
            The model class of the PEFT method.
        prefix (Optional[str], optional):
            The prefix of the PEFT method. It should be unique. If not provided, the name of the PEFT method is used as
            the prefix.
        is_mixed_compatible (bool, optional):
            Whether the PEFT method is compatible with `PeftMixedModel`. If you're not sure, leave it as False
            (default).

    Example:

        ```py
        # inside of peft/tuners/my_peft_method/__init__.py
        from peft.utils import register_peft_method

        register_peft_method(name="my_peft_method", config_cls=MyConfig, model_cls=MyModel)
        ```
    """
    if name.endswith("_"):
        raise ValueError(f"Please pass the name of the PEFT method without '_' suffix, got {name}.")

    if not name.islower():
        raise ValueError(f"The name of the PEFT method should be in lower case letters, got {name}.")

    if name.upper() not in list(PeftType):
        raise ValueError(f"Unknown PEFT type {name.upper()}, please add an entry to peft.utils.peft_types.PeftType.")

    peft_type = getattr(PeftType, name.upper())

    # model_cls can be None for prompt learning methods, which don't have dedicated model classes
    if prefix is None:
        prefix = name + "_"

    if (
        (peft_type in PEFT_TYPE_TO_CONFIG_MAPPING)
        or (peft_type in PEFT_TYPE_TO_TUNER_MAPPING)
        or (peft_type in PEFT_TYPE_TO_MIXED_MODEL_MAPPING)
    ):
        raise KeyError(f"There is already PEFT method called '{name}', please choose a unique name.")

    if prefix in PEFT_TYPE_TO_PREFIX_MAPPING:
        raise KeyError(f"There is already a prefix called '{prefix}', please choose a unique prefix.")

    model_cls_prefix = getattr(model_cls, "prefix", None)
    if (model_cls_prefix is not None) and (model_cls_prefix != prefix):
        raise ValueError(
            f"Inconsistent prefixes found: '{prefix}' and '{model_cls_prefix}' (they should be the same)."
        )

    PEFT_TYPE_TO_PREFIX_MAPPING[peft_type] = prefix
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls
    if is_mixed_compatible:
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING[peft_type] = model_cls


def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True, model_state_dict=None, adapter_name=None):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]

    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = collections.Counter(rank_dict.values()).most_common()[0][0]

        # for modules with rank different from the most occuring rank, add it to the `rank_pattern`
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".att_lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            # get the alpha occuring the most number of times
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

            # for modules with alpha different from the most occuring alpha, add it to the `alpha_pattern`
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".att_lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()

    # layer names without the Diffusers specific
    target_modules = list({name.split(".att_lora" if "att_lora" in name else ".lora")[0] for name in peft_state_dict.keys()})
    use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)
    # for now we know that the "bias" keys are only associated with `lora_B`.
    lora_bias = any("lora_B" in k and k.endswith(".bias") for k in peft_state_dict)

    # Infer attention-specific parameters from state dict
    n = None
    for key, val in peft_state_dict.items():
        # Cannot figure out rank from lora layers that don't have atleast 2 dimensions.
        if "lora_A" in key:
            n = val.shape[0]
            break

    lora_keys_dim = peft_state_dict[list(filter(lambda x: "lora_keys" in x, peft_state_dict.keys()))[0]].shape[1]
    query_proj = list(filter(lambda x: "lora_to_q" in x and '.weight' in x, peft_state_dict.keys()))
    query_dim = None
    if len(query_proj) > 1:
        query_dim = peft_state_dict[query_proj[0]].shape[1]

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
        "use_dora": use_dora,
        "lora_bias": lora_bias,
        # Add attention specific parameters
        "n": n,
        "lora_keys_dim": lora_keys_dim,
        "query_dim": query_dim,
    }

    return lora_config_kwargs


@dataclass
class AttLoraConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
    )
    bias: Literal["none", "all", "att_lora_only"] = field(
        default="none",
    )
    use_rslora: bool = field(
        default=False,
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
    )
    init_lora_weights: Union[bool, Literal["gaussian"]] = field(
        default=True,
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
    )
    use_dora: bool = field(
        default=False,
    )
    # Enables replicating layers in a model to expand it to a larger model.
    lora_bias: bool = field(
        default=False,
    )

    # New attention-specific parameters
    n: int = field(default=0, metadata={"help": "Number of lora layers"})
    query_dim: int = field(default=0, metadata={"help": "Query dimension for attention"})
    lora_keys_dim: int = field(default=0, metadata={"help": "Lora keys dimension for attention"})

    heads: int = field(default=0, metadata={"help": "Number of attention heads"})
    lora_softmax: bool = field(default=True, metadata={"help": "Whether to use softmax in attention layers"})
    mixing_coeffs_type: str = field(default="mean", metadata={"help": "Type of mixing coefficients"})
    query_mode: str = field(default="aa'bb'", metadata={"help": "Query mode"})
    query_projection_type: str = field(default="linear", metadata={"help": "Query projection type"})
    query_pooling: str = field(default="max", metadata={"help": "Query pooling type"})
    external_query: bool = field(default=False, metadata={"help": "Whether to use external query"})

    # dim_head: int = field(default=0, metadata={"help": "Dimension per attention head"})
    # norm_num_groups: Optional[int] = field(default=None, metadata={"help": "Number of groups for group normalization"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "ATT_LORA"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.lora_bias:
            if self.init_lora_weights not in (True, False):
                raise ValueError(
                    f"The argument lora_bias=True is only supported with init_lora_weights=True or False, got "
                    f"init_lora_weights={self.init_lora_weights} instead."
                )


class AttLoraLinearAdapter(nn.Module, BaseTunerLayer):
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("att_lora_A", "att_lora_B", "att_lora_to_q", "att_lora_keys")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "n", "lora_alpha", "scaling", "lora_dropout",
                         "inner_dim", "query_dim", "lora_keys_dim", "heads",
                         "lora_softmax", "mixing_coeffs_type", "query_mode", "query_projection_type", "query_pooling", "external_query",
                         "is_target_conv_1d_layer", "init_lora_weights",
                         "use_rslora", "lora_bias")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,

        query_dim: int,
        r: int = 0,
        n: int = 0,

        lora_keys_dim: int = 0,
        heads: int = 8,
        lora_softmax: bool = True,
        mixing_coeffs_type: str = "mean",
        query_mode: str = "aa'bb'",
        query_projection_type: str = "linear",
        query_pooling: str = "max",
        external_query: bool = False,

        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        # self.base_layer.requires_grad_(False)  # Freeze the base layer
        self.r = {}
        self.n = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.att_lora_A = nn.ParameterDict({})
        self.att_lora_B = nn.ParameterDict({})
        self.att_lora_keys = nn.ParameterDict({})

        self.inner_dim = {}

        self.query_dim = {}
        self.lora_keys_dim = {}
        self.heads = {}
        self.lora_softmax = {}
        self.mixing_coeffs_type = {}
        self.query_mode = {}
        self.query_projection_type = {}
        self.query_pooling = {}
        self.external_query = {}

        self.att_lora_to_q = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.lora_bias: dict[str, bool] = {}
        # self._caches: dict[str, Any] = {}
        # self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise NotImplementedError("Only Linear layer is supported for AttLoraLinearAdapter")

        # linear lora layer from here on out
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        # Initialize the adapter
        self.update_layer(
            adapter_name,
            r,
            n,
            lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
            # attn
            query_dim=query_dim,
            heads=heads,
            lora_keys_dim=lora_keys_dim,
            lora_softmax=lora_softmax,
            mixing_coeffs_type=mixing_coeffs_type,
            query_mode=query_mode,
            query_projection_type=query_projection_type,
            query_pooling=query_pooling,
            external_query=external_query,
        )

        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(self, adapter_name, r, n, lora_alpha,
                     lora_dropout,
                     init_lora_weights,
                     use_rslora,
                     lora_bias: bool = False,
                     query_dim: int = 0,
                     heads: int = 0,
                     lora_keys_dim: int = 0,
                     lora_softmax: bool = True,
                     mixing_coeffs_type: str = "mean",
                     query_mode: str = "aa'bb'",
                     query_projection_type: str = "linear",
                     query_pooling: str = "max",
                     external_query: bool = False,
                     ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.n[adapter_name] = n
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # attn params
        self.query_dim[adapter_name] = query_dim
        self.heads[adapter_name] = heads
        self.lora_keys_dim[adapter_name] = lora_keys_dim
        self.lora_softmax[adapter_name] = lora_softmax
        self.mixing_coeffs_type[adapter_name] = mixing_coeffs_type
        self.query_mode[adapter_name] = query_mode
        self.query_projection_type[adapter_name] = query_projection_type
        self.query_pooling[adapter_name] = query_pooling
        self.external_query[adapter_name] = external_query

        # self.inner_dim[adapter_name] = dim_head * heads

        # Actual trainable parameters
        self.att_lora_A[adapter_name] = nn.Parameter(torch.randn((n, self.in_features, r)))
        self.att_lora_B[adapter_name] = nn.Parameter(torch.randn((n, r, self.out_features)))
        self.att_lora_keys[adapter_name] = nn.Parameter(torch.randn((n, self.lora_keys_dim[adapter_name])))

        if not self.external_query[adapter_name]:
            query_dim = self.in_features
        # elif self.external_query[adapter_name] and 'cat-' in self.query_mode[adapter_name]:
        #     query_dim *= 3

        if self.query_projection_type[adapter_name] == "none":
            self.att_lora_to_q[adapter_name] = nn.Identity()
        elif self.query_projection_type[adapter_name] == "linear":
            self.att_lora_to_q[adapter_name] = nn.Linear(query_dim, lora_keys_dim, bias=False)
        else:
            raise ValueError(f"Invalid query projection type: {self.query_projection_type[adapter_name]}")

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight,
        # use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.att_lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                # for i in range(self.n[adapter_name]):
                    # nn.init.kaiming_uniform_(self.att_lora_As[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.att_lora_A[adapter_name], a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                # for i in range(self.n[adapter_name]):
                    # nn.init.normal_(self.att_lora_As[adapter_name][i].weight, std=1 / self.r[adapter_name])
                nn.init.normal_(self.att_lora_A[adapter_name], std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            # for i in range(self.n[adapter_name]):
                # nn.init.zeros_(self.att_lora_Bs[adapter_name][i].weight)
            nn.init.zeros_(self.att_lora_B[adapter_name])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        external_query = kwargs.pop("external_query", None)
        wtext = kwargs.pop("wtext", False)

        # If adapters are disabled, just return base layer output
        if self._disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        else:
            # Base layer computation
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.att_lora_A.keys():
                    continue

                lora_A = self.att_lora_A[active_adapter]# [n, in_features, r]
                lora_B = self.att_lora_B[active_adapter]  # [n, r, out_features]
                lora_keys = self.att_lora_keys[active_adapter] # [n, inner_dim]
                to_q = self.att_lora_to_q[active_adapter]
                lora_dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                heads = self.heads[active_adapter]
                lora_softmax = self.lora_softmax[active_adapter]
                query_mode = self.query_mode[active_adapter]
                query_pooling = self.query_pooling[active_adapter]
                mixing_coeffs_type = self.mixing_coeffs_type[active_adapter]
                query_projection_type = self.query_projection_type[active_adapter]

                # Cast input dtype
                x = self._cast_input_dtype(x, lora_A.dtype)

                query = None
                if self.external_query[active_adapter]:
                    query = external_query
                    if query.ndim == 2 or (query.ndim == 3 and query.shape[1] > 1):
                        query = query.reshape(query.shape[0], 1, -1)
                else:
                    query = x
                    if x.ndim == 2:
                        query = query.reshape(query.shape[0], 1, -1)
                    else:
                        wtext = 512 if wtext else 0
                        if query_mode == "aa'bb'":
                            query = x
                        elif query_mode == "caa'bb'":
                            query = x[:, wtext + (x.shape[1] - wtext)//2:, :]
                        elif query_mode == "caa'b":
                            query = x[:, wtext + (x.shape[1] - wtext)//2:, :]
                            query = query[:, : 3*(query.shape[1]//4), :]
                        elif query_mode == "caa'":
                            query = x[:, wtext + (x.shape[1] - wtext)//2:, :]
                            query = query[:, :query.shape[1]//2, :]
                        else:
                            raise ValueError(f"Query mode {query_mode} not supported")

                        if query_pooling == "avg":
                            query = query.mean(dim=1)
                        elif query_pooling == "max":
                            query = query.max(dim=1).values
                        else:
                            raise ValueError(f"Unknown pooling type: {query_pooling}")
                        query = query.reshape(query.shape[0], 1, -1)

                batch_size, nqueries, _ = query.shape

                # `sample` projections.
                head_dim = lora_keys.shape[-1] // heads

                query = to_q(query) # [B x n_queries x inner_dim] e.g., [1, 32, heads*head_dim=1024]
                query = query.view(batch_size, nqueries, heads, head_dim)
                query = query.transpose(1, 2)  # [B, heads, nqueries, head_dim]

                # Prepare keys - lora_keys has shape [N, inner_dim]
                keys = lora_keys.view(-1, heads, head_dim)  # [N, heads, head_dim]
                keys = keys.transpose(0, 1).unsqueeze(0)  # [1, heads, N, head_dim]
                keys = keys.expand(batch_size, -1, -1, -1)  # [B, heads, N, head_dim]

                N = lora_keys.shape[0]
                attn_scale = head_dim**-0.5

                if lora_softmax:
                    values = torch.eye(N, N, dtype=query.dtype, device=query.device).expand(batch_size, heads, -1 , -1)  # [batch_size, heads, N, N]

                    # Linter may incorrectly flag this as not callable
                    # noqa: F821
                    mixing_coeffs = F.scaled_dot_product_attention(
                        query,  # [batch_size, heads, nqueries, head_dim]
                        keys,  # [batch_size, heads, N, head_dim]
                        # [batch_size, heads, nqueries, N]
                        values,  # [batch_size, heads, N, N]
                        dropout_p=0.0, is_causal=False, scale=attn_scale
                    ) # [batch_size, heads, nqueries, N]
                else:
                    mixing_coeffs = query @ keys.transpose(-2, -1) * attn_scale
                    # mixing_coeffs /= N
                    mixing_coeffs = torch.nn.functional.tanh(mixing_coeffs)

                if mixing_coeffs_type == "sum":
                    mixing_coeffs = mixing_coeffs.sum(dim=(1,2))
                    mixing_coeffs = mixing_coeffs.clamp(min=(0 if lora_softmax else -1), max=1)
                elif mixing_coeffs_type == "mean":
                    mixing_coeffs = mixing_coeffs.mean(dim=(1,2))  # [batch_size, N]
                else:
                    raise ValueError(f"Unknown mixing coeffs type: {mixing_coeffs_type}")

                x_dropped = lora_dropout(x)  # [batch_size, sequence_length, in_features]
                lora_output = None
                if x.ndim == 3:
                    lora_output = torch.einsum("b s i, n i r, b n, n r o -> b s o", x_dropped, lora_A, mixing_coeffs, lora_B)
                elif x.ndim == 2:
                    lora_output = torch.einsum("b i, n i r, b n, n r o -> b o", x_dropped, lora_A, mixing_coeffs, lora_B)

                # result = result + lora_B(lora_A(dropout(x))) * scaling
                result = result + lora_output * scaling
            result = result.to(torch_result_dtype)

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # self.get_base_layer().bias.data -= self.att_lora_Bs[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # return output_tensor

    def _cache_store(self, key: str, value: Any) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # value = self._caches.pop(key)
        # return value

    def set_scale(self, adapter, scale):
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # if adapter not in self.scaling:
        #     # Ignore the case where the adapter is not in the layer
        #     return
        # self.scaling[adapter] = scale * self.att_lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # if scale == 1:
        #     return

        # for active_adapter in self.active_adapters:
        #     if active_adapter not in self.att_lora_As.keys():
        #         continue

        #     self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # for active_adapter in self.active_adapters:
        #     if active_adapter not in self.att_lora_As.keys():
        #         continue

        #     if scale is None:
        #         self.scaling[active_adapter] = self.att_lora_alpha[active_adapter] / self.r[active_adapter]
        #     else:
        #         self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented for AttLoraLinearAdapter")
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.att_lora_A.keys():
                continue

            lora_A = self.att_lora_A[active_adapter]
            lora_B = self.att_lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
        return result

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "att_lora." + rep


def default_dispatch_func(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: PeftConfig,
    **kwargs
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding) or \
        isinstance(target_base_layer, torch.nn.Conv2d) or \
            isinstance(target_base_layer, torch.nn.Conv3d) or \
                isinstance(target_base_layer, nn.Conv1d) or \
                    isinstance(target_base_layer, torch.nn.MultiheadAttention):
        raise NotImplementedError("Embedding, Conv2d, Conv3d, Conv1d, MultiheadAttention are not supported yet for AttLora")
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        if isinstance(lora_config, AttLoraConfig):
            new_module = AttLoraLinearAdapter(target, adapter_name, **kwargs)
        else:
            raise ValueError(f"Unknown config type: {type(lora_config)}")

    return new_module


class AttLoraModel(LoraModel):
    prefix: str = "att_lora_"

    def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):

        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "lora_bias": lora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),

            "n": lora_config.n,
            "query_dim": lora_config.query_dim,
            "lora_keys_dim": lora_config.lora_keys_dim,
            "heads": lora_config.heads,
            "lora_softmax": lora_config.lora_softmax,
            "mixing_coeffs_type": lora_config.mixing_coeffs_type,
            "query_mode": lora_config.query_mode,
            "query_projection_type": lora_config.query_projection_type,
            "query_pooling": lora_config.query_pooling,
            "external_query": lora_config.external_query,
        }

        device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None

        new_module = default_dispatch_func(target, adapter_name, lora_config=lora_config, device_map=device_map, **kwargs)

        if adapter_name not in self.active_adapters:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)

        self._replace_module(parent, target_name, new_module, target)

    def _prepare_model(self, peft_config: PeftConfig, model: nn.Module):
        pass

    def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, AttLoraLinearAdapter):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n and 'lora_' not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "att_lora_only":
                for m in model.modules():
                    if isinstance(m, AttLoraLinearAdapter) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _check_merge_allowed(self):
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _unload_and_optionally_merge(self, merge=True, progressbar=False, safe_merge=False, adapter_names=None):
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _check_add_weighted_adapter(self, adapters: list[str], combination_type: str, svd_rank: Union[int, None]) -> tuple[str, int, str]:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def add_weighted_adapter(self, adapters: list[str], combination_type: str, svd_rank: Union[int, None]) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def delete_adapter(self, adapter_name: str) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def merge_and_unload(self, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def unload(self, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def subtract_mutated_initial_weights(self, adapter_name: str) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _svd_generalized_task_arithmetic_weighted_adapter(self, adapters: list[str], weights: list[float], combination_type: str, svd_rank: Optional[int]) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _svd_generalized_task_arithmetic_weighted_adapter_with_density(self, adapters: list[str], weights: list[float], combination_type: str, svd_rank: Optional[int], density: Optional[float]) -> None:
        raise NotImplementedError("This method is not implemented for AttLoraModel")

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        raise NotImplementedError("This method is not implemented for AttLoraModel")


register_peft_method(name="att_lora", config_cls=AttLoraConfig, model_cls=AttLoraModel, is_mixed_compatible=True)


class FluxKontextAnalogyPipeline(FluxKontextPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__(scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer, image_encoder, feature_extractor)

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)

            # Split the image into 3 parts along the x-axis
            # print("@0")
            # print("image.shape", image.shape)
            # print("width", width)
            if DO_ANALOGY:
                # print("@0.1")
                # Split into 3 equal parts
                part_width = image.shape[3] // 3
                image_parts = []
                for i in range(3):
                    start_idx = i * part_width
                    end_idx = (i + 1) * part_width
                    image_part = image[:, :, :, start_idx:end_idx]
                    image_parts.append(image_part)

                # Process each part individually
                processed_parts = []
                for image_part in image_parts:
                    if image_part.shape[1] != self.latent_channels:
                        part_latents = self._encode_vae_image(image=image_part, generator=generator)
                    else:
                        part_latents = image_part
                    processed_parts.append(part_latents)

                # Concatenate the processed parts back along the x-axis
                processed_parts.reverse()
                image_latents = torch.cat(processed_parts, dim=3)
            else:
                # print("@0.2")
                # Original processing for single image
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image

            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )

            # Handle image_ids for concatenated images
            if DO_ANALOGY:# and False: #image is not None and image.shape[3] == width * 3:
                # Process image_ids for each segment individually
                part_width = image_latent_width // 3
                # part_width = image_latent_width

                image_ids_parts = []
                for i in range(3):
                    start_idx = i * part_width
                    end_idx = (i + 1) * part_width
                    part_ids = self._prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, part_width // 2, device, dtype
                    )

                    part_ids = part_ids.reshape(
                        image_latent_height // 2, part_width // 2, part_ids.shape[1]
                    )

                    # Adjust the x coordinates for each part
                    # part_ids[..., 2] = part_ids[..., 2] + start_idx // 2

                    if i == 0:
                        part_ids[..., 0] = ANALOGY_IDS["B"]
                    elif i == 1:
                        part_ids[..., 0] = ANALOGY_IDS["A"]
                    elif i == 2:
                        part_ids[..., 0] = ANALOGY_IDS["A_tag"]

                    image_ids_parts.append(part_ids)

                # Concatenate the image_ids along the x-axis
                image_ids = torch.cat(image_ids_parts, dim=1)

                image_ids = image_ids.reshape(
                    (image_latent_height // 2) * (part_width // 2) * 3, image_ids.shape[2]
                )
                # image_ids = image_ids.reshape(
                #     (image_latent_height // 2) * (part_width // 2), image_ids.shape[2]
                # )
                # Set the first dimension to 1 for all image_ids
                # image_ids[..., 0] = 1
            else:
                # Original processing for single image
                image_ids = self._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
                )
                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                image_ids[..., 0] = 1

        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids


class CustomFluxKontextPipeline(FluxKontextPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        external_query: bool = False,
        query_mode: str = "aa'bb'",
    ):
        super().__init__(scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer, image_encoder, feature_extractor)
        self.external_query = external_query
        self.query_mode = query_mode

    def encode_image(self, image, device, num_images_per_prompt, do_rescale=None, query_mode=None):
        dtype = next(self.image_encoder.parameters()).dtype

        additional_kwargs = {}
        # if type(self.feature_extractor) == SiglipImageProcessor:
            # additional_kwargs["size"] = {"height": 224, "width": 224}
        image = self.feature_extractor(image, return_tensors="pt", do_rescale=do_rescale, **additional_kwargs)

        if type(self.feature_extractor) != Siglip2ImageProcessor:
            image = image.pixel_values
            image = image.to(device=device, dtype=dtype)

        if query_mode is not None:
            # print('a')
            # image_embeds = self.image_encoder.vision_model.(image).last_hidden_state

            output_attentions = self.image_encoder.vision_model.config.output_attentions
            output_hidden_states = self.image_encoder.config.output_hidden_states
            image_embeds = self.image_encoder.vision_model.embeddings(image, interpolate_pos_encoding=False)
            image_embeds = self.image_encoder.vision_model.pre_layrnorm(image_embeds)
            image_embeds = self.image_encoder.vision_model.encoder(
                inputs_embeds=image_embeds, output_hidden_states=('3' in query_mode))
             #    |   [List (embedding + 24 layers) of [B x Cls+Patches^2 X D]]
            if '3' in query_mode:
                image_embeds = torch.cat([image_embeds.hidden_states[-16], image_embeds.hidden_states[-8], image_embeds.hidden_states[-1]], dim=1)
            else:
                image_embeds = image_embeds.last_hidden_state
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            if type(self.image_encoder) == Siglip2VisionModel:
                image_embeds = self.image_encoder(pixel_values=image.pixel_values.to(device=device, dtype=dtype),
                                                  pixel_attention_mask=image.pixel_attention_mask.to(device=device, dtype=dtype),
                                                  spatial_shapes=image.spatial_shapes)
                image_embeds = image_embeds.pooler_output
                # print("image_embeds.shape", image_embeds.shape)
                # exit(0)
            else:
                image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)[:, None]

        return image_embeds

    def get_external_query(self, image, device, query_mode, do_rescale=False):
        with torch.no_grad():
            # if type(self.feature_extractor) == Siglip2ImageProcessor:
            #     # have to transform to pil images
            #     if isinstance(image, torch.Tensor) or (isinstance(image, list) and isinstance(image[0], torch.Tensor)):
            #         print(image.shape)
            #         image = [Image.fromarray(img.cpu().numpy()) for img in image]

            is_ten = isinstance(image, torch.Tensor)
            oneside = image.shape[2]//2 if is_ten else image.size[0]//2

            # if is_ten:
            #     do_rescale = False
            # else:  # list of tensors
            #     do_rescale = True

            if query_mode == "aa'bb'" or query_mode == "caa'bb'":
                # print(image.shape) # [B x 3 x H x W]
                external_query = self.encode_image(image, device, 1, do_rescale=do_rescale).to(device=device)
            elif query_mode == "caa'b" or query_mode == "paa'b" or query_mode == "paa'b3":
                # zero out the right bottom part of the image
                if is_ten:
                    tmp_control_tensor = image.clone()
                    tmp_control_tensor[:, :, oneside:, oneside:] = 0
                else:
                    tmp_control_tensor = image.copy()
                    tmp_control_tensor.paste(0, (oneside, oneside, tmp_control_tensor.size[0], tmp_control_tensor.size[1]))
                external_query = self.encode_image(tmp_control_tensor, device, 1, do_rescale=do_rescale,
                                                   query_mode=query_mode if query_mode.startswith("paa'b") else None).to(device=device)
            elif query_mode == "caa'": # stretches the upper half of the image
                if is_ten:
                    external_query = self.encode_image(image[:, :, :oneside, :], device, 1, do_rescale=do_rescale).to(device=device)
                else:
                    external_query = self.encode_image(image.crop((0, 0, image.size[0], oneside)),
                                                       device, 1, do_rescale=do_rescale).to(device=device)
            elif query_mode == "ca'-ca":
                if is_ten:
                    tmp_a = self.encode_image(image[:, :, :oneside, :oneside],
                                              device, 1, do_rescale=do_rescale).to(device=device)
                    tmp_atag = self.encode_image(image[:, :, :oneside, oneside:],
                                                 device, 1, do_rescale=do_rescale).to(device=device)
                else:
                    tmp_a = self.encode_image(image.crop((0, 0, oneside, oneside)),
                                              device, 1, do_rescale=do_rescale).to(device=device)
                    tmp_atag = self.encode_image(image.crop((oneside, 0, image.size[0], oneside)),
                                                 device, 1, do_rescale=do_rescale).to(device=device)
                external_query = tmp_atag - tmp_a
            elif query_mode in ["ca'-ca+cb", "cat-aa'b", "cat-paa'b", "cat-paa'b3"]:
                if is_ten:
                    tmp_a = self.encode_image(image[:, :, :oneside, :oneside],
                                              device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)
                    tmp_atag = self.encode_image(image[:, :, :oneside, oneside:],
                                                 device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)
                    tmp_b = self.encode_image(image[:, :, oneside:, :oneside],
                                              device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)
                else:
                    tmp_a = self.encode_image(image.crop((0, 0, oneside, oneside)),
                                              device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)
                    tmp_atag = self.encode_image(image.crop((oneside, 0, image.size[0], oneside)),
                                                 device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)
                    tmp_b = self.encode_image(image.crop((0, oneside, oneside, image.size[1])),
                                              device, 1, do_rescale=do_rescale, query_mode=query_mode if query_mode.startswith("cat-paa'b") else None).to(device=device)

                if query_mode == "ca'-ca+cb":
                    external_query = tmp_atag - tmp_a + tmp_b
                elif query_mode == "cat-aa'b" or query_mode.startswith("cat-paa'b"):
                    external_query = torch.cat([tmp_a, tmp_atag, tmp_b], dim=1)
            else:
                raise ValueError(f"Query mode {query_mode} not supported")
        return external_query

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        **kwargs,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        lora_softmax = kwargs.pop("lora_softmax", True)
        mixing_coeffs_type = kwargs.pop("mixing_coeffs_type", "mean")
        query_mode = kwargs.pop("query_mode", "aa'bb'")
        query_projection_type = kwargs.pop("query_projection_type", "linear")
        query_pooling = kwargs.pop("query_pooling", "max")
        external_query = kwargs.pop("external_query", False)
        heads = kwargs.pop("heads", 1)

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        kwargs["return_lora_metadata"] = True
        state_dict, network_alphas, metadata = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, return_alphas=True, **kwargs
        )

        has_lora_keys = any("lora" in key for key in state_dict.keys())

        # Flux Control LoRAs also have norm keys
        has_norm_keys = any(
            norm_key in key for key in state_dict.keys() for norm_key in self._control_lora_supported_norm_keys
        )

        if not (has_lora_keys or has_norm_keys):
            raise ValueError("Invalid LoRA checkpoint.")


        # ADD ATT LORA PREFIX TO MATCH PEFT
        state_dict = {k.replace('.lora_', '.att_lora_') : state_dict[k]
                                       for k in list(state_dict.keys())}

        transformer_lora_state_dict = {
            k: state_dict.get(k)
            for k in list(state_dict.keys())
            if k.startswith(f"{self.transformer_name}.") and "lora" in k
        }

        transformer_norm_state_dict = {
            k: state_dict.pop(k)
            for k in list(state_dict.keys())
            if k.startswith(f"{self.transformer_name}.")
            and any(norm_key in k for norm_key in self._control_lora_supported_norm_keys)
        }

        transformer = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
        has_param_with_expanded_shape = False
        if len(transformer_lora_state_dict) > 0:
            has_param_with_expanded_shape = self._maybe_expand_transformer_param_shape_or_error_(
                transformer, transformer_lora_state_dict, transformer_norm_state_dict
            )

        if has_param_with_expanded_shape:
            logger.info(
                "The LoRA weights contain parameters that have different shapes that expected by the transformer. "
                "As a result, the state_dict of the transformer has been expanded to match the LoRA parameter shapes. "
                "To get a comprehensive list of parameter names that were modified, enable debug logging."
            )
        if len(transformer_lora_state_dict) > 0:
            transformer_lora_state_dict = self._maybe_expand_lora_state_dict(
                transformer=transformer, lora_state_dict=transformer_lora_state_dict
            )
            for k in transformer_lora_state_dict:
                state_dict.update({k: transformer_lora_state_dict[k]})

        self.load_lora_into_transformer(
            state_dict,
            network_alphas=network_alphas,
            transformer=transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            lora_softmax=lora_softmax,
            mixing_coeffs_type=mixing_coeffs_type,
            query_mode=query_mode,
            query_projection_type=query_projection_type,
            query_pooling=query_pooling,
            external_query=external_query,
            heads=heads,
        )

        if len(transformer_norm_state_dict) > 0:
            transformer._transformer_norm_layers = self._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer,
                discard_original_layers=False,
            )

        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder,
            prefix=self.text_encoder_name,
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        network_alphas,
        transformer,
        adapter_name=None,
        metadata=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow. This value has the
                same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
                link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
            transformer (`FluxTransformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            hotswap (`bool`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`].
            metadata (`dict`):
                Optional LoRA adapter metadata. When supplied, the `LoraConfig` arguments of `peft` won't be derived
                from the state dict.
        """
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Load the layers corresponding to transformer.
        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=network_alphas,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            **kwargs,
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        control_image = image

        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )


        if self.external_query:
            external_query = self.get_external_query(control_image, device, self.query_mode, do_rescale=True)
        else:
            external_query = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    external_query=external_query,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        external_query=external_query,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


@maybe_allow_in_graph
class CustomAttention(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        external_query: Optional[torch.Tensor] = None,
        wtext: bool = False,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning("cross_attention_kwargs %s are not expected by %s and will be ignored.",
                           unused_kwargs, self.processor.__class__.__name__)
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            external_query=external_query,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            wtext=wtext,
            **cross_attention_kwargs,
        )


class CustomFluxAttnProcessor2_0:
    # Not a subclass of diffusers' FluxAttnProcessor2_0: since diffusers 0.35 that name is a
    # deprecation shim whose __new__ returns a FluxAttnProcessor, which drops this __call__.
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CustomFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        external_query: Optional[torch.Tensor] = None,
        wtext: bool = False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        # if external_query is not None and 'external_query' in inspect.signature(attn.to_q.forward).parameters:
        query = attn.to_q(hidden_states, external_query=external_query, wtext=wtext)
        # else:
            # query = attn.to_q(hidden_states)

        # if external_query is not None and 'external_query' in inspect.signature(attn.to_k.forward).parameters:
        key = attn.to_k(hidden_states, external_query=external_query, wtext=wtext)
        # else:
            # key = attn.to_k(hidden_states)

        # if external_query is not None and 'external_query' in inspect.signature(attn.to_v.forward).parameters:
        value = attn.to_v(hidden_states, external_query=external_query, wtext=wtext)
        # else:
            # value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            if 'network_mixins' in inspect.getsourcefile(attn.add_q_proj.forward) or 'flux_kontext' in inspect.getsourcefile(attn.add_q_proj.forward):
                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states,
                                                                   external_query=external_query, wtext=wtext)
                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states,
                                                                   external_query=external_query, wtext=wtext)
                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states,
                                                                   external_query=external_query, wtext=wtext)
            else:
                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, external_query=external_query, wtext=wtext)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if 'network_mixins' in inspect.getsourcefile(attn.to_add_out.forward) or 'flux_kontext' in inspect.getsourcefile(attn.to_add_out.forward):
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states,
                                                        external_query=external_query, wtext=wtext)
            else:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class CustomAdaLayerNormZero(AdaLayerNormZero):
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__(embedding_dim, num_embeddings, norm_type, bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        external_query: Optional[torch.Tensor] = None,
        wtext: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        if 'network_mixins' in inspect.getsourcefile(self.linear.forward) or 'flux_kontext' in inspect.getsourcefile(self.linear.forward):
            emb = self.linear(self.silu(emb), external_query=external_query, wtext=wtext)
        else:
            emb = self.linear(self.silu(emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class CustomAdaLayerNormZeroSingle(AdaLayerNormZeroSingle):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__(embedding_dim, norm_type=norm_type, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        external_query: Optional[torch.Tensor] = None,
        wtext: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb), external_query=external_query, wtext=wtext)
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class CustomGELU(torch.nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type == "mps" and is_torch_version("<", "2.0.0"):
            # fp16 gelu not supported on mps before torch 2.0
            return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states, external_query=None, wtext=False):
        if 'network_mixins' in inspect.getsourcefile(self.proj.forward) or 'flux_kontext' in inspect.getsourcefile(self.proj.forward):
            hidden_states = self.proj(hidden_states, external_query=external_query, wtext=wtext)
        else:
            hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class CustomFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = CustomGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = CustomGELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            # act_fn = GEGLU(dim, inner_dim, bias=bias)
            raise NotImplementedError("geglu is not supported yet")
        elif activation_fn == "geglu-approximate":
            # act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
            raise NotImplementedError("geglu-approximate is not supported yet")
        elif activation_fn == "swiglu":
            # act_fn = SwiGLU(dim, inner_dim, bias=bias)
            raise NotImplementedError("swiglu is not supported yet")
        elif activation_fn == "linear-silu":
            # act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
            raise NotImplementedError("linear-silu is not supported yet")

        self.net = torch.nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(torch.nn.Dropout(dropout))
        # project out
        self.net.append(torch.nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(torch.nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor,
                external_query: Optional[torch.Tensor] = None,
                wtext: bool = False,
                *args, **kwargs) -> torch.Tensor:
        for i, module in enumerate(self.net):
            if (not i % 2) and ('flux_kontext' in inspect.getsourcefile(module.forward)
                                or 'network_mixins' in inspect.getsourcefile(module.forward)):
                hidden_states = module(hidden_states, external_query=external_query, wtext=wtext)
            else:
                hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class CustomFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio)

        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = CustomAdaLayerNormZeroSingle(dim)

        if is_torch_npu_available():
            raise NotImplementedError("NPU is not supported yet")
            # processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = CustomFluxAttnProcessor2_0()
        self.attn = CustomAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        external_query = joint_attention_kwargs.get('external_query', None)
        wtext = joint_attention_kwargs.get('wtext', False)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb, external_query=external_query, wtext=wtext)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states, external_query=external_query, wtext=wtext))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states, external_query=external_query, wtext=wtext)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class CustomFluxTransformerBlock(FluxTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm=qk_norm, eps=eps)

        self.norm1 = CustomAdaLayerNormZero(dim)
        self.norm1_context = CustomAdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = CustomFluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = CustomAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.ff = CustomFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.ff_context = CustomFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        external_query = joint_attention_kwargs.get('external_query', None)
        wtext = joint_attention_kwargs.get('wtext', False)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb, external_query=external_query, wtext=wtext
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb, external_query=external_query, wtext=wtext
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states, external_query=external_query, wtext=wtext)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states, external_query=external_query, wtext=wtext)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class CustomFluxTransformer2DModel(FluxTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        ):
        super().__init__(patch_size=patch_size,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         num_layers=num_layers,
                         num_single_layers=num_single_layers,
                         attention_head_dim=attention_head_dim,
                         num_attention_heads=num_attention_heads,
                         joint_attention_dim=joint_attention_dim,
                         pooled_projection_dim=pooled_projection_dim,
                         guidance_embeds=guidance_embeds,
                         axes_dims_rope=axes_dims_rope)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                CustomFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                CustomFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples = None,
        controlnet_single_block_samples = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        external_query: Optional[torch.Tensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        joint_attention_kwargs = {'external_query': external_query, 'wtext': False}

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        joint_attention_kwargs['wtext'] = True

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_lora_adapter(
        self, pretrained_model_name_or_path_or_dict, prefix="transformer",
        hotswap: bool = False, **kwargs
    ):
        from peft import inject_adapter_in_model, set_peft_model_state_dict
        from peft.tuners.tuners_utils import BaseTunerLayer

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        _pipeline = kwargs.pop("_pipeline", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        metadata = kwargs.pop("metadata", None)

        lora_softmax = kwargs.pop("lora_softmax", True)
        mixing_coeffs_type = kwargs.pop("mixing_coeffs_type", "mean")
        query_mode = kwargs.pop("query_mode", "aa'bb'")
        query_projection_type = kwargs.pop("query_projection_type", "linear")
        query_pooling = kwargs.pop("query_pooling", "max")
        external_query = kwargs.pop("external_query", False)
        heads = kwargs.pop("heads", 1)

        allow_pickle = False

        if low_cpu_mem_usage and is_peft_version("<=", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. "
                "Please update it with `pip install -U peft`."
            )

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
            metadata=metadata,
        )
        if network_alphas is not None and prefix is None:
            raise ValueError("`network_alphas` cannot be None when `prefix` is None.")
        if network_alphas and metadata:
            raise ValueError("Both `network_alphas` and `metadata` cannot be specified.")

        if prefix is not None:
            state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            if metadata is not None:
                metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

        if len(state_dict) > 0:
            if adapter_name in getattr(self, "peft_config", {}) and not hotswap:
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the model - please select a new adapter name."
                )
            elif adapter_name not in getattr(self, "peft_config", {}) and hotswap:
                raise ValueError(
                    f"Trying to hotswap LoRA adapter '{adapter_name}' but there is no existing adapter by that name. "
                    "Please choose an existing adapter name or set `hotswap=False` to prevent hotswapping."
                )

            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            second_key = (list(state_dict.keys()))[1]
            if "lora_A" not in first_key and "lora_A" not in second_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            rank = {}
            for key, val in state_dict.items():
                # Cannot figure out rank from lora layers that don't have at least 2 dimensions.
                # Bias layers in LoRA only have a single dimension
                if "lora_B" in key and val.ndim > 1:
                    # Check out https://github.com/huggingface/peft/pull/2419 for the `^` symbol.
                    # We may run into some ambiguous configuration values when a model has module
                    # names, sharing a common prefix (`proj_out.weight` and `blocks.transformer.proj_out.weight`,
                    # for example) and they have different LoRA ranks.
                    rank[f"^{key}"] = val.shape[1]

            if network_alphas is not None and len(network_alphas) >= 1:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
                network_alphas = {
                    k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys
                }

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            if metadata is not None:
                lora_config_kwargs = metadata
            else:
                lora_config_kwargs = get_peft_kwargs(
                    rank,
                    network_alpha_dict=network_alphas,
                    peft_state_dict=state_dict,
                    is_unet=True,
                    model_state_dict=self.state_dict(),
                    adapter_name=adapter_name,
                )

            lora_config = AttLoraConfig(**lora_config_kwargs,
                                        heads=heads,
                                        lora_softmax=lora_softmax,
                                        mixing_coeffs_type=mixing_coeffs_type,
                                        query_mode=query_mode,
                                        query_projection_type=query_projection_type,
                                        query_pooling=query_pooling,
                                        external_query=external_query,
                                        )
            # <Unsafe code
            # We can be sure that the following works as it just sets attention processors,
            # lora layers and puts all in the same dtype
            # Now we remove any existing hooks to `_pipeline`.

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error.
            if HIGH_TRANSFORMERS_VERSION:
                is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload = self._optionally_disable_offloading(
                    _pipeline
                )
            else:
                is_group_offload = False
                is_model_cpu_offload, is_sequential_cpu_offload = self._optionally_disable_offloading(
                    _pipeline
                )

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            # To handle scenarios where we cannot successfully set state dict. If it's unsuccessful,
            # we should also delete the `peft_config` associated to the `adapter_name`.
            try:
                inject_adapter_in_model(
                    lora_config, self, adapter_name=adapter_name, **peft_kwargs
                )
                incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name, **peft_kwargs)

                # Set peft config loaded flag to True if module has been successfully injected and incompatible keys retrieved
                if not self._hf_peft_config_loaded:
                    self._hf_peft_config_loaded = True
            except Exception as e:
                # In case `inject_adapter_in_model()` was unsuccessful even before injecting the `peft_config`.
                if hasattr(self, "peft_config"):
                    for module in self.modules():
                        if isinstance(module, BaseTunerLayer):
                            active_adapters = module.active_adapters
                            for active_adapter in active_adapters:
                                if adapter_name in active_adapter:
                                    module.delete_adapter(adapter_name)

                    self.peft_config.pop(adapter_name)
                logger.error(f"Loading {adapter_name} was unsuccessful with the following error: \n{e}")
                raise

            _maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            elif is_group_offload:
                for component in _pipeline.components.values():
                    if isinstance(component, torch.nn.Module):
                        _maybe_remove_and_reapply_group_offloading(component)
            # Unsafe code />

        if prefix is not None and not state_dict:
            model_class_name = self.__class__.__name__
            logger.warning(
                f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
                "This is safe to ignore if LoRA state dict didn't originally have any "
                f"{model_class_name} related params. You can also try specifying `prefix=None` "
                "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
                "https://github.com/huggingface/diffusers/issues/new"
            )


class FluxKontextModel(BaseModel):
    arch = "flux_kontext"

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['FluxTransformer2DModel', 'CustomFluxTransformer2DModel']
        self.external_query = kwargs.get("external_query", False)
        self.query_mode = kwargs.get("query_mode", "aa'bb'")
        self.external_query_model = kwargs.get("external_query_model", None)

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, do_rescale=None, query_mode=None):
        dtype = next(self.image_encoder.parameters()).dtype

        image = self.feature_extractor(image, return_tensors="pt", do_rescale=do_rescale).pixel_values

        image = image.to(device=device, dtype=dtype)
        if query_mode is not None:
            print('b')
            # image_embeds = self.image_encoder.vision_model.(image).last_hidden_state

            output_attentions = self.image_encoder.vision_model.config.output_attentions
            output_hidden_states = self.image_encoder.config.output_hidden_states
            image_embeds = self.image_encoder.vision_model.embeddings(image, interpolate_pos_encoding=False)
            image_embeds = self.image_encoder.vision_model.pre_layrnorm(image_embeds)
            image_embeds = self.image_encoder.vision_model.encoder(
                inputs_embeds=image_embeds, output_hidden_states=('3' in query_mode))
             #    |   [List (embedding + 24 layers) of [B x Cls+Patches^2 X D]]
            if '3' in query_mode:
                image_embeds = torch.cat([image_embeds.hidden_states[-16], image_embeds.hidden_states[-8], image_embeds.hidden_states[-1]], dim=1)
            else:
                image_embeds = image_embeds.last_hidden_state
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)[:, None]

        return image_embeds

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux Kontext model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        # this is the original path put in the model directory
        # it is here because for finetuning we only save the transformer usually
        # so we need this for the VAE, te, etc
        base_model_path = self.model_config.extras_name_or_path

        transformer_path = model_path
        transformer_subfolder = 'transformer'
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, 'text_encoder')
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading transformer")
        # used_flux_transformer_model = FluxTransformer2DModel
        # if self.external_query:
        used_flux_transformer_model = CustomFluxTransformer2DModel
        transformer = used_flux_transformer_model.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=dtype
        )
        # transformer.external_query = self.external_query
        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        self.print_and_status_update("Loading T5")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            base_model_path, subfolder="tokenizer_2", torch_dtype=dtype
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        text_encoder_2.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing T5")
            quantize(text_encoder_2, weights=get_qtype(
                self.model_config.qtype))
            freeze(text_encoder_2)
            flush()

        self.print_and_status_update("Loading CLIP")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder.to(self.device_torch, dtype=dtype)

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)

        image_encoder = None
        feature_extractor = None
        if self.external_query:
            # load the external query model
            if 'clip' in self.external_query_model:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.external_query_model, ignore_mismatched_sizes=True).to(self.device, dtype=dtype)
                feature_extractor = CLIPImageProcessor.from_pretrained(
                    self.external_query_model,
                    size=image_encoder.config.image_size,
                    crop_size=image_encoder.config.image_size
                )

            elif 'siglip' in self.external_query_model:
                image_encoder = Siglip2VisionModel.from_pretrained(self.external_query_model,
                                                                  ignore_mismatched_sizes=True
                                                                  ).to(self.device, dtype=dtype)
                feature_extractor = Siglip2ImageProcessor.from_pretrained(self.external_query_model,
                                                                        #  size=image_encoder.config.image_size,
                                                                        #  crop_size=image_encoder.config.image_size
                                                                        )
            elif 'vit' in self.external_query_model:
                from transformers import ViTFeatureExtractor, ViTForImageClassification
                image_encoder = ViTForImageClassification.from_pretrained(self.external_query_model).to(self.device, dtype=dtype)
                feature_extractor = ViTFeatureExtractor.from_pretrained(self.external_query_model,
                                                                        size=image_encoder.config.image_size,
                                                                        crop_size=image_encoder.config.image_size)
            else:
                raise ValueError(f"unknown image encoder arch: {self.external_query_model}")
        flush()

        self.noise_scheduler = FluxKontextModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        # chosen_pipeline = FluxKontextPipeline
        chosen_pipeline = CustomFluxKontextPipeline
        # if self.external_query:
        if DO_ANALOGY:
            chosen_pipeline = FluxKontextAnalogyPipeline

        pipe: FluxKontextPipeline = chosen_pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            external_query=self.external_query,
            query_mode=self.query_mode,
        )

        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
        tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        text_encoder[1].to(self.device_torch)
        text_encoder[1].requires_grad_(False)
        text_encoder[1].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = FluxKontextModel.get_train_scheduler()

        chosen_pipeline = FluxKontextPipeline
        image_encoder = None
        feature_extractor = None
        chosen_pipeline = CustomFluxKontextPipeline
        if self.external_query:
            image_encoder = unwrap_model(self.image_encoder)
            feature_extractor = unwrap_model(self.feature_extractor)
        elif DO_ANALOGY:
            chosen_pipeline = FluxKontextAnalogyPipeline

        pipeline: FluxKontextPipeline = chosen_pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            external_query=self.external_query,
            query_mode=self.query_mode,
        )
        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: FluxKontextPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if gen_config.ctrl_img is None:
            raise ValueError(
                "Control image is required for Flux Kontext model generation."
            )
        else:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if gen_config.is_box_analogy:
                # print("is box analogy")
                # exit(0)
                # Resize control image to target size
                # target_size = (gen_config.width, gen_config.height)
                # control_img = control_img.resize(target_size, Image.BICUBIC)

                # Split control image into 3 parts (left, middle, right)
                w, h = (control_img.size[0], control_img.size[1])
                part_w = w // 3
                control_parts = [
                    control_img.crop((i * part_w, 0, (i + 1) * part_w, h))
                    for i in range(3)
                ]

                # Create 2x2 grid with target part masked out (black)
                # Top row: Control part 1, Control part 2
                top_row = Image.new('RGB', (2 * w, h))
                top_row.paste(control_parts[0].resize((w, h), Image.BICUBIC), (0, 0))
                top_row.paste(control_parts[1].resize((w, h), Image.BICUBIC), (w, 0))

                # Bottom row: Control part 3, Black (masked target)
                bottom_row = Image.new('RGB', (2 * w, h))
                bottom_row.paste(control_parts[2].resize((w, h), Image.BICUBIC), (0, 0))
                # black_rect = Image.new('RGB', (w, h), (0, 0, 0))
                black_rect = control_parts[2].resize((w, h), Image.BICUBIC)
                bottom_row.paste(black_rect, (w, 0))

                # Combine rows
                control_img = Image.new('RGB', (2 * w, 2 * h))
                control_img.paste(top_row, (0, 0))
                control_img.paste(bottom_row, (0, h))

                control_img = control_img.resize((gen_config.width, gen_config.height), Image.BICUBIC)
            if gen_config.is_analogy:
                # print("is analogy")
                # exit(0)
                if control_img.size != (gen_config.width * 3, gen_config.height):
                    control_img = control_img.resize(
                        (gen_config.width * 3, gen_config.height), Image.BILINEAR
                    )
            elif control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )

        gen_config.width = int(gen_config.width  // 16 * 16)
        gen_config.height = int(gen_config.height // 16 * 16)
        img = pipeline(
            image=control_img,
            prompt_embeds=conditional_embeds.text_embeds,
            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            max_area=gen_config.height * gen_config.width,
            _auto_resize=False,
            **extra
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        bypass_guidance_embedding: bool,
        **kwargs
    ):
        with torch.no_grad():
            bs, c, h, w = latent_model_input.shape
            factor = 1
            # print("latent_model_input.shape", latent_model_input.shape)
            # if we have a control on the channel dimension, put it on the batch for packing
            has_control = False
            has_analogy = False
            if latent_model_input.shape[1] == 32:
                # chunk it and stack it on batch dimension
                # dont update batch size for img_its
                lat, control = torch.chunk(latent_model_input, 2, dim=1)
                latent_model_input = torch.cat([lat, control], dim=0)
                has_control = True
            if latent_model_input.shape[1] == 64:
                # chunk it and stack it on batch dimension
                # dont update batch size for img_its
                lat, A, A_tag, B = torch.chunk(latent_model_input, 4, dim=1)

                # Resize A and A_tag by factor
                # factor = 1
                A = F.interpolate(A, size=(h // factor, w // factor), mode='bilinear')
                A_tag = F.interpolate(A_tag, size=(h // factor, w // factor), mode='bilinear')

                latent_model_input = torch.cat([lat, A_tag, A, B], dim=0)
                has_control = True
                has_analogy = True

            latent_model_input_packed = rearrange(
                latent_model_input,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=2,
                pw=2
            )

            # print("latent_model_input.shape", latent_model_input.shape)
            # print("latent_model_input_packed.shape", latent_model_input_packed.shape)

            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            # img_ids = repeat(img_ids, "h w c -> b (h w) c",
            #                  b=bs).to(self.device_torch)

            # handle control image ids
            all_ctrl_ids = []

            if DO_ANALOGY: # and False:
                if has_control:
                    ctrl_ids = img_ids.clone()
                    ctrl_ids[..., 0] = ANALOGY_IDS["B"]
                    # img_ids = torch.cat([img_ids, ctrl_ids], dim=1)
                    all_ctrl_ids.append(ctrl_ids)
                if has_analogy:
                    analogy_ids_A = torch.zeros(h // (2 * factor), w // (2 * factor), 3)
                    analogy_ids_A[..., 1] = analogy_ids_A[..., 1] + torch.arange(h // (2 * factor))[:, None]
                    analogy_ids_A[..., 2] = analogy_ids_A[..., 2] + torch.arange(w // (2 * factor))[None, :]
                    analogy_ids_A[..., 0] = ANALOGY_IDS["A"]

                    analogy_ids_A_tag = torch.zeros(h // (2 * factor), w // (2 * factor), 3)
                    analogy_ids_A_tag[..., 1] = analogy_ids_A_tag[..., 1] + torch.arange(h // (2 * factor))[:, None]
                    analogy_ids_A_tag[..., 2] = analogy_ids_A_tag[..., 2] + torch.arange(w // (2 * factor))[None, :]
                    analogy_ids_A_tag[..., 0] = ANALOGY_IDS["A_tag"]

                    # analogy_ids_A = repeat(analogy_ids_A, "h w c -> b (h w) c",
                                        #  b=bs).to(self.device_torch)
                    # analogy_ids_A_tag = repeat(analogy_ids_A_tag, "h w c -> b (h w) c",
                                        #  b=bs).to(self.device_torch)

                    all_ctrl_ids.append(analogy_ids_A)
                    all_ctrl_ids.append(analogy_ids_A_tag)
                    # img_ids = torch.cat([img_ids, analogy_ids_A, analogy_ids_A_tag], dim=1)
                all_ctrl_ids = torch.cat(all_ctrl_ids, dim=1)
                all_ctrl_ids = repeat(all_ctrl_ids, "h w c -> b (h w) c",
                                        b=bs).to(self.device_torch)
            elif has_control:
                all_ctrl_ids = img_ids.clone()
                all_ctrl_ids[..., 0] = 1
                all_ctrl_ids = repeat(all_ctrl_ids, "h w c -> b (h w) c",
                                        b=bs).to(self.device_torch)
            else:
                all_ctrl_ids = torch.zeros(h // (2 * factor), (w // (2 * factor)) * 3, 3)
                all_ctrl_ids[..., 1] = all_ctrl_ids[..., 1] + torch.arange(h // (2 * factor))[:, None]
                all_ctrl_ids[..., 2] = all_ctrl_ids[..., 2] + torch.arange((w // (2 * factor)) * 3)[None, :]
                all_ctrl_ids[..., 0] = 1
                all_ctrl_ids = repeat(all_ctrl_ids, "h w c -> b (h w) c",
                                        b=bs).to(self.device_torch)

            # print("all_ctrl_ids.shape", all_ctrl_ids.shape)
            # print("img_ids.shape", img_ids.shape)
            img_ids = repeat(img_ids, "h w c -> b (h w) c",
                             b=bs).to(self.device_torch)
            img_ids = torch.cat([img_ids, all_ctrl_ids], dim=1)

            txt_ids = torch.zeros(
                bs, text_embeddings.text_embeds.shape[1], 3).to(self.device_torch)

            # # handle guidance
            if self.unet_unwrapped.config.guidance_embeds:
                if isinstance(guidance_embedding_scale, list):
                    guidance = torch.tensor(
                        guidance_embedding_scale, device=self.device_torch)
                else:
                    guidance = torch.tensor(
                        [guidance_embedding_scale], device=self.device_torch)
                    guidance = guidance.expand(latent_model_input.shape[0])
            else:
                guidance = None

        if bypass_guidance_embedding:
            bypass_flux_guidance(self.unet)

        cast_dtype = self.unet.dtype
        # changes from orig implementation
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        latent_size = latent_model_input_packed.shape[1]
        # move the kontext channels. We have them on batch dimension to here, but need to put them on the latent dimension

        if has_analogy:
            latent, B, A, A_tag = torch.chunk(latent_model_input_packed, 4, dim=0)
            latent_model_input_packed = torch.cat(
                [latent, B, A, A_tag], dim=1
            )
            latent_size = latent.shape[1]
        elif has_control:
            latent, control = torch.chunk(latent_model_input_packed, 2, dim=0)
            latent_model_input_packed = torch.cat(
                [latent, control], dim=1
            )
            latent_size = latent.shape[1]

        if DO_ANALOGY and guidance.shape[0] > 2:
            guidance = guidance[:guidance.shape[0] // 4]
        elif DO_BOX_ANALOGY and guidance.shape[0] > 2:
            guidance = guidance[:guidance.shape[0] // 2]

        # print("latent_model_input_packed.shape", latent_model_input_packed.shape)
        # print("timestep.shape", timestep.shape)
        # print("text_embeddings.text_embeds.shape", text_embeddings.text_embeds.shape)
        # print("text_embeddings.pooled_embeds.shape", text_embeddings.pooled_embeds.shape)
        # print("txt_ids.shape", txt_ids.shape)
        # print("img_ids.shape", img_ids.shape)
        # print("guidance.shape", guidance.shape)
        # exit(0)
        noise_pred = self.unet(
            hidden_states=latent_model_input_packed.to(
                self.device_torch, cast_dtype),
            timestep=timestep / 1000,
            encoder_hidden_states=text_embeddings.text_embeds.to(
                self.device_torch, cast_dtype),
            pooled_projections=text_embeddings.pooled_embeds.to(
                self.device_torch, cast_dtype),
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=guidance,
            return_dict=False,
            **kwargs,
        )[0]

        # remove kontext image conditioning
        noise_pred = noise_pred[:, :latent_size]

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()

        noise_pred = rearrange(
            noise_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_model_input.shape[2] // 2,
            w=latent_model_input.shape[3] // 2,
            ph=2,
            pw=2,
            c=self.vae.config.latent_channels
        )

        if bypass_guidance_embedding:
            restore_flux_guidance(self.unet)

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, pooled_prompt_embeds = train_tools.encode_prompts_flux(
            self.tokenizer,
            self.text_encoder,
            prompt,
            max_length=512,
        )
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.pooled_embeds = pooled_prompt_embeds
        return pe

    def get_model_has_grad(self):
        # return from a weight if it has grad
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        # return from a weight if it has grad
        return self.text_encoder[1].encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: FluxTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        return (noise - batch.latents).detach()

    def condition_noisy_latents(self, latents: torch.Tensor, batch:'DataLoaderBatchDTO'):
        # print("condition_noisy_latents start latents.shape", latents.shape)
        # exit(0)
        with torch.no_grad():
            control_tensors = batch.control_tensor # [bs, ch, h, w]

            # print("control_tensors.shape", control_tensors.shape)
            # exit(0)

            if control_tensors is not None:
                self.vae.to(self.device_torch)
                # we are not packed here, so we just need to pass them so we can pack them later
                # print("control_tensors.min", control_tensors.min())
                # print("control_tensors.max", control_tensors.max())
                # print("control_tensors.mean", control_tensors.mean())
                # exit(0)

                control_tensors = control_tensors.to(self.vae_device_torch, dtype=self.torch_dtype)
                # print("control_tensors.min", control_tensors.min())
                # print("control_tensors.max", control_tensors.max())
                # print("control_tensors.mean", control_tensors.mean())
                # exit(0)


                if control_tensors.shape[1] == 9:
                    # split it into 3 channels
                    control_tensors = torch.chunk(control_tensors, 3, dim=1)
                else:
                    # control_tensors = control_tensors * 2 - 1
                    control_tensors = [control_tensors]

                for control_tensor in control_tensors:
                # if it is not the size of batch.tensor, (bs,ch,h,w) then we need to resize it
                    if batch.tensor is not None:
                        target_h, target_w = batch.tensor.shape[2], batch.tensor.shape[3]
                    else:
                        # When caching latents, batch.tensor is None. We get the size from the file_items instead.
                        target_h = batch.file_items[0].crop_height
                        target_w = batch.file_items[0].crop_width

                    if control_tensor.shape[2] != target_h or control_tensor.shape[3] != target_w:
                        control_tensor = F.interpolate(control_tensor, size=(target_h, target_w), mode='bilinear')

                    control_latent = self.encode_images(control_tensor).to(latents.device, latents.dtype)
                    latents = torch.cat((latents, control_latent), dim=1)

        # print("condition_noisy_latents end latents.shape", latents.shape)
        return latents.detach()

    def get_base_model_version(self):
        return "flux.1_kontext"
