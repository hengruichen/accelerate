# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import contextlib
import gc
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..state import AcceleratorState
from .constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import AutocastKwargs, CustomDtype, DistributedType
from .imports import is_mps_available, is_npu_available, is_xpu_available, is_peft_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index
from .tqdm import is_tqdm_available, tqdm


if is_npu_available(check_device=False):
    import torch_npu  # noqa: F401

from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file


WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

logger = logging.getLogger(__name__)


def is_peft_model(model):
    from .other import extract_model_from_parallel

    if is_peft_available():
        from peft import PeftModel

    return is_peft_available() and isinstance(extract_model_from_parallel(model), PeftModel)


def check_device_same(first_device, second_device):
    """
    Utility method to check if two `torch` devices are similar. When dealing with CUDA devices, torch throws `False`
    for `torch.device("cuda") == torch.device("cuda:0")` whereas they should be the same

    Args:
        first_device (`torch.device`):
            First device to check
        second_device (`torch.device`):
            Second device to check
    """
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        # In case the first_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        # In case the second_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        second_device = torch.device("cuda", index=0)

    return first_device == second_device


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    mem_size = 0
    err_msg = (
        f"`size` {size} is not in a valid format. Use an integer for bytes, or a string with an unit (like '5.0GB')."
    )
    try:
        if isinstance(size, int):
            mem_size = size
        elif size.upper().endswith("GIB"):
            mem_size = int(float(size[:-3]) * (2**30))
        elif size.upper().endswith("MIB"):
            mem_size = int(float(size[:-3]) * (2**20))
        elif size.upper().endswith("KIB"):
            mem_size = int(float(size[:-3]) * (2**10))
        elif size.upper().endswith("GB"):
            int_size = int(float(size[:-2]) * (10**9))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("MB"):
            int_size = int(float(size[:-2]) * (10**6))
            mem_size = int_size // 8 if size.endswith("b") else in
# ... [truncated] ...
  for key in keep_in_fp32_modules:
                            if ((key in param_name) and (key + "." in param_name)) or key == param_name:
                                proceed = True
                                break
                        if proceed:
                            new_dtype = torch.float32

                if "weight" in param_name and param_name.replace("weight", "SCB") in checkpoint.keys():
                    if param.dtype == torch.int8:
                        fp16_statistics = checkpoint[param_name.replace("weight", "SCB")]
                else:
                    fp16_statistics = None

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        if new_dtype is None:
                            new_dtype = param.dtype
                        if offload_8bit_bnb:
                            quantize_and_offload_8bit(
                                model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics
                            )
                            continue
                        else:
                            set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    if new_dtype is None:
                        new_dtype = param.dtype
                    if offload_8bit_bnb:
                        quantize_and_offload_8bit(
                            model, param, param_name, new_dtype, state_dict_folder, state_dict_index, fp16_statistics
                        )
                    else:
                        set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(
                        model,
                        param_name,
                        param_device,
                        value=param,
                        dtype=new_dtype,
                        fp16_statistics=fp16_statistics,
                    )

        # Force Python to clean up.
        del checkpoint
        gc.collect()

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)


def get_mixed_precision_context_manager(native_amp: bool = False, autocast_kwargs: AutocastKwargs = None):
    """
    Return a context manager for autocasting mixed precision

    Args:
        native_amp (`bool`, *optional*, defaults to False):
            Whether mixed precision is actually enabled.
        cache_enabled (`bool`, *optional*, defaults to True):
            Whether the weight cache inside autocast should be enabled.
    """
    state = AcceleratorState()
    if autocast_kwargs is None:
        autocast_kwargs = {}
    else:
        autocast_kwargs = autocast_kwargs.to_kwargs()
    if native_amp:
        if state.mixed_precision == "fp16":
            return torch.autocast(device_type=state.device.type, dtype=torch.float16, **autocast_kwargs)
        elif state.mixed_precision == "bf16" and state.distributed_type in [
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_NPU,
            DistributedType.MULTI_XPU,
            DistributedType.FSDP,
        ]:
            return torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, **autocast_kwargs)
        else:
            return torch.autocast(device_type=state.device.type, **autocast_kwargs)
    else:
        return contextlib.nullcontext()

