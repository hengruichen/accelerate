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

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .hooks import (
    AlignDevicesHook,
    CpuOffload,
    UserCpuOffloadHook,
    add_hook_to_module,
    attach_align_device_hook,
    attach_align_device_hook_on_blocks,
)
from .utils import (
    OffloadedWeightsLoader,
    check_device_map,
    extract_submodules_state_dict,
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    is_npu_available,
    is_torch_version,
    load_checkpoint_in_model,
    offload_state_dict,
    parse_flag_from_env,
    retie_parameters,
)


logger = logging.getLogger(__name__)


@contextmanager
def init_empty_weights(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param, otherwise dispatch is not called.
    </Tip>
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

    # TODO(shingjan): remove the torch version check once older versions are deprecated
    if is_torch_version(">=", "2.0") and include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name]
# ... [truncated] ...
 or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if isinstance(device_map, str):
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=dtype
        )
    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True
    load_checkpoint_in_model(
        model,
        checkpoint,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=dtype,
        offload_state_dict=offload_state_dict,
        offload_buffers=offload_buffers,
    )
    if device_map is None:
        return model
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        skip_keys=skip_keys,
        preload_module_classes=preload_module_classes,
        force_hooks=force_hooks,
    )

