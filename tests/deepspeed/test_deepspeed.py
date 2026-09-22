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

import inspect
import io
import itertools
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from parameterized import parameterized
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, get_scheduler
from transformers.testing_utils import mockenv_context
from transformers.trainer_utils import set_seed
from transformers.utils import is_torch_bf16_available

import accelerate
from accelerate.accelerator import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    TempDirTestCase,
    execute_subprocess_async,
    require_non_cpu,
    require_deepspeed,
    require_multi_device,
    slow,
)
from accelerate.test_utils.training import RegressionDataset
from accelerate.utils.dataclasses import DeepSpeedPlugin
from accelerate.utils.deepspeed import (
    DeepSpeedEngineWrapper,
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
    DummyOptim,
    DummyScheduler,
)
from accelerate.utils.other import patch_environment


set_seed(42)

GPT2_TINY = "sshleifer/tiny-gpt2"

ZERO2 = "zero2"
ZERO3 = "zero3"

FP16 = "fp16"
BF16 = "bf16"

CUSTOM_OPTIMIZER = "custom_optimizer"
CUSTOM_SCHEDULER = "custom_scheduler"
DS_OPTIMIZER = "deepspeed_optimizer"
DS_SCHEDULER = "deepspeed_scheduler"

stages = [ZERO2, ZERO3]
optims = [CUSTOM_OPTIMIZER, DS_OPTIMIZER]
schedulers = [CUSTOM_SCHEDULER, DS_SCHEDULER]
if is_torch_bf16_available():
    dtypes = [FP16, BF16]
else:
    dtypes = [FP16]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Cartesian-product of zero stages with models to test
params = list(itertools.product(stages, dtypes))
optim_scheduler_params = list(itertools.product(optims, schedulers))


@require_deepspeed
@require_non_cpu
class DeepSpeedConfigIntegration(AccelerateTestCase):
    def setUp(self):
        super().setUp()

        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self.test_file_dir_str = str(path.parents[0])

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        with io.open(self.ds_config_file[ZERO2], "r", encoding="utf-8") as f:
            config_zero2 = json.load(f)
        with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
            config_zero3 = json.load(f)
            # The following setting slows things down, so don't enable it by default unless needed by a test.
            # It's in the file as a demo for users since we want everything to work out of the box even if slower.
            config_zero3["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False

        self.ds_config_dict = dict(zero2=config_zero2, zero3=config_zero3)

        self.dist_env = dict(
            ACCELERATE_USE_DEEPSPEED="true",
        )

    def tearDown(self):
        super().tearDown()

        # Remove the dummy config file
        if os.path.exists(self.ds_config_file[ZERO2]):
            os.remove(self.ds_config_file[ZERO2])
        if os.path.exists(self.ds_config_file[ZERO3]):
            os.remove(self.ds_config_file[ZERO3])

    def test_peak_memory_usage(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_peak_memory_usage.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
        ]
        for spec, peak_mem_upper_bound in self.peak_memory_usage_upper_bound.items():
            cmd_stage = cmd.copy()
            if "fp16" in spec:
                cmd_stage.extend(["--mixed_precision=fp16"])

            if "multi_gpu" in spec:
                continue
            else:
                cmd_stage.extend(
                    [
                        "--use_deepspeed",
                        "--gradient_accumulation_steps=1",
                        "--gradient_clipping=1",
                        "--zero3_init_flag=True",
                        "--zero3_save_16bit_model=True",
                    ]
                )
                for i in range(3):
                    if f"stage_{i+1}" in spec:
                        cmd_stage.extend([f"--zero_stage={i+1}"])
                        break
                cmd_stage.extend(
                    [
                        "--offload_optimizer_device=none",
                        "--offload_param_device=none",
                        "--offload_optimizer_nvme_path=none",
                        "--offload_param_nvme_path=none",
                    ]
                )
                if "cpu_offload" in spec:
                    with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
                        ds_config = json.load(f)
                        del ds_config["bf16"]
                        del ds_config["fp16"]
                        del ds_config["optimizer"]["params"]["torch_adam"]
                        del ds_config["optimizer"]["params"]["adam_w_mode"]
                        ds_config_path = os.path.join(self.tmpdir, "ds_config.json")
                        with open(ds_config_path, "w") as out_file:
                            json.dump(ds_config, out_file)

                    cmd_stage.extend([f"--deepspeed_config_file={ds_config_path}"])

            cmd_stage.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    f"--peak_memory_upper_bound={peak_mem_upper_bound}",
                    f"--n_train={self.n_train}",
                    f"--n_val={self.n_val}",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_stage, env=os.environ.copy())

