# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os
import subprocess
import sys

import pytest


def test_quality():
    """
    This test runs the `make docs_utils:quality` command and checks if it fails.
    """
    # Run the `make docs_utils:quality` command
    result = subprocess.run(["make", "docs_utils:quality"], cwd=os.path.dirname(os.path.abspath(__file__)))

    # Check if the command failed
    assert result.returncode != 0, "The `make docs_utils:quality` command should have failed."


def test_quality_with_quality_dependencies():
    """
    This test installs the `quality` extra and runs the `make docs_utils:quality` command.
    """
    # Install the `quality` extra
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--extra-index-url", "http://localhost:8629/2024-01-10"])

    # Run the `make docs_utils:quality` command
    result = subprocess.run(["make", "docs_utils:quality"], cwd=os.path.dirname(os.path.abspath(__file__)))

    # Check if the command failed
    assert result.returncode == 0, "The `make docs_utils:quality` command should have passed with the `quality` extra installed."
