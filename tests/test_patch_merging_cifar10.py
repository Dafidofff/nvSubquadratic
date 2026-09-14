# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


"""Download-free coverage of the two recovered CIFAR-10 recipe adapters."""

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset


@pytest.mark.parametrize("source", ["torchvision", "huggingface"])
def test_cifar_validation_batch_contract(monkeypatch, source):
    if source == "torchvision":
        from experiments.datamodules import cifar10 as module

        samples = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4))
        monkeypatch.setattr(module.datasets, "CIFAR10", lambda *args, **kwargs: samples)
        dm = module.CIFAR10DataModule(batch_size=2, num_workers=0, pin_memory=False)
        dm.setup("validate")
        batch = next(iter(dm.val_dataloader()))
    else:
        from experiments.datamodules import cifar10_hf as module

        samples = [{"img": Image.new("RGB", (32, 32)), "label": i} for i in range(4)]
        monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: samples)
        dm = module.CIFAR10DataModule(
            data_dir="unused", batch_size=2, num_workers=0, pin_memory=False, final_image_size=32
        )
        dm.setup("validate")
        batch = dm.on_before_batch_transfer(next(iter(dm.val_dataloader())), 0)
    assert batch["input"].shape[1:] == (32, 32, 3)
    assert batch["label"].dtype == torch.long
    assert batch["condition"] is None
    assert dm.output_channels == 10
