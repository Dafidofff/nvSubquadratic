# 3D motion spatial recall

This moving-image copy task extends spatial recall from a static image to a
short video block. A source image is resized, translated along an ink-aware
monotone path and optionally rotated in quarter turns. Depth represents time.
The model must reproduce the entire block at the back-bottom-right readout
corner of a larger cubic canvas.

`SpatialRecall3DMotionDataset` returns a canvas `[C, S, S, S]` and target
`[C, b, b, b]`, where `S = canvas_size` and `b = block_size`. The target is
placed at the front-top-left corner (`placement="fixed"`) or a random origin
that does not overlap the readout (`placement="random"`). The readout is filled
with `readout_value` (zero by default). The target is the video itself, not the
source dataset's class label.

## Download-free example

Run from the repository root in an environment with the project's core Python
dependencies installed:

```python
import torch
from torch.utils.data import TensorDataset

from experiments.datamodules.spatial_recall_dataset import SpatialRecall3DMotionDataset

image = torch.zeros(1, 1, 8, 8)
image[:, :, 2:6, 3:5] = 1
source = TensorDataset(image, torch.zeros(1, dtype=torch.long))
dataset = SpatialRecall3DMotionDataset(
    base_dataset=source,
    digit_size=4,
    block_size=6,
    canvas_size=32,
    generator=torch.Generator().manual_seed(42),
    placement="fixed",
    max_step=2.0,
    spin=True,
)
canvas, target = dataset[0]
assert canvas.shape == (1, 32, 32, 32)
assert target.shape == (1, 6, 6, 6)
assert torch.equal(canvas[:, :6, :6, :6], target)
assert torch.count_nonzero(canvas[:, -6:, -6:, -6:]) == 0
```

## Lightning integration

`SpatialRecall3DMotionDataModule` wraps a grayscale base datamodule such as
MNIST or EMNIST using its existing lazy configuration:

```python
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import (
    SpatialRecall3DMotionDataModule,
)
from nvsubquadratic.lazy_config import LazyConfig

module = SpatialRecall3DMotionDataModule(
    base_datamodule_cfg=LazyConfig(EMNISTDataModule)(
        data_dir=".data/emnist",
        batch_size=16,
        data_type="image",
        num_workers=4,
        pin_memory=True,
        permuted=False,
        seed=42,
        normalize_input=True,
        split="byclass",
    ),
    digit_size=4,
    block_size=6,
    canvas_size=32,
    data_type="volume",
)
# Explicitly call module.prepare_data() to download EMNIST if needed,
# then module.setup("fit") to construct the train/validation datasets.
```

The batch-transfer hook returns `{"input": x, "label": y, "condition": None}`.
With `data_type="volume"`, shapes are `[B, S, S, S, 1]` and `[B, b, b, b, 1]`;
with `"sequence"`, they are `[B, S**3, 1]` and `[B, b**3, 1]`. A regression
network must read out the final `b` positions along each spatial axis (for
`ResidualNetwork`, use `target_size=[b, b, b]`) and compare that output to the
label. No model architecture change is required by the dataset.

## Sampling semantics

- Require `0 < digit_size <= block_size` and `canvas_size >= 2 * block_size`,
  including fixed placement, so source and readout cannot overlap.
- `max_step` is a finite, nonnegative **per-axis continuous** step limit.
  Voxel rounding permits integer jumps up to `ceil(max_step)`; this is not a
  bound on Euclidean speed or on pixel displacement due to rotation. Zero
  disables translation but does not disable spin.
- Sweeps shrink about their centre when needed to respect the continuous
  limit. They need not reach both endpoints or fill every spatial slice.
- Ink bounds use a threshold 15% above the source image's minimum intensity.
  Low-intensity background pixels may clip at the block boundary. Each block
  uses the per-channel source minimum as its background; the surrounding
  canvas uses zero.
- Spin selects an initial quarter-turn orientation and one to three turns
  (limited by the available time steps), with a random direction. Rotations
  happen before resizing. For a single time step only the initial orientation
  is observed.
- Sampling advances generator state on each access. Equal seeds reproduce
  equal access sequences; a sample is not a fixed function of its index.
  The datamodule starts train/validation/test generators at base seed plus
  1000/2000/3000. Loader workers receive distinct PyTorch worker seeds.
  Reproduction also requires the same loader order and worker count. Validation
  draws new motion samples on successive passes; compare frozen samples when
  exact repeated inputs are required.

These semantics describe the public implementation. They do not certify the
source used for historical research runs. No research results, cluster
manifests, colour-conditioning variants or unpublished model changes are
included with this dataset.

## Validation

```bash
python -m pytest tests/test_spatial_recall_motion.py -q -o addopts=''
```

The tests use synthetic tensors and run on CPU without dataset downloads.
