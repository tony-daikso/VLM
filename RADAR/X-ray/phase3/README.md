# RADAR X-ray train (Phase 3 port)

2D fork of `RADAR/CT/RADAR_train/`, ported per `RADAR/X-ray/PLAN.md` Phase 3.
See that file for the overall 5-phase plan; this directory only covers the
model-architecture port (Phase 3), not the dataset pipeline (Phase 4).

## What changed vs. the CT version

- **`lavis/models/radar_models/vision_branch.py`**: `PlainConvUNetLightD`
  rebuilt with `Conv2d`/`BatchNorm2d` and 2-tuple `kernel_sizes`/`strides`
  instead of `Conv3d`/`BatchNorm3d`/3-tuples (the underlying
  `dynamic_network_architectures` library is dimension-agnostic and needed
  no changes itself). Downsampling schedule: stage 0 full-res, stages 1-5
  each halve H,W (cumulative 2/4/8/16/32x) -- the CT config's stage-0/1
  asymmetry only existed to handle anisotropic CT slice spacing along D,
  which doesn't apply here. `forward()`'s `mode='trilinear'` ->
  `'bilinear'` and `max_pool3d` -> `max_pool2d` (kernel/stride
  `(2,8,8)/(4,16,16)/(8,32,32)` -> `(8,8)/(16,16)/(32,32)`, dropping the
  depth-axis factor). `organs` is now `["left_lung", "right_lung", "heart"]`
  (3 CheXmask regions instead of 36 TotalSegmentator organs -- heart
  doubles as heart/mediastinum, see `RADAR/X-ray/phase2/region_rules.py`).
  `checkpoint_unet.pth` loading was removed entirely: there's no X-ray
  equivalent (a 2D UNet's weights are rank-4, incompatible with the CT
  checkpoint's rank-5 Conv3d weights anyway), so the segmentation head
  trains end-to-end from scratch using CheXmask masks as the Dice-loss
  target.
- **`lavis/models/radar_models/radar_pretrain.py`**: `organs` list simplified
  to the same 3 regions (dropped the 36-organ Chinese/English dict). The
  `intact_organ_ids` boundary check (was this file's only *hardcoded*
  dimension-specific spot -- everywhere else, e.g. `forward_test_win`'s
  version, already loops generically over `mask.dim()`) changed from
  checking 6 faces of a (D,H,W) cuboid to 4 edges of an (H,W) rectangle.
  The Dice-loss target size changed from `shape[-3:]` to `shape[-2:]`.
  Everything else in this file (RADAR+'s even/odd-iteration alternation,
  the cross-attention ITC pooling, `dice.py`) is confirmed dimension-agnostic
  and needed zero changes -- it all operates on pre-flattened token
  sequences or dynamically-computed spatial axes.
- **`lavis/processors/radar_processors.py`**: dropped the third (depth-axis)
  `RandFlipd`, keeping only the 2 (H,W) flips.

## What's NOT ported (out of scope for this PoC)

`infer_merlin_anatomy.py`, `infer_merlin_whole.py`, `preprocess_code/`,
and `calc_metrics.py` are copied over unchanged from the CT track but are
CT/MERLIN/NIfTI-specific (external test-set inference, TotalSegmentator
mask merging, `.nii.gz` I/O) and not needed for the X-ray training loop.
They're left in place as reference rather than deleted, but treat them as
stale until/unless a later phase actually ports them.

`caption_datasets.py` (and the fixed `(96,256,384)` `SpatialPadd`/
`CenterSpatialCropd`/NIfTI-loading dataset pipeline it contains) is
**intentionally untouched** -- writing its X-ray equivalent (PNG + 2D mask +
region caption dataset) is Phase 4's job, not this one.

## Verification done

`VisionBranch` was smoke-tested standalone with a dummy `(2, 1, 512, 512)`
batch (no real data / no dataset pipeline needed for this, since Phase 4
hasn't been built yet):

```
organs: ['left_lung', 'right_lung', 'heart']
total params: 9.2M
seg_probs:  (2, 4, 256, 256)      # 3 organs + background, half-res pre-upsample
pred_mask:  (2, 512, 512)         # full input resolution
res_x1/2/3: (2, 256, 256) / (2, 1024, 256) / (2, 4096, 256)   # 16x16 / 32x32 / 64x64 token grids
flags1/2/3: (2, 3, 256) / (2, 3, 1024) / (2, 3, 4096)
```

All shapes match the designed downsampling schedule. `radar_pretrain.py`'s
two line-level fixes (boundary check, Dice target size) were verified by
direct code review against the working 3D pattern, not by a runtime test --
a full `RadarPretrain` forward pass needs a real batch (image + seg + per-
region captions) and a `bert-base-uncased` checkpoint, neither of which
exist yet; that end-to-end run is Phase 5's job once Phase 4's dataset
pipeline exists.
