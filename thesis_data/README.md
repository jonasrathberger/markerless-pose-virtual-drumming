# Thesis Data

This directory is the local regeneration target for thesis data products. Generated CSVs, generated overlay images, and metric outputs in this tree are ignored by Git.

Tracked contents are limited to small reference inputs and placeholders:

- `virtual_drumming/evaluation_60bpm.json`: evaluation annotation input.
- `.gitkeep` files: preserve expected output directories in a fresh checkout.

Large source recordings and regenerated data should be supplied locally:

```text
thesis_data/
  raw/mocap/
    air_forefoot.csv
    air_knees.csv
    drums.csv
  raw/pose/apple_vision/
    landmarks.csv
    metadata.json
  raw/pose/mediapipe/
    landmarks.csv
    metadata.json
  virtual_drumming/
    evaluation_60bpm.mp4
```

The thesis pipelines recreate ignored outputs under:

- `thesis_data/pose_evaluation/`
- `thesis_data/motion_analysis/`
- `thesis_data/virtual_drumming/`

Final generated figures are written under `thesis_figures/` and are also ignored by Git.
