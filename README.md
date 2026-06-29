# Markerless Pose Estimation for Virtual Drumming

This repository contains the code and curated thesis artifacts for my Master thesis **Markerless Pose Estimation for Virtual Drumming: Detecting Hits and Classifying Drum Targets from Body and Hand Motion**. The project investigates camera-based and marker-based drumming motion, evaluates Apple Vision and MediaPipe pose estimates against OptiTrack recordings, and implements a real-time Apple Vision virtual drumming prototype with hit detection, drum-target classification, MIDI output, and evaluation tooling.

The repository is organized as a thesis reproduction bundle: source code, pipeline configuration, small reference inputs, and the trained virtual-drumming KNN artifacts are included. Generated thesis CSVs, figure images, large raw recordings, videos, local model bundles, and regenerated intermediate folders are intentionally ignored; recreate them locally from the original data.

## Repository Structure

- `pose_recording/`: recorder for Apple Vision and MediaPipe landmark sessions.
- `motion_evaluation/`: OptiTrack motion analysis and pose-estimation evaluation pipelines.
- `virtual_drumming_app/`: real-time virtual drumming prototype and evaluation scripts.
- `thesis_data/`: ignored local thesis data outputs plus small tracked reference inputs.
- `thesis_figures/`: shared figure-generation code and ignored local figure outputs.
- `docs/`: short folder-by-folder command reference.

## Setup

### Tested Environment

The project was developed and tested primarily on a MacBook Pro with Apple Silicon using macOS 26.5.1 and Python 3.11.15. The working environment was managed with conda, while the Python package dependencies are listed in `requirements.txt`.

A basic conda setup can be created with:

```bash
conda create -n virtual-drumming python=3.11
conda activate virtual-drumming
python -m pip install -r requirements.txt
```

The live Apple Vision prototype requires macOS and PyObjC bindings for Apple frameworks. The analysis and figure-generation scripts are more portable, but the real-time Apple Vision application is macOS-specific.

## Raw Data Placement

Large source files are not committed. If you have the raw files, place OptiTrack CSVs in `thesis_data/raw/mocap/` as `air_forefoot.csv`, `air_knees.csv`, and `drums.csv`. Place pose-estimation recordings in `thesis_data/raw/pose/apple_vision/` and `thesis_data/raw/pose/mediapipe/`; each folder should contain `landmarks.csv` and `metadata.json`. Place the virtual-drumming evaluation media in `thesis_data/virtual_drumming/`, especially `evaluation_60bpm.mp4` and `evaluation_60bpm.json`.

## Thesis Commands

For a folder-by-folder command reference, see `docs/commands.md`.

Regenerate all final thesis figures after running the data pipelines:

```bash
python -m thesis_figures.generate
```

Run the pose-estimation evaluation pipeline and regenerate its figures:

```bash
cd motion_evaluation/pose_evaluation
python data_preparation/prepare_landmark_csvs.py
python data_alignment/trim_to_alignment_window.py
python data_analysis/run_analysis.py
cd ../..
python -m thesis_figures.generate --sections pose-evaluation
```

The pose commands write ignored prepared, aligned, and metric CSVs under `thesis_data/pose_evaluation/`.

Run the OptiTrack motion-analysis pipeline and regenerate its figures:

```bash
cd motion_evaluation/motion_analysis
python run_full_analysis.py
cd ../..
python -m thesis_figures.generate --sections motion-analysis
```

The motion-analysis command writes ignored canonical figure-input CSVs and required overlay PNGs under `thesis_data/motion_analysis/`.

Regenerate both motion-evaluation figure sets:

```bash
python -m thesis_figures.generate --sections motion-evaluation
```

Regenerate the virtual-drumming evaluation metrics and figures:

```bash
cd virtual_drumming_app
python -m drum_engine.export_thesis_metrics
cd ..
python -m thesis_figures.generate --sections virtual-drumming
```

The virtual-drumming metric export writes ignored CSV outputs under `thesis_data/virtual_drumming/`; generated JSON reports are opt-in debugging artifacts.

Run the live virtual drumming prototype:

```bash
cd virtual_drumming_app
python apple_vision_live.py
```

Record new pose-estimation sessions, if needed:

```bash
cd pose_recording
python main.py --backend apple_vision
python main.py --backend mediapipe_pose_hands
python scripts/validate_recording.py recordings/<session_id>
```

## Notes

Default thesis pipelines write regenerated CSV/data products to `thesis_data/` and final figures to `thesis_figures/`, but those generated outputs are ignored by Git. Source-local generated folders such as `motion_evaluation/**/result`, `motion_evaluation/**/results`, `motion_evaluation/**/output`, and `motion_evaluation/**/reports` are treated as scratch leftovers and ignored by Git.

This project is licensed under the MIT License. See [LICENSE](LICENSE).
