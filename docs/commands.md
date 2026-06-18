# Command Reference

Short command list for the main repository folders. Run commands from the repository root unless a `cd` step is shown.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Raw Data

Large raw files are ignored by Git. Put OptiTrack CSVs in `thesis_data/raw/mocap/` as `air_feet.csv`, `air_knees.csv`, and `drums.csv`. Put pose recordings in `thesis_data/raw/pose/apple_vision/` and `thesis_data/raw/pose/mediapipe/`; each session should contain `landmarks.csv` and `metadata.json`. Put virtual-drumming evaluation media in `thesis_data/virtual_drumming/`, especially `evaluation_60bpm.mp4` and `evaluation_60bpm.json`.

## pose_recording/

```bash
cd pose_recording
python main.py --list-backends
python main.py --list-cameras
python main.py --backend apple_vision
python main.py --backend apple_vision --no-preview-hud
python main.py --backend mediapipe_pose_hands
python main.py --backend apple_vision --save-raw-video --save-annotated-video
python scripts/validate_recording.py recordings/<session_id>
```

Use this folder to record new Apple Vision or MediaPipe landmark sessions.

## motion_evaluation/pose_evaluation/

```bash
cd motion_evaluation/pose_evaluation
python data_preparation/prepare_landmark_csvs.py
python data_alignment/trim_to_alignment_window.py
python data_analysis/run_analysis.py
python data_alignment/generate_thesis_alignment_figures.py
python data_analysis/generate_thesis_result_figures.py
```

Use this folder to prepare pose and mocap landmark CSVs, align trajectories, run pose-estimator evaluation metrics, and generate pose-evaluation figures.
Default CSV outputs are written under `thesis_data/pose_evaluation/`; pass explicit output arguments only for scratch runs.

## motion_evaluation/motion_analysis/

```bash
cd motion_evaluation/motion_analysis
python run_full_analysis.py
python generate_results_section_figures.py
```

Use this folder to analyze OptiTrack motion recordings and generate motion-analysis figures.
Default analysis outputs are written under `thesis_data/motion_analysis/`; non-final diagnostic plots require an explicit `--diagnostic-dir`.

## virtual_drumming_app/

```bash
cd virtual_drumming_app
python apple_vision_live.py
python apple_vision_live.py --list-cameras
python apple_vision_live.py --list-midi-outputs
python apple_vision_live.py --midi-out
python apple_vision_live.py --midi-out "<port name>"
python apple_vision_live.py --no-midi-out
```

Run the live Apple Vision prototype, inspect cameras and MIDI devices, and choose whether detected drum hits are printed only or sent to MIDI.

```bash
python apple_vision_live.py --collect-target-samples target_samples_sticks_100.json --samples-per-target 50
python apple_vision_live.py --collect-target-samples target_samples_sticks_100.json --collection-raw-video target_sample_videos/session.mp4
python -m drum_engine.train_target_model target_samples_sticks_100.json knn_100.json --k 5
```

Collect labeled hand-target samples and train the KNN drum-target model.

```bash
python apple_vision_live.py --record-evaluation-video ../thesis_data/virtual_drumming/evaluation_60bpm.mp4 --evaluation-annotations ../thesis_data/virtual_drumming/evaluation_60bpm.json
python -m drum_engine.evaluate_recording ../thesis_data/virtual_drumming/evaluation_60bpm.mp4 --midi computer.mid
python -m drum_engine.evaluate_recording ../thesis_data/virtual_drumming/evaluation_60bpm.mp4 --midi human.mid --json
python -m drum_engine.export_thesis_metrics
python -m drum_engine.export_thesis_graphics
```

Record the evaluation take, compare it with reference MIDI files, and export virtual-drumming thesis metrics and figures.

```bash
python -m drum_engine.analyze_target_features target_samples_sticks_100.json
python -m drum_engine.rebuild_target_samples target_samples_sticks_100.json rebuilt_target_samples.json
python -m drum_engine.clean_reference_midi input.mid output.mid
```

Optional utilities for target-sample diagnostics, sample-feature rebuilding, and MIDI cleanup. Generated JSON reports from these utilities are scratch/debug artifacts, not default thesis outputs.

## thesis_figures/

```bash
python -m thesis_figures.generate
python -m thesis_figures.generate --sections pose-evaluation
python -m thesis_figures.generate --sections motion-analysis
python -m thesis_figures.generate --sections motion-evaluation
python -m thesis_figures.generate --sections virtual-drumming
```

Use this package to regenerate final thesis figures after the relevant pipeline has recreated its local data products in `thesis_data/`.

## thesis_data/

`thesis_data/raw/` is for ignored local source files. Generated thesis CSV/data outputs are written under `thesis_data/` but are not committed; only small reference inputs such as `virtual_drumming/evaluation_60bpm.json` are tracked.

Default thesis pipelines write regenerated CSV/data products to `thesis_data/` and final figures to `thesis_figures/`; both are ignored by Git. Source-local `result/`, `results/`, `output/`, and `reports/` folders are scratch leftovers only.
