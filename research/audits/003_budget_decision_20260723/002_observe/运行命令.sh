#!/usr/bin/env bash
set -euo pipefail

cd "/home/XT/gsslam/1_droidsplat/code/DROID-Splat"

exec "/home/XT/.local/miniconda3-droidsplat/envs/droidsplat/bin/python" run.py \
  data=TUM_RGBD/fr1 \
  data.input_folder="/home/XT/gsslam/3_datasets/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk" \
  tracking=tum \
  mapping=tum \
  mode=rgbd \
  stride=1 \
  run_frontend=True \
  run_backend=True \
  run_mapping=True \
  run_loop_detection=False \
  evaluate=True \
  backend_every=8 \
  mapper_every=20 \
  mapping.refinement.iters=0 \
  mapping.refinement.sampling.use_non_keyframes=False \
  mapping.online_opt.pruning.use_covisibility=True \
  render_images=True \
  save_rendered_predictions=True \
  run_visualization=False \
  run_mapping_gui=False \
  show_stream=False \
  +mapping.resource_admission.mode=observe \
  hydra.job.name=002_m01_observe_fr1desk_full_eval_r1 \
  hydra.run.dir="/home/XT/gsslam/1_droidsplat/records/002_m01_observe_fr1desk_full_eval_r1"
