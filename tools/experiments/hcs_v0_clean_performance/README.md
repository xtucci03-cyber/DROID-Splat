# HCS-v0 Clean Performance Pair v1

本目录固化 HCS-v0 的正式配对性能协议。它只包含运行与验收脚本，不修改
DROID-Splat、HCS、Gaussian、Loss、Mapping iteration、Observer 或 CUDA
算法实现。

## 固定基线

- 算法父提交：`f3fb8d0dfb9bb1b774132777b7cfa3d7b2628230`
- 实验包分支：`exp/hcs-v0-clean-performance-pair-v1`
- 服务器 worktree：
  `/home/XT/gsslam/1_droidsplat/code/DROID-Splat-hcs-v0-clean-perf-pair-v1`
- 数据集：完整 TUM RGB-D `fr1/desk`
- Python：
  `/home/XT/.local/miniconda3-droidsplat/envs/droidsplat/bin/python`
- CUDA 物理设备：`0`
- `PYTHONHASHSEED=43`；`run.py` 仍使用项目既有 seed 43

脚本在运行前要求：

1. 当前分支名精确为实验包分支；
2. 当前 `HEAD` 与本地 `origin/exp/hcs-v0-clean-performance-pair-v1`
   远程跟踪 ref 完全一致；
3. 当前提交只有一个父提交，且父提交精确为算法父提交；
4. 父提交到实验包提交之间仅包含本目录的三个协议文件；
5. worktree 干净；
6. `configs/`、`src/`、`tools/analysis/` 相对算法父提交无变化。

因此服务器在运行前必须显式 fetch 实验包分支。脚本把实际
`EXPERIMENT_PACKAGE_SHA` 和固定 `ALGORITHM_PARENT_SHA` 一并写入记录。

## Hydra 最终配置约束

命令先选用 `mapping=tum`，再显式冻结关键覆盖。TUM 的最终配置为：

- `mapping.online_opt.n_last_frames=10`
- `mapping.online_opt.n_rand_frames=20`
- `mapping.online_opt.iters=100`
- `mapping.camera_scheduler.preserve_all_history_until=30`

HCS 启用时要求
`0 <= history_budget <= n_rand_frames`，所以 Budget10 和 Budget20 都合法。
两组均保护最近 10 个历史 Camera 和全部 new Camera，只对更旧历史池应用
预算。

## 唯一配对变量

| 字段 | 018 | 019 |
|---|---|---|
| Run ID | `018_perf_hcs_v0_budget10_no_activity_fr1desk_full_r1` | `019_perf_hcs_v0_budget20_no_activity_fr1desk_full_r1` |
| 记录目录 | `/home/XT/gsslam/1_droidsplat/records/018_perf_hcs_v0_budget10_no_activity_fr1desk_full_r1` | `/home/XT/gsslam/1_droidsplat/records/019_perf_hcs_v0_budget20_no_activity_fr1desk_full_r1` |
| `mapping.camera_scheduler.history_budget` | `10` | `20` |
| 顺序 | 第一组 | 仅在 018 人工审核通过后 |

除 Run ID、对应记录目录、顺序门禁和 `history_budget` 外，两份脚本的算法、
Hydra、环境、评价、监控、Parser 与后处理逻辑相同。

## 共同运行条件

- `mode=rgbd`、`stride=1`
- Frontend、Backend、Mapping 开启
- Loop Detection、Visualization、Mapping GUI、stream display 关闭
- `backend_every=8`、`mapper_every=20`
- Mapping online optimization 使用 TUM 的 100 iterations
- `mapping.refinement.iters=0`
- M01 Resource Admission 为 `observe`
- Lifecycle Observer 关闭
- Mapping Activity Observer 关闭
- HCS 详细日志关闭
- Performance Resource Monitor 开启
- Performance Monitor GPU timing、memory sampling、结构化日志开启
- 外部 `nvidia-smi` 每秒采样
- `evaluate=true`
- `render_images=true`
- `save_rendered_predictions=true`
- 不设置 `t_start` 或 `t_stop`

Performance Resource Monitor 是唯一代码内性能观察器。它启用 GPU timing
时会在每个 Mapper update 粗粒度边界执行一次 Event 同步，并在 finalize
再同步一次；018 和 019 的监控配置完全相同。这里的“clean”表示 Activity、
Lifecycle 和 HCS 详细诊断全部关闭，不能把带 Performance Monitor 的绝对时间
解释成无监控原生时间。

## 固定运行顺序

先运行 018，人工审核其 `PASS_NOT_YET_SEALED` 结果。审核通过后，由人工在
018 记录目录创建：

```text
018_REVIEW_PASS.txt
```

且文件必须包含唯一非空行：

```text
018_REVIEW_PASS=true
```

019 脚本固定声明：

```text
019_REQUIRES_018_REVIEW_PASS=true
```

缺少该文件或内容不匹配时，019 fail-closed。两份脚本不会相互调用，也不会
自动串联。

## 服务器准备与运行

先在服务器获取并建立独立 worktree；以下只是一份人工执行模板，本仓库不会
自动连接服务器：

```bash
git -C /home/XT/gsslam/1_droidsplat/code/DROID-Splat fetch origin \
  refs/heads/exp/hcs-v0-clean-performance-pair-v1:refs/remotes/origin/exp/hcs-v0-clean-performance-pair-v1

git -C /home/XT/gsslam/1_droidsplat/code/DROID-Splat worktree add \
  /home/XT/gsslam/1_droidsplat/code/DROID-Splat-hcs-v0-clean-perf-pair-v1 \
  refs/remotes/origin/exp/hcs-v0-clean-performance-pair-v1

git -C /home/XT/gsslam/1_droidsplat/code/DROID-Splat-hcs-v0-clean-perf-pair-v1 \
  switch -c exp/hcs-v0-clean-performance-pair-v1 \
  --track origin/exp/hcs-v0-clean-performance-pair-v1
```

运行 018：

```bash
cd /home/XT/gsslam/1_droidsplat/code/DROID-Splat-hcs-v0-clean-perf-pair-v1
bash tools/experiments/hcs_v0_clean_performance/018_hcs_v0_budget10_clean_performance_run.sh
```

人工审核并写入 018 gate 后，才可运行 019：

```bash
cd /home/XT/gsslam/1_droidsplat/code/DROID-Splat-hcs-v0-clean-perf-pair-v1
bash tools/experiments/hcs_v0_clean_performance/019_hcs_v0_budget20_clean_performance_run.sh
```

不要在同一 Python 进程连续执行两组实验。每份脚本启动独立 Python
进程，并在运行前要求 GPU 没有计算进程。若环境温度或后台负载差异明显，
不得直接把单次时间差解释为算法收益。

## 记录与验收

每份脚本：

- 拒绝覆盖已有记录目录；
- 保存 Git、submodule、环境、完整 Hydra config/overrides、UTC 时间；
- 保存完整 `run.log`；
- 保存 GPU 整卡采样和 compute-process 采样；
- 检测运行期间非本次进程树的 GPU compute PID；
- 运行当前提交中的 `build_performance_summary.py`；
- 区分 `online_slam`、`evaluation`、`mapper_update`、
  `mapping_optimization`、`add_new_gaussians` 和
  `covisibility_pruning`；
- 保存地图规模、ATE、PSNR、SSIM、LPIPS 和 L1 depth；
- 生成 `RUN_STATUS.txt`、`PERFORMANCE_VALIDATION.json`、
  `PERFORMANCE_KEY_METRICS.json`、`POSTRUN_SUMMARY_中文.md`；
- 生成 `record_checksums.sha256`。

成功状态只能是 `PASS_NOT_YET_SEALED`；任何运行、Parser、配置、GPU污染、
评价或守恒门禁失败均为 `FAIL_PRESERVED`。脚本不会写
`PASS_SEALED`。

## 性能解释边界

- `online_slam` 在 `SLAM.run()` 中于 evaluation 之前闭合；
- `evaluation` 是独立 main-process stage；
- `mapper_update` 是父阶段，不能与其
  `mapping_optimization`、`add_new_gaussians`、`covisibility_pruning`
  子阶段相加；
- `nvidia-smi` 是整卡/驱动层辅助口径，不等同于 PyTorch allocator；
- Performance Monitor 的 `allocated`、`reserved` 与 max 字段属于 Mapper
  进程的 PyTorch allocator 口径；
- 单次配对差异接近噪声时，应人工决定是否进行反序重复；本协议不自动创建
  后续实验。
