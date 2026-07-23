# 001 官方完整序列 Baseline

## 实验定位

- 实验：`001_official_baseline_fr1desk_full_eval_r1`
- 数据集：TUM RGB-D `fr1/desk`完整序列
- 代码：纯官方提交 `f0cb0316e1ee7f88f8a63c37a6fe19dd623e6eb2`
- M01：未启用
- 评价：ATE、PSNR、SSIM、LPIPS、Depth L1全部开启
- 验收：**PASS**

## 核心结果

| 指标 | 结果 |
|---|---:|
| 最终 Gaussian | 67,554 |
| 在线 SLAM 时间 | 2.50 min |
| 在线 Total FPS | 3.95 |
| 端到端墙钟时间 | 299 s |
| All-frame ATE RMSE | 0.015213 m |
| Keyframe ATE RMSE | 0.017720 m |
| Keyframe PSNR | 23.6723 dB |
| Keyframe SSIM | 0.999569 |
| Keyframe LPIPS | 0.200651 |
| Non-keyframe PSNR | 22.8491 dB |
| Non-keyframe SSIM | 0.999509 |
| Non-keyframe LPIPS | 0.217845 |
| Online Mean VRAM | 5182.87 MiB |
| Online Peak VRAM | 8903 MiB |

## 资源统计口径

主论文资源指标采用在线SLAM阶段：

- Mean VRAM：5182.87 MiB
- Peak VRAM：8903 MiB

评价阶段与端到端显存单独保留，避免离线评价阶段拉低在线平均显存。

## 结果完整性

- 最终PLY：存在
- 全帧与关键帧轨迹评价：存在
- 关键帧与非关键帧渲染评价：存在
- 渲染RGB数量：关键帧 69，非关键帧 105
- GPU有效采样：294
- Traceback：0
- CUDA错误/OOM：0

## 注意事项

日志出现CUDA IPC共享张量退出警告 2 次。程序退出码为0，评价、轨迹、渲染和PLY均完整生成，因此记录为退出阶段警告，不判定为实验失败。

`metrics_summary.json`为根据官方评价JSON、日志、PLY及GPU监控生成的统一汇总，不是官方程序原生文件。
