# 002 M01 Observe完整序列实验

## 实验定位

- 数据集：TUM RGB-D `fr1/desk`完整序列
- 方法：M01 Observe
- 对照：001纯官方Baseline
- 候选行为：全部透传
- 验收：**PASS**

## 核心结果

| 指标 | 结果 |
|---|---:|
| M01事件 | 69 |
| Candidate | 64,651 |
| Admitted | 64,651 |
| Dropped | 0 |
| 最终Gaussian | 67,420 |
| All-frame ATE | 0.015211889 m |
| Keyframe PSNR | 23.770012 dB |
| Non-keyframe PSNR | 22.847126 dB |
| Online Mean VRAM | 5122.36 MiB |
| Online Peak VRAM | 9540 MiB |
| 墙钟时间 | 298 s |
| M01平均CPU开销 | 0.232640 ms |

Observe满足Candidate=Admitted且Dropped=0。与001的质量、
ATE和最终地图规模基本一致。单次Peak VRAM存在采样波动，
不据此判断Observe资源开销。
