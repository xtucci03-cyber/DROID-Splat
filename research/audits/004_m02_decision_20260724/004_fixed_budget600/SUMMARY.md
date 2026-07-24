# 004 M01 Fixed Candidate Budget 600

## 验收结论

验收状态：**PASS**

本实验是固定预算阶段的强边界消融。
Budget 600真实影响67/68
个普通准入事件，拒绝16,825个候选。

## 核心指标

| 指标 | 004结果 | 相对001 |
|---|---:|---:|
| 最终Gaussian | 57,860 | -14.350% |
| All-frame ATE | 0.015212924 m | +0.000000193 m |
| Keyframe PSNR | 23.534516 dB | -0.137754 dB |
| Keyframe SSIM | 0.999555615 | -0.000013034 |
| Keyframe LPIPS | 0.201882 | +0.001230 |
| Online Mean VRAM | 5150.11 MiB | -32.76 MiB |
| Online Peak VRAM | 8946 MiB | +43 MiB |
| 墙钟时间 | 297 s | -2 s |

## M01运行行为

- Candidate total：64,678
- Ordinary candidate total：57,623
- Admitted total：47,853
- Dropped total：16,825
- 全部候选减少：26.013%
- 普通候选减少：29.198%
- Fixed-budget applied：67/68
- Contract violations：0

## 与Budget 700的边际变化

- 额外拒绝候选：6,417
- 最终Gaussian额外减少：1,199
- 最终Gaussian相对700变化：-2.030%
- Keyframe PSNR相对700变化：-0.056209 dB
- 墙钟时间相对700变化：+1 s
- Online Mean VRAM相对700变化：
  +45.04 MiB

## 当前解释

Budget 600将候选丢弃率提高到约26%，最终Gaussian相对官方
Baseline下降约14.35%，同时PSNR、SSIM和ATE仍处于安全范围。

但相对Budget 700，额外拒绝大量候选只带来约2.03%的最终Gaussian
下降，时间和平均显存没有同步改善。在当前fr1_desk单次实验中，
固定预算已表现出明显边际收益递减。

该结果支持停止继续密集扫描固定预算，并进入动态预算、
质量保护或覆盖保护机制设计。该结论仍需其他序列和重复运行验证。
