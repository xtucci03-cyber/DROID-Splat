# M02方法决策说明

M01固定预算阶段已经完成以下严格对照：

- 001：官方Baseline
- 002：Resource Admission Observe
- 003：Fixed Budget 700
- 004：Fixed Budget 600

004结果：

- candidate_total=64678
- dropped_total=16825
- candidate drop=26.013%
- final_gaussians=57860
- 相对001 Final Gaussian下降14.350%
- Keyframe PSNR相对001下降0.137754 dB
- All-frame ATE基本不变
- Online Mean VRAM=5150.11 MiB
- Wall time=297 s

相对Budget 700：

- 额外拒绝候选6417
- 最终Gaussian仅额外减少1199
- PSNR下降0.056209 dB
- 时间增加1秒
- Online Mean VRAM增加约45 MiB

需要独立判断下一模块M02优先方向：

A. 资源状态驱动的动态预算
B. 重建质量/误差感知候选选择
C. 空间覆盖感知候选选择
D. 动态预算与质量/覆盖选择联合设计

固定预算只是控制基线和消融，不是最终创新方法。
不得继续建议密集扫描550、650或500等固定预算点。
