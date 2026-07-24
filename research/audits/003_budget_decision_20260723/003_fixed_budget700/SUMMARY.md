# 003 M01 Fixed Candidate Budget 700

## 实验结论

验收状态：**PASS**

固定预算700真实影响61/68个普通准入事件，
拒绝10,408个候选，最终Gaussian为
59,059。

## 核心指标

| 指标 | 003结果 | 相对001 |
|---|---:|---:|
| 最终Gaussian | 59,059 | -12.575% |
| All-frame ATE | 0.015216059 m | +0.000003329 m |
| Keyframe PSNR | 23.590724 dB | -0.081546 dB |
| Keyframe LPIPS | 0.202006 | +0.001355 |
| Online Mean VRAM | 5105.08 MiB | -77.79 MiB |
| Online Peak VRAM | 9457 MiB | +554 MiB |
| 墙钟时间 | 296 s | -3 s |

## M01行为

- Candidate：64,658
- Admitted：54,250
- Dropped：10,408
- 全部候选减少：16.097%
- 普通候选减少：18.069%
- Fixed-budget applied：61/68
- Contract violations：0

## 当前解释

Fixed Budget 700能够显著减少最终Gaussian，质量与ATE损失很小。
但单次运行的时间与nvidia-smi显存下降有限，Peak VRAM不能作为
显著资源收益证据。该方法目前属于固定预算工程控制点和消融基线，
不是最终动态资源感知方法。
