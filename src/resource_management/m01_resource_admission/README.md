# M01 Resource Admission

## 1. 模块目标

在新增 Gaussian 进入持久地图和优化器之前进行资源准入管理。

## 2. 当前阶段

M01 当前支持：

- Candidate Observe：只观察，不筛选。
- Fixed Candidate Budget v1：使用固定候选预算进行确定性准入。

## 3. 官方数据流

```text
create_pcd_from_image
→ 五组候选张量
→ M01准入接入口
→ extend_from_pcd
→ 持久地图和优化器
```

## 4. 官方接入口

- `src/gaussian_mapping.py`
- `src/gaussian_splatting/scene/gaussian_model.py`

## 5. 当前输入

`xyz`、`features`、`scales`、`rotations`、`opacities`、`camera_uid`、`kf_id`、`init`、`gaussian_before`。

## 6. 当前输出

`AdmissionResult`与单行 JSON 准入事件日志。

## 7. disabled行为

配置不存在时 `ResourceAdmission` 为 `None`，直接走官方 `extend_from_pcd` 路径。

## 8. observe行为

候选原对象透传，`candidate_count=admitted_count`，`dropped_count=0`。

## 9. fixed_budget行为

配置字段：

- `mapping.resource_admission.mode=fixed_budget`
- `mapping.resource_admission.fixed_budget=<正整数>`
- `mapping.resource_admission.selection=deterministic_uniform`

初始化候选始终受保护并全部放行。普通相机候选数不超过预算时原对象透传；超过预算时，使用同一组严格递增的 `deterministic_uniform` 索引选择固定数量候选。

## 10. 当前未实现

- control
- dynamic budget
- quality-aware selection
- coverage protection
- lifecycle pruning

## 11. 日志前缀

`[M01:ResourceAdmission]`

## 12. 当前分支

`feat/m01-resource-admission-fixed-budget`

## 13. 首个功能提交

`a24b16b2363654ad591302bab37640c984474666`

## 14. 已封存观察实验

`002_m01_candidate_observe_fr1desk_300f`

## 15. 计划诊断实验

`003_m01_fixed_candidate_budget700_fr1desk_300f`
