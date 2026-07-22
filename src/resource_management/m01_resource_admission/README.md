# M01 Resource Admission

## 1. 模块目标

在新增 Gaussian 进入持久地图和优化器之前进行资源准入管理。

## 2. 当前阶段

Candidate Observe，只观察，不筛选。

## 3. 官方数据流

```text
create_pcd_from_image
→ 五组候选张量
→ M01观察接入口
→ extend_from_pcd
→ 持久地图和优化器
```

## 4. 官方接入口

- `src/gaussian_mapping.py`
- `src/gaussian_splatting/scene/gaussian_model.py`

## 5. 当前输入

`xyz`、`features`、`scales`、`rotations`、`opacities`、`camera_uid`、`kf_id`、`init`、`gaussian_before`。

## 6. 当前输出

`AdmissionResult`与单行 JSON 观察日志。

## 7. disabled行为

配置不存在时 `ResourceAdmission` 为 `None`，直接走官方 `extend_from_pcd` 路径。

## 8. observe行为

候选原对象透传，`candidate_count=admitted_count`，`dropped_count=0`。

## 9. 当前未实现

- control
- fixed budget
- dynamic budget
- quality-aware selection
- coverage protection
- lifecycle pruning

## 10. 日志前缀

`[M01:ResourceAdmission]`

## 11. 当前分支

`feat/resource-admission-candidate-observe`

## 12. 首个功能提交

`a24b16b2363654ad591302bab37640c984474666`

## 13. 对应诊断实验

`002_m01_candidate_observe_fr1desk_300f`
