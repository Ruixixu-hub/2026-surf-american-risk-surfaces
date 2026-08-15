# Boundary-aligned / Localized Basis 可证伪实验结论

本阶段只使用 202 个 train 快照和 19 个 validation regimes；没有读取 test、stress 或 no-dividend call。
oracle boundary 只用于诊断，结果不是可部署方法，也不是在线速度结果。

## Transformation gate

- put: `DEFER_INTERPOLATION`；选定 canonical points = None。
- call: `DEFER_INTERPOLATION`；选定 canonical points = None。

## Basis decision

- put: `STOP_RB_ROUTE`；selected = None。
  - 最佳完整 price 配置：L, m=16, bins=2；worst RMSE=7.38247e-05。
  - 最佳 boundary 配置：L, m=4, bins=4；worst conditional MAE=0.104294（门槛 0.066667）。
  - 最佳 active-set 配置：L, m=8, bins=4；minimum F1=0.927477（门槛 0.98）。
- call: `STOP_RB_ROUTE`；selected = None。
  - 最佳完整 price 配置：L, m=8, bins=9；worst RMSE=0.000110505。
  - 最佳 boundary 配置：L, m=4, bins=9；worst conditional MAE=0.521667（门槛 0.066667）。
  - 最佳 active-set 配置：L, m=8, bins=9；minimum F1=0.977862（门槛 0.98）。

总决策：`STOP_RB_ROUTE`。

若 aligned transform 被 DEFER，本结论只否定当前 PCHIP/Jacobian 实现达到预注册门槛，不能据此断言所有 alignment 都无效。
若 physical localization 仍未通过绝对 gate，则下一方法固定为 positive-premium basis operator。

## 解释

physical localization 的确能在部分配置显著降低价格或边界误差，但没有一个配置同时通过 price、boundary、Greek、active-set 和 full-LCP residual 门槛。
因此不能用“某一个平均价格指标改善”替代整个 obstacle problem 的结构性验证；本阶段不打开 held-out。
