# 护具仿真 Auto Research 计划

目标：为 PM01 摔倒仿真找到一个"好的"护具建模方法，用 AMP 摔倒策略
（`engineai_ros2_workspace/src/interface_example/config/pm01/rl_basic/basic/rl_basic_param_AMP.yaml`）闭环测试，
在**力衰减符合实验**的同时，**不穿模、不打滑、数值稳定、速度可用**。

## 1. 待比较的方法

| 代号 | 方法 | MuJoCo | 说明 |
|---|---|---|---|
| M0 | 无护具，刚体接触 | 3.2.3 / 4.0 | 参照基线：穿模、打滑的"正常水平" |
| M1 | 现行方案：CHR 公式事后缩放 `efc_force` + 低通 + min-scale | 3.2.3 / 4.0 | 现状，`sim_manager.cc` `ApplyProtectionToContactForces()` |
| M2 | 柔顺接触：按护具厚度给每个 link geom 设 `solref`/`solimp`（width≈厚度，power 做非线性） | 3.2.3 / 4.0 | 不建 mesh，接触力由求解器算出，无需事后改力 |
| M3 | 3D flex 护具垫 pin 在硬 link 上 | 4.0 | 真实建模；材料 StVK / Stable Neo-Hookean (SNH)，可选 `yield` 塑性 |

M2 是 M1 和 M3 之间的折中，若 M2 已足够好，M3 主要作为物理佐证。

## 2. "好仿真"判据（打分卡，每次试验自动算）

对每个场景、每个方法，以同场景 M0 为参照：

1. **穿模**：硬 link 与地面最大穿透深度 `max(-contact.dist)`；M3 另记护垫压缩率 = 压缩量 / 厚度。
   通过：硬 link 穿透 ≤ M0 + 2 mm；护垫不"压穿"（压缩率 < 90%）。
2. **打滑**：着地期间接触点切向滑移速度、接触 link 和 root 的水平滑移距离（着地到静止）。
   通过：滑移距离在 M0 的 ±20% 内（护具不应改变摩擦行为）。
3. **力衰减**：护具区域峰值法向力 vs CHR 预测值 `CHR(F_M0峰值, t)`。
   通过：对数误差 |ln(F/F_CHR)| ≤ 0.25（约 ±28%）。
4. **稳定性**：无 NaN、无发散，无"弹飞"（护具 link 动能尖峰不超过 M0 的 1.5 倍）；全部场景 100% 跑完。
5. **速度**：headless 实时率 ≥ 1×（否则无法批量采样），记录每步耗时。

判据阈值是初值，跑完 M0/M1 基线后按数据再定一次，之后冻结，避免边调边改标准。

## 3. 测试场景

复用 `scripts/automated_collection_continuous.sh` 的 AMP 站立推倒流程：
- 推力方向：8 个方向（前、后、左、右、4 个斜向）× 2 档推力，共 16 个场景，固定随机种子。
- 研究期间用这 16 个快速迭代；候选方法最终在你论文用的完整 24 方向集上复验。
- 每次运行保存 contact / joint / kinetic energy CSV 到 `~/data/mujoco_logs/protector_research/<方法>_<参数hash>/`。

## 4. 阶段

### P0 基础设施（不改物理）
- 在 ROS 仿真的 contact CSV 中新增两列：`penetration`（`-contact.dist`）和 `slip_speed`（接触点相对切向速度，`mj_contactForce` 同帧用 `mj_objectVelocity` 差计算）。
- 写 `protector_research/score.py`：读 CSV 输出打分卡（第 2 节）和一张汇总表。
- 跑 M0、M1（3.2.3）基线，确定判据阈值。

### P1 离线落锤标定（独立于机器人，快）
- 用 MuJoCo 4.0 复现落锤实验：刚性冲头质量 1–10 kg，护垫 6/12/18/24 mm，
  目标数据 `scripts/ThicknessCalculate/RT-FEM.tsv`（泡沫）与 `fitted_parameters_silicone_meta.json`（硅胶）。
- M3：扫 `young`、`poisson`、`damping`、`yield`、网格密度、timestep；对比 StVK 与 SNH（SNH 只能用 mjSpec 打开）。
- M2：用同一落锤场景扫 `solref`/`solimp`，拟合 CHR 曲线。
- 产出：每种厚度的最佳参数表 + 拟合误差图。pip 上没有 4.0，需从源码编译 Python 绑定（或用 C++ 写落锤程序）。

### P2 ROS 仿真移植到 MuJoCo 4.0
- `mujoco_simulator` 改为链接本 fork 编译的 4.0 库（单独 CMake 选项，不动 `/opt` 下的 3.2.3）。
- 修 4.0 API 破坏性变更；vendored `simulate/` 视情况更新。
- 验收：M0、M1 在 4.0 上与 3.2.3 的打分卡一致（误差 < 5%）。否则后续对比不公平。

### P3 机器人上实现 M2、M3
- M2：从 `protector_map` 的厚度表生成每个 link geom 的 `solref`/`solimp`（扩展现有 `apply_solref_6mm`）。
- M3：先在 1–2 个高风险 link（按 fig2 接触频率选，预计膝、肘）挂 flex 垫；顶点 pin 到 link，`contype` 设置为不与本 link 碰撞；
  `integrator="discrete"` + Newton。跑通后再扩展到 protector map 覆盖的全部区域。
- 注意：discrete 积分器本身会改变刚体动力学，M3 的参照基线必须是"4.0 + discrete 下的 M0"。

### P4 自动迭代
循环：改参数 → 跑 16 场景 → 打分 → 记录 → 选下一组参数。每轮写入 `protector_research/RESULTS.md`
（参数、打分卡、结论、失败原因），直到某方法全部判据通过，或确认某方法做不到并写明原因。

## 5. 代码与数据存放

- 本仓库（fork `wczmiracle/mujoco`，分支 `protector-research`）：计划、落锤标定脚本、flex 护垫模型、打分脚本、`RESULTS.md`。
- `engineai_ros2_workspace`：仿真器改动（CSV 新列、4.0 链接、M2/M3 加载）放在从 `bench` 拉出的新分支 `protector-sim`。
- 原始数据：`~/data/mujoco_logs/protector_research/`，不进 git。

## 6. 主要风险

- 4.0 的 Python 绑定需要自己编译；若失败，落锤标定改用 C++。
- flex 可能需要比 0.002 s 更小的 timestep，实时率可能低于 1×；对策：interpolated flex（`dof="trilinear"`）减少自由度，只在高风险 link 挂垫。
- StVK 在大压缩下会软化甚至单元翻转，泡沫的致密化可能拟合不出来；对策：SNH、塑性、或回退到 M2。
- 非牛顿泡沫的率相关性 flex 不支持（只有线性 Rayleigh 阻尼），M3 可能只能贴合某一冲击速度区间；结论中如实说明适用范围。
