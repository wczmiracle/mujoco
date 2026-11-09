# 护具仿真研究 Agent 说明

本文件给接手这项研究的 agent（或人）看：目标、数据、规则、怎么跑、怎么记录和汇报。
总计划见 [PLAN.md](PLAN.md)，所有结果按时间追加到 [RESULTS.md](RESULTS.md)。

## 目标

让 PM01 摔倒仿真里的护具**力衰减符合实验**，同时**不穿模、不打滑、数值稳定、速度可用**。
闭环测试策略：`engineai_ros2_workspace/src/interface_example/config/pm01/rl_basic/basic/rl_basic_param_AMP.yaml`。

方法代号：M0 无护具，M1 现行 CHR 事后缩放，M2 柔顺接触（`solref` + 非线性 `solimp`），M3 3D flex 护垫。

## 数据（只读，不要修改）

| 数据 | 路径 | 内容 |
|---|---|---|
| 落锤实验（主标定目标） | `/home/wang22/engineai/PressureCalculation/force/data1.csv` | 冲头 5.487 kg；厚度 6/12/18/24 mm；密度 0.3/0.4；高度 0.1/0.4/0.8/1.0 m；`Fpk` 为三次测量平均峰值 (kN) |
| 裸地冲击脉冲 | `engineai_ros2_workspace/scripts/FreeBallTest/non.csv` | 只能作形状参考：主脉冲 ≈0.27 ms；冲量闭合不了（16.5 N·s < m·v），加速度计 -500 g 饱和 |
| 裸地峰值目标 | 由 `force_fit.py` 定义 | `F = m v / 0.26554 ms`，与 CHR 拟合中的"缓冲前力"一致 |
| CHR 公式参数 | `engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters*.json` | M1 现行用 |
| 旧 M2 参数 | `engineai_ros2_workspace/scripts/FreeBallTest/iron_ball_drop.xml` 注释 | 用户确认不准 |

## 已知事实（改动前先读）

- 机器人仿真加载 `pm_v2.xml` → `serial_pm_v2_new.xml`，**timestep = 2 ms**，Euler，pyramidal，Newton；仿真器链接 MuJoCo **3.2.3**（`/opt/engineai_robotics_third_party`）。
- 2 ms 下一步内能传递的力上限约为 Δp/dt，裸地冲击被低估约 6.7 倍。所以任何"按 2 ms 仿真力"标定的方法都要单独说明 timestep 依赖。
- MuJoCo 软约束的力与接触的有效质量成正比，冲头质量（5.487 kg）不等于机器人 link 有效质量，迁移到机器人时要验证。
- 机器人上 geom 与地面的接触参数会混合（solmix）。用 M2 时给护具 geom 设更高的 `priority`，让护具参数直接生效。

## 代码

| 文件 | 作用 |
|---|---|
| `drop.py` | 落锤仿真与实验数据读取 |
| `eval_old_m2.py` | 旧 M2 参数在各 timestep 下的误差 |
| `fit_m2.py` | M2 拟合：`--dt`；分组拟合（每厚度×密度一组），或 `--joint`（同密度共享材料参数，width ∝ 厚度） |
| `summarize_m2.py`、`plot_m2.py` | 汇总拟合结果、画图到 `runs/figs/` |
| `tools/run_robot.py` | 单次闭环摔倒：AMP 控制器 + 无界面仿真，推一次；`--method M0/M1 --dt --direction --force` |
| `tools/batch_robot.py` | 16 场景（8 方向 × 200/300 N）批量跑，结果在 `~/data/mujoco_logs/protector_research/<tag>/` |
| `score_robot.py` | 读 `contact_quality.csv`，按 body 输出穿透、峰值力、滑移距离、有效质量 |
| `tools/send_report.py` | 每小时邮件报告 |

ROS 仿真器改动在工作空间分支 `protector-sim`（只在本地提交）：设置环境变量 `PROTECTOR_QUALITY_CSV=<路径>` 后，每个物理步写一份接触质量 CSV；设置 `PROTECTOR_SIM_TIMESTEP=<秒>` 可覆盖 timestep。改完 C++ 后执行 `colcon build --packages-select mujoco_simulator`。

**注意：** install 下的 `pm_v2.yaml` 是指向 src 的软链接。`run_robot.py` 运行期间会临时修改它，结束时从备份恢复。批量运行期间不要提交这个文件；批量结束后用 `git diff` 确认它已恢复。同一时间只能运行一个仿真。

环境：`protector_research/.venv`（MuJoCo 3.2.3，与机器人仿真一致）。MuJoCo 4.0 的 `simulate` 在 `../build/bin/`；M3 需要 4.0 的 Python 绑定，从源码编译。
大的运行输出放 `runs/`（不进 git），图放 `runs/figs/`（邮件会附上新图）。

## 规则

1. **不改原始数据和 `/opt` 下的库。** ROS 工作空间的改动只放在新分支 `protector-sim`（从 `bench` 拉），不推到 NUS-HcRL 远程，除非用户同意。
2. 本仓库的提交推到 `origin`（`git@github.com:wczmiracle/mujoco.git`）分支 `protector-research`。
3. 每个结论都要有对应的数字：拟合误差、对比表或图。失败的尝试也要记录，并写明原因。
4. 判据冻结后不再改动（见 PLAN.md 第 2 节）；确实要改，必须在 RESULTS.md 里写清理由。
5. 过拟合检查：参数个数不能多于数据点个数的一半；优先使用联合模型；用留一高度（leave-one-height-out）验证。
6. 需要用户决策（如数据含义不明、要改真机配置）时，写进 RESULTS.md 的"待用户确认"，继续做不受影响的部分。

## 研究循环

```
读 RESULTS.md 最新状态 → 选下一个最有信息量的实验 → 跑（长任务放后台，日志进 runs/logs/）
→ 打分 / 画图 → 追加到 RESULTS.md → git commit + push → 下一轮
```

RESULTS.md 每条格式：

```
## YYYY-MM-DD HH:MM  <一句话标题>
- 做了什么：
- 结果：（表格或数字，图放 runs/figs/ 并写文件名）
- 结论：
- 下一步：
```

## 汇报

- 每小时运行 `tools/send_report.py --force`，把 RESULTS.md 的新增内容、新提交、正在跑的任务和新图发到 `~/.config/sample_plan_mail/env` 里的 `MAIL_TO`。
- 目前由 Claude Code 会话内定时任务触发（每小时第 7 分钟，会话关闭即停止，最长 7 天）。若要与会话无关地持续发送，由用户自行加入 crontab：
  `7 * * * * cd /home/wang22/engineai/mujoco/protector_research && .venv/bin/python tools/send_report.py --force >> runs/logs/mail.log 2>&1`
