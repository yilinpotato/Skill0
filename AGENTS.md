# Repository Guidelines


## 实验主线与计算资源约定（2026-10-26）

- 三方法按统一实验标准比较，方法机制保留差异；CoSkill no-RL 与 Tree-RL 分为两个独立对照组。公共配置和实验记录由 `yilinpotato/train-alfworld-data` 的 `comparison/` 管理。
- 正式仓库映射：Skill0 使用 `yilinpotato/Skill0` 的 `main`；SkillRL 使用 `yilinpotato/SkillRL` 的 `main`；CoSkill 使用同一个 `yilinpotato/SkillRL` 仓库的 `Co-Skill` 分支。独立的 `yilinpotato/CoSkill` 仓库不是本轮实验主线的提交目标。
- 大规模 SFT、RL、批量推理及高计算量实验安排到用户后续提供的超算，不在共享 3090 上启动。超算环境由用户预先配置；收到登录方式后先核验已有环境、数据、模型和调度器资源，再做小规模测试，禁止擅自重装或改变实验设置来适配资源。
- 本地/3090 可做必要的代码与轻量 CPU 检查；不得影响其他用户进程。GPU 作业继承调度器分配，不擅自固定物理卡号。不得沿用历史超算登录信息来猜测本次目标机器。
- 提交只包含源码、配置、说明与经过筛选的实验指标。大模型权重继续使用模型存储，凭据不进 Git。完整轨迹按既有实验数据归档规范保存。
