# climbrobo 项目说明

## 项目目标

本项目使用 PyBullet 开展八轴双端吸附爬塔机器人的运动学、碰撞检测和轨迹规划研究。

当前规划阶段已经从单步诊断转入任务驱动机构设计；已有单步结果仍作为基线和回归数据。当前目标是比较对称6R与对称8R，并最终验证整塔连续爬行：

1. 一个吸盘固定在铁塔附着点；
2. 另一个吸盘运动到最远可达的候选附着点；
3. 运动全过程满足关节限位和碰撞约束；
4. 在 PyBullet GUI 中显示机器人、铁塔、吸盘坐标系和运动轨迹。

机构设计阶段的额外约束：

* 只研究`3R + central body + mirrored 3R`和`4R + central body + mirrored 4R`；7R不是本项目候选；
* 中央本体必须保留真实`L4.STL`mesh，Tower必须使用真实`Tower.STL`碰撞模型；
* 新连杆可以在没有CAD时使用有半径的Capsule代理，不能使用零半径线段；
* 设计评价以真实attach-line任务和连续contact-state graph为目标，不以workspace体积为最终目标；

## 目录说明

* `robots_model/`

  * 机器人权威模型目录；
  * 包含 URDF、STL meshes 和两个吸盘局部坐标系的 YAML 配置；
  * 不得修改其中的 URDF、STL 或 YAML 几何参数。
* `models/`

  * 包含铁塔模型、铁塔附着面模型、候选附着点及其坐标数据；
  * 先检查 CSV、NPZ 和 STL 文件的内容、字段、单位及坐标系，再决定读取方式；
  * 不得覆盖或重新生成原始数据。
* `legacy_scripts/`

  * 包含早期测试和试验脚本；
  * 仅用于理解已有做法和复用经过验证的代码片段；
  * 不得将其中代码直接视为正确实现；
  * 不要修改这些历史脚本。
* `docs/`

  * 存放规划任务说明、坐标系约定和算法设计文档。

## 机器人运动链

URDF运动链为：

`base_link -> J1 -> L1 -> J2 -> L2 -> J3 -> L3 -> J4 -> L4 -> J5 -> L5 -> J6 -> L6 -> J7 -> L7 -> J8 -> L8`

* `base_link`端为一个吸附端；
* `L8`端为另一个吸附端；
* PyBullet中 `base_link` 的 link index 为 `-1`；
* 其他 link index 必须根据 URDF link 名称动态查询，不得硬编码。

## 吸盘坐标系

两个吸盘中心、法向和面内参考方向统一从：

`robots_model/config/suction_frames.yaml`

读取。

吸盘功能坐标系的构造约定为：

* 原点：YAML中的 `position`；
* Z轴：归一化后的 `normal`；
* Y轴：由 `y_reference` 投影并正交化后获得；
* X轴：按照右手坐标系计算；
* 不得在不同脚本中重复硬编码吸盘位置和法向。
* 总共存在两个附着面：`surface1`和`surface2`。`base_end`与`l8_end`是两个独立的物理磁吸端，任意一个端都可以附着任意一个表面；是否可行由目标法向、运动学、碰撞和轨迹约束共同决定。旧数据文件名`foot1/foot2`仅作为`surface1/surface2`的兼容命名，不再表示物理端绑定。
* 吸盘Z轴是由吸附面向外的，附着点法向是由附着面向外的，吸附的时候两者是反向的。

## 编程要求

* 使用 Python、NumPy 和 PyBullet；
* 路径使用 `pathlib.Path` 管理；
* 传入 PyBullet 文件接口前将 `Path` 转换为 `str`；
* 所有长度单位统一为米，角度计算统一使用弧度；
* 坐标变换应明确写出源坐标系和目标坐标系；
* 优先编写短小、可验证的函数；
* 对配置字段、文件不存在、数组形状错误和零法向进行显式检查；
* 不得静默猜测数据字段或坐标系；
* 不得修改已经验证的模型文件来掩盖程序错误。

## 开发流程

执行新任务时：

1. 先阅读 `AGENTS.md`、任务说明、`pyproject.toml`、相关配置和数据文件；
2. 先总结所理解的目录、数据格式、坐标系和运行入口；
3. 在实施前列出拟新建或修改的文件；
4. 尽量新建脚本或模块，避免破坏已经验证的代码；
5. 修改后运行语法检查和能够执行的测试；
6. 检查 `git diff`；
7. 最终说明修改内容、运行命令、验证结果和未解决问题。

当规划体系、运动学假设、附着策略、重要验证结果或关键文件结构发生实质变化时，应同步更新`docs/planning_history.md`。只记录重要变化，不记录琐碎调试过程。

## Git要求

* 开始修改前检查 `git status`；
* 不修改或删除用户已有的未提交工作；
* 不自动提交或推送；
* 不使用破坏性Git命令；
* 完成后由用户审核并自行提交。

## Autonomous morphology research

When working on robot morphology / whole-tower climbing research:

1. Read `/AUTONOMOUS_MORPHOLOGY_RESEARCH.md` before making design or planning decisions.
2. Read and update `/docs/morphology_research_status.md`.
3. Resume from existing checkpoints instead of repeating completed expensive experiments.
4. Important research results must also be reflected in `/docs/planning_history.md`.
5. Do not declare morphology infeasibility from a single IK, RRT, contact route, or optimizer failure.
