# kaiwu

2025 腾讯开悟人工智能全球公开赛（智能体决策赛道）参赛代码。

用 PyTorch 实现 PPO 训练流程，在官方基线代码的基础上改了观测编码、策略网络和奖励函数三部分。
成绩：四川赛区一等奖（赛区前 10%）。

---

## 运行前提

代码依赖开悟平台提供的两个包：`kaiwu_agent` 和 `kaiwudrl`。
这两个包不在 PyPI 上发布，只随平台的环境镜像提供，所以**本仓库不能脱离平台环境运行**——
在普通机器上直接执行 `python train_test.py` 会因为找不到 `kaiwudrl` 报错。

网络结构、特征工程、奖励计算这三部分是纯 PyTorch 实现，不依赖平台，可以直接阅读。

## 依赖

- Python 3.7+
- PyTorch
- NumPy
- 腾讯开悟平台 SDK（`kaiwu_agent`、`kaiwudrl`）

## 运行

在开悟平台环境中，先修改 `train_test.py` 顶部的 `algorithm_name` 指定用哪套实现：

```python
algorithm_name = "ppo"   # 可选 "ppo" 或 "diy"
```

然后：

```bash
python train_test.py
```

---

## 文件说明

### 顶层

| 文件 | 说明 |
| --- | --- |
| `train_test.py` | 训练/测试入口。调用 `kaiwudrl` 的 learner 和 aisrv 启动平台训练流程 |
| `main.py` | 一份独立的 PPO 实现草稿，与竞赛提交链路无关，保留作参考 |
| `kaiwu.json` | 平台配置 |

### `agent_ppo/` — PPO 实现

| 文件 | 说明 |
| --- | --- |
| `agent.py` | Agent 主体。实现平台要求的接口，负责观测解析、动作采样、回报计算；含学习率调度（3e-3 退火到 1e-4） |
| `model/model.py` | 网络结构。`EntityAttention` 对同类实体做多头注意力（4 头）+ LayerNorm 残差；`MLP` 是带可选 BatchNorm 的全连接模块；`Model` 把特征编码、实体注意力、LSTM 和 6 个动作头组装起来 |
| `feature/definition.py` | 特征定义。观测中每个字段的含义和维度 |
| `feature/reward_process.py` | 奖励计算。`GameRewardManager` 按 `conf.py` 的权重字典逐项计算，每项带上下界裁剪 |
| `algorithm/algorithm.py` | PPO 损失和参数更新，含梯度裁剪 |
| `algorithm/buff.py` | 样本缓冲，存放 rollout 数据 |
| `conf/conf.py` | 所有配置：奖励权重、特征维度、训练超参 |
| `conf/train_env_conf.toml` | 平台训练环境配置 |
| `workflow/train_workflow.py` | 训练工作流和模型保存 |

### `agent_diy/`

备选实现，结构同 `agent_ppo/`。

### `kaiwurace2/`

第二个赛道的版本，结构同 `kaiwu/`，差别是少了 `algorithm/buff.py`。

---

## 三处主要改动

如果想和官方基线对比，改动集中在下面三处。

### 观测编码 —— `conf/conf.py` 的 `DimConfig`

基线把 725 维观测拉平成一个向量，所有实体混在一起。改成按实体类型和敌我阵营分组：

```
敌方英雄      DIM_OF_HERO_EMY        [235]
我方英雄      DIM_OF_HERO_FRD        [235]
主控英雄      DIM_OF_HERO_MAIN       [14]
小兵 1-10     DIM_OF_SOLDIER_1_10    [18, 18, 18, 18]
小兵 11-20    DIM_OF_SOLDIER_11_20   [18, 18, 18, 18]
防御塔 1-2    DIM_OF_ORGAN_1_2       [18, 18]
防御塔 3-4    DIM_OF_ORGAN_3_4       [18, 18]
全局信息      DIM_OF_GLOBAL_INFO     [25]
```

每个实体再经过 MLP 压成 32 维的 target embedding（`TARGET_EMBED_DIM`），供注意力模块使用。

### 策略网络 —— `model/model.py`

基线对各类实体做池化，压成同一个向量。池化是置换不变的聚合，实体之间的相对关系就丢了——
比如"我方正被两个敌方英雄夹击"这种信息，池化之后看不出来。

改成对每个实体类别分别做多头注意力，用可学习的 query 去聚合，保留类内实体间的关系。
另外给全连接层加了 BatchNorm（`MLP` 类的 `use_bn` 参数）。

### 奖励函数 —— `conf/conf.py` 的 `REWARD_WEIGHT_DICT`

从基线的 2 项扩展到 11 项：

```python
REWARD_WEIGHT_DICT = {
    "hp_point": 3.0,          # 血量
    "tower_hp_point": 6.0,    # 防御塔血量
    "money": 0.008,           # 经济
    "exp": 0.006,             # 经验
    "ep_rate": 0.75,          # 能量
    "death": -2.0,            # 阵亡
    "kill": -0.6,             # 击杀
    "last_hit": 0.6,          # 补刀
    "forward": 0.02,          # 推进
    "in_attack_range": 0.01,  # 处于攻击范围
    "hit_hero_times": 0.01,   # 命中英雄次数
}
```

实际训练中发现奖励项太多反而不稳定，逐项消融后只保留了其中几项。
因为权重是配置化的字典，消融只需要改这个字典重跑，不用动代码。

> 消融阶段的线上对战记录没有留存，所以这里给不出各项的具体影响数据。

---

## 训练超参

配置在 `agent_ppo/conf/conf.py` 的 `Config` 类里。

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `CLIP_PARAM` | 0.2 | PPO 裁剪系数 |
| `GAMMA` | 0.995 | 折扣因子 |
| `LAMDA` | 0.95 | GAE λ |
| `INIT_LEARNING_RATE_START` | 3e-3 | 初始学习率 |
| `TARGET_LR` / `TARGET_STEP` | 1e-4 / 5000 | 学习率退火目标与步数 |
| `BETA_START` | 0.025 | 熵正则系数 |
| `USE_GRAD_CLIP` / `GRAD_CLIP_RANGE` | True / 0.5 | 梯度裁剪 |
| `LSTM_TIME_STEPS` / `LSTM_UNIT_SIZE` | 16 / 512 | LSTM 时序建模 |
| `TARGET_EMBED_DIM` | 32 | 实体目标嵌入维度 |
| `MODEL_SAVE_INTERVAL` | 1800 | 模型保存间隔 |

---

## 致谢

基线框架和 `train_test.py`、`workflow/`、`feature/definition.py` 等平台对接代码来自腾讯开悟官方示例。
改动集中在网络结构、特征分组和奖励函数三部分。
