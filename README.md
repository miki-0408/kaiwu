# kaiwu — 腾讯开悟多智能体对抗决策

**2025 腾讯开悟人工智能全球公开赛**（智能体决策赛道）参赛代码。
基于 **PyTorch** 实现 PPO 训练全流程，并在官方基线之上重新设计了**观测编码、策略网络聚合方式与奖励函数**三部分。

---

## 核心结果

**四川赛区一等奖（赛区前 10%）** ｜ 当届全球 530 余所高校、智能体决策赛道 1900 余支队伍。

三项技术改进（细节见下文）：

| 改进 | 基线做法 | 本实现 |
|---|---|---|
| **观测编码** | 725 维扁平向量拼接，网络无法区分实体归属 | 按**英雄 / 小兵 / 防御塔 × 敌我阵营**分组编码 + 目标嵌入 + LSTM 时序建模 |
| **策略网络** | 各类实体池化为同一向量，**丢失实体间关系** | 按实体类型分组的 **6 路并行多头注意力**聚合（4 头，LayerNorm 残差）+ 全连接层批归一化 |
| **奖励函数** | 2 项 | 扩展至 **11 项加权特征**（配置化权重字典），再**逐项消融**收敛到关键项 |

> ⚠️ **消融实验的量化记录已不可考**（当时的线上对战记录未留存），因此本文只描述方法与结构，
> 不给出性能提升数字。这一点在项目报告中已如实说明。

---

## ⚠️ 运行前提（先读这段）

本仓库依赖腾讯开悟平台的 SDK —— **`kaiwu_agent`** 与 **`kaiwudrl`**。
这两个包**不公开发布在 PyPI 上**，只随开悟平台的环境镜像提供。

因此：

- **本仓库无法脱离开悟平台环境直接运行**。`python train_test.py` 在普通机器上会因找不到 `kaiwudrl` 而失败。
- 仓库的价值在于**代码本身**：网络结构、特征工程与奖励设计三个模块是自包含的纯 PyTorch 实现，**不需要平台环境就能读懂**。
- 想复现训练，需要先申请开悟平台账号并按其文档准备环境。

> 另外：根目录的 `main.py` 是一份独立的 PPO 实现草稿（自带简陋的 actor/critic 与 `diy.config`），
> 与竞赛提交链路无关，保留仅作参考。**真正的入口是 `train_test.py`。**

---

## 目录结构

仓库内含**两个赛道版本**，均为 `agent_ppo` + `agent_diy` 双实现：

```
kaiwu/            # 版本一
kaiwurace2/       # 版本二（第二个赛道，相比版本一少了 algorithm/buff.py）
```

单版本内部结构（以 `kaiwu/` 为例）：

```
agent_ppo/                  # PPO 实现（主版本）
├── agent.py                # Agent 主体：与平台环境的交互接口、学习率调度
├── algorithm/
│   ├── algorithm.py        # PPO 损失与梯度更新（含 grad clip）
│   └── buff.py             # 经验回放 / 样本缓冲
├── conf/
│   ├── conf.py             # ★ 奖励权重、维度、超参集中配置
│   └── train_env_conf.toml # 平台训练环境配置
├── feature/
│   ├── definition.py       # 特征定义与样本数据结构
│   └── reward_process.py   # ★ 奖励计算（GameRewardManager）
├── model/
│   └── model.py            # ★ 网络结构（EntityAttention / MLP / LSTM）
└── workflow/
    └── train_workflow.py   # 训练工作流、模型保存
agent_diy/                  # 另一套自实现，结构同上
main.py                     # 独立 PPO 草稿（见上文说明）
train_test.py               # ★ 训练/测试入口（需平台环境）
kaiwu.json                  # 平台配置
```

★ = 本次改动集中在这几个文件。

---

## 1. 观测空间重构

原始观测是 725 维的扁平向量，所有实体的特征被拼接在一起，网络无法区分"这个数属于谁"。

改法：**按实体类型 + 敌我阵营拆分**，每一类实体单独编码。分组定义在 `agent_ppo/conf/conf.py` 的 `DimConfig`：

| 实体类别 | 维度配置 | 说明 |
|---|---|---|
| 敌方英雄 `DIM_OF_HERO_EMY` | `[235]` | 单个实体 235 维 |
| 我方英雄 `DIM_OF_HERO_FRD` | `[235]` | |
| 主控英雄 `DIM_OF_HERO_MAIN` | `[14]` | |
| 小兵 1–10 `DIM_OF_SOLDIER_1_10` | `[18,18,18,18]` | 4 个 × 18 维 |
| 小兵 11–20 `DIM_OF_SOLDIER_11_20` | `[18,18,18,18]` | 敌方小兵 |
| 防御塔 1–2 `DIM_OF_ORGAN_1_2` | `[18,18]` | 我方 |
| 防御塔 3–4 `DIM_OF_ORGAN_3_4` | `[18,18]` | 敌方 |
| 全局信息 `DIM_OF_GLOBAL_INFO` | `[25]` | 时间、比分等 |

同时为每个实体引入 **target embedding**（`TARGET_EMBED_DIM = 32`）：把实体的原始特征经 MLP 压到 32 维的"目标嵌入"，供后续注意力模块使用。

配置里 `SERI_VEC_SPLIT_SHAPE = [(725,), (85,)]` 即 725 维观测 + 85 维附加输入。

---

## 2. 策略网络：实体分组 + 多头注意力

**原实现的问题**：把英雄、小兵、防御塔等实体各自池化成一个向量再拼接。池化是置换不变的聚合，会**丢失实体之间的关系**——比如"我方正被两个敌方英雄夹击"这种信息在池化后就没了。

**改进**（`agent_ppo/model/model.py`）：

```python
class EntityAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, entities):
        # 以可学习的 global_query 作为 query，对同类实体做注意力聚合
        output = self.layer_norm(global_query + attn_output)   # 残差 + LayerNorm
```

- 对**英雄（敌/我）、小兵、防御塔**等类别**分别**做多头注意力聚合（4 头），保留类内实体间关系；
- 用 `LayerNorm` + 残差连接稳定训练；
- **全连接层引入批归一化**：`model.py` 中 `MLP` 类带 `use_bn` 开关，开启后每层后接 `nn.BatchNorm1d`；
- 实体编码后经 **LSTM 时序建模**（`LSTM_TIME_STEPS = 16`，`LSTM_UNIT_SIZE = 512`）；
- 输出 **6 个动作头**（`LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]`），多离散动作空间。

---

## 3. 奖励塑形与消融

奖励权重集中在 `agent_ppo/conf/conf.py` 的 `REWARD_WEIGHT_DICT`，共 **11 项**加权特征：

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

计算逻辑在 `agent_ppo/feature/reward_process.py`（`GameRewardManager`），每个奖励项是一个 `RewardStruct`，带权重与上下界裁剪。

**消融实验的方法论**：因为权重是**配置化的字典**，消融只需改一个键值、重跑训练，无需改代码。做法是：

1. 从基线 2 项逐步扩展到 11 项；
2. 观察发现**奖励设计过细反而不利于训练**——每一项都在引导策略，项数越多，稀疏的胜负信号越容易被淹没，容易学到"刷分"的次优行为；
3. **逐项消融**：每次去掉一项重跑，只保留对最终收益贡献最关键的少数项。

> 这段经验的价值在于：奖励设计不是"特征越多越好"，而是要控制引导信号的密度。
> 这与 RLHF 中 reward model 设计面对的问题同源——**模型会钻奖励函数的空子，而不是把事情真正做对**。

---

## 训练超参

集中在 `agent_ppo/conf/conf.py` 的 `Config`：

| 参数 | 值 | 说明 |
|---|---|---|
| `CLIP_PARAM` | 0.2 | PPO 裁剪系数 |
| `GAMMA` / `LAMDA` | 0.995 / 0.95 | 折扣因子 / GAE λ |
| `INIT_LEARNING_RATE_START` | 3e-3 | 初始学习率 |
| `TARGET_LR` / `TARGET_STEP` | 1e-4 / 5000 | 学习率退火目标与步数 |
| `BETA_START` | 0.025 | 熵正则系数 |
| `USE_GRAD_CLIP` / `GRAD_CLIP_RANGE` | True / 0.5 | 梯度裁剪 |
| `LSTM_TIME_STEPS` / `LSTM_UNIT_SIZE` | 16 / 512 | 时序建模 |
| `TARGET_EMBED_DIM` | 32 | 实体目标嵌入维度 |
| `MODEL_SAVE_INTERVAL` | 1800 | 模型保存间隔 |

---

## 环境与依赖

- **PyTorch**：网络结构与训练循环
- **腾讯开悟平台**：`kaiwu_agent` / `kaiwudrl`（环境交互、分布式训练与模型同步）
- **NumPy**：特征工程与观测处理

---

## 已知限制

- **无法脱离平台运行**：依赖开悟平台 SDK，本仓库不含该 SDK，也不含环境模拟器。
- **缺量化实验记录**：消融实验的性能数据未留存（见开头说明），因此无法给出"改进前后"的数值对照。
- **含 IDE 配置**：仓库历史中跟踪了 `.idea/` 与 `.vscode/`，`.gitignore` 已添加但对已跟踪文件无效。
- **双赛道版本并存**：`kaiwu/` 与 `kaiwurace2/` 是为两个赛道分别提交的版本，有大量重复代码，未做合并整理。

---

## 致谢

基线框架与 `train_test.py`、`workflow/`、`feature/definition.py` 等平台对接代码来自腾讯开悟官方示例；
本次提交的改动集中在**网络结构、特征分组与奖励设计**三部分。
