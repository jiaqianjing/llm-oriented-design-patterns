# LLM 导向设计模式（LLM-Oriented Design Patterns）

**副标题：** 放弃人类可读性，重铸齿轮法则、反馈回路与工具化的新秩序

---

## 目录

- [序言：代码读者的物种级跃迁](#prologue)
- [第一部分：上下文管理——从单体到微齿轮](#part-i-context-management)
  - [第一章：极致拆分](#chapter-1-radical-fragmentation)
  - [第二章：动态组装与上下文涌现](#chapter-2-dynamic-assembly)
  - [第三章：黑箱化与语义压缩](#chapter-3-black-boxing)
- [第二部分：多维反馈回路](#part-ii-feedback-loops)
  - [第四章：自顶向下的目标分解](#chapter-4-goal-decomposition)
  - [第五章：进化式齿轮编排](#chapter-5-gear-orchestration)
  - [第六章：多层反馈网络](#chapter-6-feedback-network)
- [第三部分：工具化——剥离概率性，锚定确定性边界](#part-iii-toolification)
  - [第七章：概率性与确定性](#chapter-7-probabilistic-vs-deterministic)
  - [第八章：基础设施即 LLM 工具](#chapter-8-infrastructure-as-tools)
  - [第九章：零幻觉契约](#chapter-9-zero-hallucination-contracts)
- [尾声：从代码编写者到系统牧羊人](#epilogue)
- [附录：完整的前后文件对照表](#appendix)

---

## 序言

### 代码读者的物种级跃迁

在不远的将来，逐行阅读源代码将成为前工业时代的遗物。标准的开发工作流将变成：

> 把 GitHub 仓库链接丢给大模型 → 描述业务意图 → 让 AI 自主理解并集成。

代码的**第一读者**正式从碳基人类转变为硅基模型。

### 传统设计的"上下文毒药"

面向对象编程、SOLID 原则和微服务架构都是围绕**人类认知带宽**和短期记忆来设计的。它们强调高内聚、深继承和人类可读的命名。但在大模型看来，这些反而制造了：

- **臃肿的模块** —— 超出模型最优智能窗口的类
- **深层依赖链** —— 必须加载 5 个以上文件才能理解一个方法
- **隐形间接层** —— 工厂模式、注册表模式和策略模式遮蔽了真正的调用图

这些问题具有普遍性。它们出现在 Web 框架、数据管道、ML 训练系统、CLI 工具和企业后端中。任何拥有超过 500 行代码（LOC）的类、深层继承或工厂套工厂模式的代码库，都对 LLM 的理解构成阻碍。

### "长上下文智能坍塌"

即使拥有 100 万 token 的上下文窗口，**"Lost in the Middle"** 效应依然存在。过于冗长、对人类友好的代码会稀释模型的注意力，导致推理能力下降和幻觉增多。

### 新架构宣言

> **软件设计的第一原则必须转变——从人类友好到 LLM 友好。**

我们必须牺牲直觉上的人类可读性，以最大化模型的解析、组装和执行效率。

### 案例研究：oxRL

我们以 ([https://github.com/warlockee/oxRL](https://github.com/warlockee/oxRL)) 作为案例来演示这些原则。oxRL 是一个 LLM 后训练框架，实现了 RL 和 SL 路径下的 18 种算法。它呈现出任何成熟代码库中常见的架构债务：上帝类、单体入口点、重复的工具函数和耦合的关注点。在其 [v1.0 版本](https://github.com/warlockee/oxRL/tree/LOD-example-v1.0)中，它已经做出了一些 LLM 友好的选择——但仍然背负着大量技术债，影响了 AI 的理解能力。

在本书中，我们展示 [v1.0 版本](https://github.com/warlockee/oxRL/tree/LOD-example-v1.0)作为"改造前"，并提供 [v2.0 版本](https://github.com/warlockee/oxRL/tree/LOD-example-v2.0)作为每个原则的"改造后"示例。这些模式本身适用于任何代码库——oxRL 只是使它们变得具象化。

---

## 第一部分：上下文管理

**论点：** 设计必须在 AI 的注意力和算力限制内起舞。上下文压缩是最高优先级。

### 黄金约束

> 每个微齿轮的代码及其依赖必须严格控制在模型的"最优智能窗口"内——**4k 到 8k tokens**（约 200-800 LOC）——以确保完全无损的理解。

这不是 RL 专属的规则。它适用于任何系统中的任何模块：React 组件、数据库迁移工具、API 处理器或训练算法。

---

### 第一章：极致拆分

> *将单体类和深层调用链打碎为绝对单一职责的扁平微齿轮。*

核心问题：一个承担 N 项职责的类迫使 LLM 将所有 N 项职责加载到上下文中，即使它只需要修改其中一项。拆分之后，修改第 3 项职责只需要阅读包含第 3 项职责的那个文件。

#### 违规 1：上帝类

任何将编排、I/O、数学运算和状态管理混合在一个超过 500 LOC 的文件中的类都是上帝类。LLM 必须在无关的代码中艰难跋涉，才能找到它真正需要的部分。

**oxRL 示例 — `oxrl/algs/grpo.py` — 665 LOC，7 项职责：**

```python
@ray.remote
class GRPO(BaseAlgorithm):
    def __init__(self, ...):              # 80 LOC — 存储 15+ 属性
        self.init_training_engine()       # 调用 load_model + LoRA + DeepSpeed + optimizer

    def init_training_engine(self):       # 72 LOC — 职责：引擎初始化
        deepspeed.init_distributed()
        model, ref_model = self.load_model()
        # LoRA 应用 (20 LOC)
        # 优化器构建 (10 LOC)
        # DeepSpeed 初始化 (15 LOC)
        # 参考模型 DeepSpeed 初始化 (10 LOC)

    def load_model(self):                 # 8 LOC — 职责：模型 I/O

    def ref_forward(self, ...):           # 25 LOC — 职责：参考模型推理

    def policy_forward(self, ...):        # 43 LOC — 职责：策略推理

    def compute_kl_distance(self, ...):   # 12 LOC — 职责：KL 散度计算

    def compute_policy_loss(self, ...):   # 106 LOC — 职责：损失函数（3 种变体！）

    def train_step(self, ...):            # 72 LOC — 职责：训练编排

    def _get_base_model_config(self):     # 11 LOC — 职责：配置提取

    def _strip_lora_and_merge(self, ...): # 42 LOC — 职责：权重合并

    def gather_state_dict(self):          # 30 LOC — 职责：分布式收集

    def save_checkpoint(self, ...):       # 98 LOC — 职责：检查点 I/O
```

**LLM 看到的情况：** 为了理解 GRPO 如何计算损失，模型必须将 665 LOC（约 2,700 tokens）加载到上下文中。但它只需要 `compute_policy_loss`（106 LOC）。其余 559 LOC 都是稀释注意力的噪音。

**更糟的是：** `compute_policy_loss` 方法在 if-else 分支后包含**三种完全不同的算法**（SGRPO、GSPO、CISPO）。LLM 在修改其中一个变体时，必须同时在注意力中保持全部三个变体。

**改造后（v2.0）—— 拆分为 6 个微齿轮：**

```
oxrl/algs/
├── base.py                    # 41 LOC — 接口契约（未改动）
├── losses/
│   ├── sgrpo_loss.py          # ~40 LOC — 变体 A
│   ├── gspo_loss.py           # ~35 LOC — 变体 B
│   └── cispo_loss.py          # ~30 LOC — 变体 C
├── grpo.py                    # ~250 LOC — 仅做编排
│   - __init__: 存储参数，调用 setup_engine()
│   - train_step: 数据准备 → 前向传播 → loss_fn() → 反向传播
│   - policy_forward / ref_forward: 模型调用
├── engine_setup.py            # ~100 LOC — 引擎初始化
│   - setup_training_engine(model_path, ds_config, lora_config) → engine
│   - setup_ref_engine(ref_model_path, ds_ref_config) → engine
├── weight_tools.py            # ~80 LOC — 检查点 + 权重合并
│   - strip_lora_and_merge(state_dict, lora_config) → state_dict
│   - gather_state_dict(engine) → state_dict
│   - save_checkpoint(state_dict, output_dir) → None
└── kl.py                      # ~15 LOC — 散度计算
    - compute_kl_distance(logprobs, ref_logprobs) → tensor
```

**为什么这对 LLM 更好：**

| 改造前（v1.0） | 改造后（v2.0） | 对 LLM 的影响 |
|---|---|---|
| 修改变体 B → 读 665 LOC | 修改变体 B → 读 35 LOC | **减少 19 倍上下文** |
| 添加新变体 → 编辑 106 LOC 的方法 | 添加新变体 → 创建 35 LOC 的新文件 | **零风险破坏现有变体** |
| 调试检查点保存 → 读 665 LOC | 调试检查点保存 → 读 80 LOC | **减少 8 倍上下文** |
| 权重合并耦合在一个类中 | 权重合并是独立工具 | **可被任何算法复用** |

**具体的 v2.0 提取变体 — `gspo_loss.py`：**

```python
# oxrl/algs/losses/gspo_loss.py — 35 LOC，单一职责
import torch

def gspo_loss(logprobs, old_logprobs, advantages, mask, clip_low, clip_high):
    """面向 MoE 模型的序列级裁剪代理损失。

    在裁剪之前对序列内的对数比率取平均，使得
    token 级 MoE 路由噪声可以相互抵消。

    Args:
        logprobs:      [B, T-1] 当前策略的对数概率
        old_logprobs:  [B, T-1] 旧策略的对数概率
        advantages:    [B, T-1] 组归一化优势值
        mask:          [B, T-1] 响应 token 掩码
        clip_low:      float，裁剪下界（如 0.2）
        clip_high:     float，裁剪上界（如 0.2）

    Returns:
        loss:     标量损失
        metrics:  包含 clipfrac、approx_kl 的字典
    """
    adv = advantages.detach().to(torch.float32)
    mask_f = (mask > 0.5).to(logprobs.dtype)

    logratio = (logprobs - old_logprobs).to(torch.float32)
    seq_lens = mask_f.sum(dim=-1).clamp(min=1.0)
    seq_logratio = (logratio * mask_f).sum(dim=-1) / seq_lens
    seq_ratio = torch.exp(seq_logratio)
    seq_adv = (adv * mask_f).sum(dim=-1) / seq_lens

    unclipped = seq_ratio * seq_adv
    clipped = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high) * seq_adv
    loss = -torch.minimum(unclipped, clipped).mean()

    with torch.no_grad():
        clipfrac = ((seq_ratio > 1.0 + clip_high) | (seq_ratio < 1.0 - clip_low)).float().mean()
        approx_kl = (seq_logratio + torch.exp(-seq_logratio) - 1.0).mean()

    return loss, {"clipfrac": clipfrac.item(), "approx_kl": approx_kl.item()}
```

35 LOC。约 140 tokens。LLM 可以在一次注意力跨度内阅读、理解、修改并验证其正确性。

---

#### 违规 2：单体入口点

任何将初始化、处理、指标收集、状态持久化和清理混合在一个文件中的主脚本，对 LLM 都是不友好的。模型必须在上下文中同时保持 8 个以上的关注点，才能修改其中任何一个。

**oxRL 示例 — `main_rl.py` — 595 LOC，8 个以上关注点集中在一个脚本中：**

```python
# main_rl.py — 一个文件做所有事情
def setup_ray(ray_address): ...              # 20 LOC — 基础设施初始化
def training_engine_setup(params, ...): ...  # 56 LOC — 训练 Actor 创建
def rollout_engine_setup(params, ...): ...   # 52 LOC — 推理 Actor 创建
def rollout_dataloader_setup(params, ...):   # 25 LOC — 数据加载
def collect_rollouts(...): ...               # 71 LOC — 生成编排
def main(config_file, experiment_id):        # 330 LOC — 处理循环 + 持久化 + 刷新
    # 所有逻辑交织在一起：
    # - 配置加载
    # - 基础设施初始化
    # - 引擎创建
    # - 分词器加载
    # - 评分函数导入
    # - 数据生成
    # - 缓冲区管理
    # - 训练步骤调度
    # - 指标聚合和日志
    # - 状态保存（2 条代码路径！）
    # - 引擎刷新（2 条代码路径！）
    # - 实验追踪
```

**改造后（v2.0）—— 拆分为专注的微齿轮：**

```
main_rl.py                    # ~120 LOC — 纯编排
                              #   main() 调用: setup → loop → cleanup

oxrl/setup/
├── ray_setup.py              # ~30 LOC — 基础设施初始化
├── engine_factory.py         # ~60 LOC — 创建训练 Actor
└── dataloader_factory.py     # ~30 LOC — 创建数据管道

oxrl/loops/
├── rollout_phase.py          # ~60 LOC — 数据生成阶段
├── train_phase.py            # ~50 LOC — 调度训练，聚合指标
└── checkpoint_phase.py       # ~80 LOC — 收集 → 保存 → 刷新（两条路径）
```

**v2.0 `main_rl.py` — 纯编排，约 120 LOC：**

```python
# main_rl.py v2.0 — 编排器读起来像一份菜谱
from oxrl.setup.ray_setup import setup_ray
from oxrl.setup.engine_factory import create_training_engines, create_rollout_engines
from oxrl.setup.dataloader_factory import create_rollout_dataloader
from oxrl.loops.rollout_phase import collect_rollouts
from oxrl.loops.train_phase import run_training_steps
from oxrl.loops.checkpoint_phase import save_and_refresh

def main(config_file, experiment_id):
    config = cfg.load_and_verify(method="rl", input_yaml=config_file, ...)
    ray_engine, master_addr = setup_ray(config.run.ray_address)

    training_engines = create_training_engines(config, master_addr)
    rollout_engines = create_rollout_engines(config, reward_fnc, eos_id)
    rollout_dataloader = create_rollout_dataloader(config, tokenizer, len(rollout_engines))
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id, max_seq_len=config.data.max_seq_len)

    for epoch in range(config.train.total_number_of_epochs):
        # 阶段 1：生成数据
        rollout_stats = collect_rollouts(rollout_dataloader, rollout_engines, epoch, policy_version, replay_buffer)

        # 阶段 2：处理数据
        epoch_metrics = run_training_steps(training_engines, replay_buffer, config, epoch)

        # 阶段 3：保存状态 + 刷新引擎
        policy_version = save_and_refresh(training_engines, rollout_engines, config, epoch, policy_version, tokenizer)

        # 重置准备下一轮迭代
        replay_buffer.reset()
```

LLM 阅读这个文件时，可以在 40 LOC 内看到**整个管道**。要修改任何阶段，只需阅读相关的 50-80 LOC 文件。不会在无关的阶段上浪费注意力。

---

#### 违规 3：多关注点服务

任何将生命周期管理、参数构建、数据解析、核心处理和后处理混合在一个文件中的服务类，都是多关注点服务。每个关注点都应当可以提取为纯函数或专注的模块。

**oxRL 示例 — `oxrl/rollouts/vllm_engine.py` — 565 LOC，5 项职责：**

```python
@ray.remote
class VLLMRolloutEngine:
    # 关注点 1: 模型生命周期（加载、刷新、从状态字典刷新）     ~140 LOC
    # 关注点 2: 采样参数构建 + 校验                           ~50 LOC
    # 关注点 3: 输出解析（对数概率提取）                       ~40 LOC
    # 关注点 4: 核心处理（生成 + 样本构建）                    ~120 LOC
    # 关注点 5: 后处理（评分 + 归一化）                       ~45 LOC
```

**改造后（v2.0）—— 提取纯函数，保持服务层精简：**

```
oxrl/rollouts/
├── vllm_engine.py             # ~200 LOC — 服务：生命周期 + 核心处理
├── sampling.py                # ~50 LOC  — make_sampling_params() + 校验
├── logprob_utils.py           # ~40 LOC  — extract_logprobs() 纯函数
├── reward_scoring.py          # ~30 LOC  — score_response() 纯函数
├── normalization.py           # ~40 LOC  — normalize_rewards() 纯函数
└── replay_buffer.py           # ~208 LOC — 未改动（已经足够专注）
```

**关键提取 —— 后处理作为纯函数：**

```python
# 改造前：嵌入在 VLLMRolloutEngine 类中（v1.0）
class VLLMRolloutEngine:
    def normalize_rewards(self, samples, stats, prompt_len, is_per_token):
        # 使用 self.eps_reward_norm 和 self.reward_broadcast
        # 就地修改 samples
        ...

# 改造后：独立纯函数（v2.0）
# oxrl/rollouts/normalization.py
def normalize_rewards(samples, stats, prompt_len, is_per_token, eps=1e-8, broadcast=False):
    """对一组样本内的奖励进行 Z-score 归一化。

    Args:
        samples:      一个输入对应的结果字典列表
        stats:        {"rewards": [float], "lengths": [int]}
        prompt_len:   int，输入前缀的长度
        is_per_token: bool，评分是否为逐 token 的
        eps:          float，归一化 epsilon
        broadcast:    bool，是否将标量评分广播到所有 token

    Returns:
        None（就地修改 samples）
    """
    ...
```

现在 LLM 可以单独测试归一化逻辑，在不同的服务中复用它，或者替换它——而无需触碰服务类。

---

#### 违规 4：Schema + 逻辑 + I/O 耦合

任何将数据定义（schema）、推导逻辑（同步/计算）和文件 I/O（加载）混合在一个模块中的配置文件，都会迫使 LLM 无论需要哪个关注点都必须阅读全部内容。

**oxRL 示例 — `oxrl/configs/load.py` — 483 LOC，3 个不同的关注点：**

```python
# 关注点 1: 配置 SCHEMA（Pydantic 模型）— ~275 LOC
class Run(BaseModel): ...
class Train(BaseModel): ...
class Model(BaseModel): ...
class Data(BaseModel): ...
class DeepSpeed(BaseModel): ...
class DeepSpeedRef(BaseModel): ...
class InferenceEngine(BaseModel): ...
class Lora(BaseModel): ...
class Reward(BaseModel): ...
class Rollout(BaseModel): ...
class Config(BaseModel): ...

# 关注点 2: 配置逻辑（同步 + 推导）— ~150 LOC
class Config(BaseModel):
    def sync_deepspeed_config(self, world_size):
        self._sync_batch_sizes(world_size)
        self._sync_gradient_clipping()
        self._sync_dtype()
        self._sync_optimizer()
        self._sync_scheduler()
        self._sync_zero_defaults()
        self._sync_ref_model_config()

# 关注点 3: 配置加载（文件 I/O）— ~50 LOC
def load_and_verify(method, input_yaml, experiment_id, world_size=None): ...
```

**改造后（v2.0）—— 将 Schema 与逻辑分离：**

```
oxrl/configs/
├── schema.py                 # ~200 LOC — 纯 Pydantic 模型，零逻辑
│   class Run, Train, Model, Data, DeepSpeed, Lora, Reward, Rollout, Config
│
├── sync.py                   # ~150 LOC — 配置同步/推导
│   def sync_deepspeed_config(config, world_size) → None
│   def sync_batch_sizes(config, world_size) → None
│   def sync_dtype(config) → None
│   ...
│
└── loader.py                 # ~50 LOC — YAML 加载 + 校验
    def load_and_verify(method, input_yaml, experiment_id, world_size) → Config
```

**为什么：** LLM 添加新配置字段时只需要 `schema.py`（200 LOC）。LLM 调试同步逻辑时只需要 `sync.py`（150 LOC）。而现在无论做什么都必须阅读全部 483 LOC。

---

### 第二章：动态组装

> *人类停止编写僵化的粘合代码。模型获得动态组装的权力。*

原则：使用扁平调度（字典、importlib）代替僵化的硬连线或工厂模式。当所有选项在一个字典中一目了然时，LLM 可以立即知道有哪些可用选项，无需追踪间接层。

#### v1.0 做对的地方

oxRL v1.0 已经使用普通字典进行算法调度，而不是工厂模式：

```python
# main_rl.py — 好的做法：显式字典，零间接层
RL_ALGORITHMS = {"sgrpo": GRPO, "cispo": GRPO, "gspo": GRPO, "rlhf": GRPO, "rlaif": GRPO, "ppo": PPO}
alg = RL_ALGORITHMS[alg_name]
```

以及通过配置动态加载函数：

```python
reward_module = importlib.import_module("oxrl.rewards")
reward_fnc = getattr(reward_module, config.reward.reward_func)
```

这些已经是 LLM 原生的做法。**保持不变。**

#### v1.0 做错的地方：僵化的引擎硬连线

**改造前（v1.0）—— 调用方硬编码参数提取：**

```python
# main_rl.py — training_engine_setup() 手动构建 kwargs
def training_engine_setup(params, alg, world_size, master_addr, master_port):
    kwargs = {
        'model_path': params.model.name,
        'ref_model_path': params.model.ref_model,
        'model_dtype': safe_string_to_torch_dtype(params.model.dtype),
        # ... 20+ 个参数从 config 中手动提取 ...
        'loss_variant': params.train.alg_name.lower(),
        'lr': params.train.lr,
        'betas': params.train.betas,
    }
    # 不同算法需要不同的 kwargs
    if params.train.alg_name.lower() == "ppo":
        kwargs['vf_clip'] = params.train.ppo_vf_clip
        kwargs['tau'] = params.train.ppo_tau
        kwargs['gamma'] = params.train.ppo_gamma
    ...
```

这是**僵化的硬连线**。每当新变体需要一个新参数时，这个函数就必须手动更新。kwargs 字典是一个隐式契约，如果构造函数发生变化，它可能会悄无声息地失效。

**改造后（v2.0）—— 配置驱动的组装：**

```python
# oxrl/setup/engine_factory.py v2.0 — 配置本身就是契约
def create_training_engines(config, master_addr):
    """每个变体接收完整配置，自行提取所需内容。"""
    alg_cls = RL_ALGORITHMS[config.train.alg_name.lower()]

    engines = []
    for rank in range(config.run.training_gpus):
        env_vars = _build_env_vars(master_addr, config.run.ray_master_port, rank, config.run.training_gpus)
        engine = alg_cls.options(num_gpus=1, runtime_env={"env_vars": env_vars}).remote(config=config)
        engines.append(engine)
    return engines
```

```python
# oxrl/algs/grpo.py v2.0 — 每个变体从 config 中提取所需内容
@ray.remote
class GRPO(BaseAlgorithm):
    def __init__(self, config):
        self.model_path = config.model.name
        self.loss_fn = LOSS_FUNCTIONS[config.train.alg_name.lower()]
        self.clip_low = config.train.clip_low
        self.clip_high = config.train.clip_high
        # ...只提取需要的内容...
```

**为什么：** 在 v1.0 中，添加一个变体特有的参数需要同时修改调用方和被调用方。在 v2.0 中，添加参数只需修改被调用方——配置已经携带了这个参数。

---

### 第三章：黑箱化与语义压缩

> *一旦一个大齿轮组装并验证完毕，立即将其封装为黑箱，并附上极简的调用文档。*

原则：每个模块都应该在顶部有一个**调用规范**——对其接口（方法、输入、输出、副作用）的紧凑摘要——这样 LLM 永远不需要阅读实现代码就能使用它。这就是语义压缩：将 500+ LOC 的实现压缩为约 30 行规范。

#### v1.0 做对的地方

`BaseAlgorithm` 抽象类（41 LOC）已经是一个良好的黑箱规范：

```python
class BaseAlgorithm(ABC):
    def is_ready(self) -> bool: ...
    def train_step(self, *args, **kwargs) -> Dict[str, float]: ...
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None): ...
    def gather_state_dict(self) -> Optional[dict]: ...
```

评分函数也遵循统一的纯函数签名。

#### v1.0 做错的地方：服务缺少调用规范

VLLMRolloutEngine **没有调用规范**。要使用它，你必须阅读 565 LOC 才能了解：
- 有哪些方法
- 它们接受什么参数
- 它们返回什么
- 它们有什么副作用

**改造前（v1.0）—— 没有规范，必须阅读源码：**

```python
# 要调用 VLLMRolloutEngine.generate()，你必须搞清楚：
# - `prompts` 期望什么格式？
# - `current_iter` 有什么作用？
# - `policy_version` 有什么作用？
# - 返回格式是什么？
# 答案：阅读 generate() 的 120 LOC + 辅助方法的 40 LOC = 160 LOC
```

**改造后（v2.0）—— 添加显式的调用规范作为模块级文档字符串：**

```python
# oxrl/rollouts/vllm_engine.py v2.0 — 顶部的黑箱规范
"""VLLMRolloutEngine — 用于生成数据样本的推理引擎。

CALLING SPEC（供 LLM 代理使用）:
    engine = VLLMRolloutEngine.remote(config=config, reward_func=fn, eos_id=id)

    # 刷新权重（2 种方法）:
    engine.refresh_model.remote(model_path, version)           → bool
    engine.refresh_model_from_state_dict.remote(sd, cfg, ver)  → bool

    # 生成样本:
    engine.generate.remote(
        prompts=[{"prompt_token_ids": [int, ...], "metadata": {...}}, ...],
        current_iter=int,
        policy_version=int,
    ) → List[Dict]:
        每个 dict 包含:
          "input_ids":          Tensor[T]    — 输入 + 输出拼接
          "rewards":            Tensor[T]    — 逐 token 评分（输入部分为 0）
          "pred_zscores":       Tensor[T]    — z-score 优势值
          "pred_masks":         Tensor[T]    — 输出预测位为 1
          "pred_old_logprobs":  Tensor[T]    — 与预测对齐的对数概率
          "response_len":       int          — 输出 token 数量

TOKEN BUDGET: 重构后约 200 LOC（原为 565 LOC）。
"""
```

现在 LLM 阅读规范（约 30 行，约 120 tokens）就能准确知道如何使用该引擎，**而无需阅读任何实现代码**。这就是语义压缩——将 565 LOC 压缩为 120 tokens。

#### 在所有地方添加规范

**v2.0 为每个模块添加调用规范：**

```python
# oxrl/rollouts/replay_buffer.py v2.0 — 规范
"""ReplayBuffer — 存储生成的样本用于迭代处理。

CALLING SPEC:
    buffer = ReplayBuffer(pad_token_id=int, max_seq_len=int)
    buffer.add_batch_seqs(samples: List[Dict])  — 添加样本
    buffer.reset()                               — 清空以准备下一轮迭代
    len(buffer)                                  → int
    DataLoader(buffer, collate_fn=buffer.collate_fn)  → 产出批次:
        {"input_ids": [B,T], "attn_mask": [B,T], "old_logprobs": [B,T],
         "mask": [B,T], "rewards": [B,T], "zscore": [B,T], "v_olds": [B,T]|None}
"""
```

```python
# oxrl/algs/losses/gspo_loss.py v2.0 — 规范
"""GSPO Loss — 面向 MoE 模型的序列级裁剪代理损失。

CALLING SPEC:
    loss, metrics = gspo_loss(logprobs, old_logprobs, advantages, mask, clip_low, clip_high)
    # loss:    标量 tensor
    # metrics: {"clipfrac": float, "approx_kl": float}
"""
```

**递归压缩**原则：在每一层级，LLM 只阅读下一层级的规范：

```
Level 0: variant_fn 规范     → "loss, metrics = variant_fn(inputs, ...)"     ~20 tokens
Level 1: engine.step 规范    → "metrics = engine.train_step(eid, batches)"   ~20 tokens
Level 2: phase 规范          → "epoch_metrics = run_processing_steps(...)"   ~15 tokens
Level 3: main.py             → "调用 phase_1 → phase_2 → phase_3"           ~10 tokens
```

理解整个管道所需的总上下文：**约 65 tokens**，而不是**约 10,500 tokens**。

---

## 第二部分：反馈回路

**论点：** 静态的代码拓扑已经死亡。AI 原生系统像数字生命体一样通过分层反馈进化。

本部分讲述的是能够检测自身失败并自适应的系统——不仅仅是记录指标，而是对指标采取行动。这些模式适用于任何迭代系统：训练循环、CI/CD 管道、数据处理管道或自主代理。

---

### 第四章：目标分解

> *入口点是高层的"北极星"目标。系统将其分解为可度量、可测试的子目标。*

原则：每个管道阶段都应返回一个结构化结果，包含成功/失败状态和可度量的指标。下游阶段可以检查这些结果并据此调整。这将一个盲目的顺序管道转变为一个有自我意识的系统。

#### v1.0 做对的地方

swarm 系统已经实现了目标分解：

```python
# oxrl/swarm/scout.py — 好的做法：清晰的子目标分解
def onboard_model(model_id, entry):
    discover_info = step_discover(model_id, entry)    # 子目标 1：验证输入
    step_preprocess(dataset, model_slug)               # 子目标 2：准备数据
    config_path = step_generate_config(...)             # 子目标 3：生成配置
    log_path = step_train(config_path, ...)             # 子目标 4：训练
    eval_result = step_evaluate(log_path)               # 子目标 5：评估
    step_archive(...)                                   # 子目标 6：归档
    step_gc(model_id)                                   # 子目标 7：清理
    step_update_manifest(...)                           # 子目标 8：更新状态
```

每个子目标都是可度量的（失败时抛出 `RuntimeError`）、可测试的（可以独立运行）和独立的（如果已完成可以跳过）。

#### v1.0 做错的地方：训练循环没有目标分解

**改造前（v1.0）—— `main_rl.py` 主循环——一个没有子目标的扁平序列：**

```python
for epoch in range(number_of_epochs):
    # 数据生成 ... 20 LOC 的内联逻辑
    rollout_stats = collect_rollouts(...)

    # 数据准备 ... 30 LOC 的内联逻辑
    train_batches = list(DataLoader(...))
    # 填充逻辑 ... 10 LOC
    train_batches_padded = ...

    # 训练 ... 40 LOC 的内联调度 + 指标收集
    for tidx in range(number_of_training_steps_per_epoch):
        train_futures = []
        for eid, engine in enumerate(training_engine_runners):
            shard = train_batches_padded[eid::num_train_engines]
            train_futures.append(engine.train_step.remote(...))
        train_metrics = ray.get(train_futures)
        # 10 LOC 的指标聚合
        # 5 LOC 的日志记录

    # 状态持久化 ... 60 LOC，包含 2 条不同的代码路径
    try:
        gather_futures = [engine.gather_state_dict.remote() for engine in training_engine_runners]
        gather_results = ray.get(gather_futures)
        state_dict = next((r for r in gather_results if r is not None), None)
    except:
        state_dict = None

    if state_dict is not None:
        # 快速路径：对象存储 ... 30 LOC
    else:
        # 旧路径：基于磁盘 ... 15 LOC
```

没有命名的子目标。没有每个阶段的可度量结果。系统无法检测到"训练正在失败"并做出调整。

**改造后（v2.0）—— 带有可度量结果的命名子目标：**

```python
# main_rl.py v2.0 — 每个阶段返回可度量的结果
for epoch in range(number_of_epochs):
    # 子目标 1：生成数据 → 度量输出质量
    rollout_result = rollout_phase(rollout_dataloader, rollout_engines, epoch, policy_version, replay_buffer)
    assert rollout_result.total_samples > 0, "No samples generated"

    # 子目标 2：处理数据 → 度量收敛情况
    train_result = train_phase(training_engines, replay_buffer, config, epoch)
    assert not math.isnan(train_result.avg_loss), "NaN loss detected"

    # 子目标 3：保存状态 + 刷新 → 度量成功与否
    checkpoint_result = checkpoint_phase(training_engines, rollout_engines, config, epoch, policy_version, tokenizer)
    assert checkpoint_result.success, f"Checkpoint failed: {checkpoint_result.error}"

    # 子目标 4：健康检查 → 检测退化
    health = health_check(rollout_result, train_result, epoch)
    if health.primary_metric_stalled:
        logger.warning("Primary metric stalled for 5 iterations — consider adjusting parameters")
    if health.divergence_detected:
        logger.warning("Divergence detected — consider reducing step size")

    policy_version += 1
    replay_buffer.reset()
```

每个子目标返回一个**结构化结果**，下游阶段可以对其进行检查。`health_check` 是 v1.0 中不存在的新反馈机制。

---

### 第五章：进化式齿轮编排

> *系统自行决定复用哪些齿轮、淘汰哪些齿轮、锻造哪些新齿轮。*

原则：当系统遇到故障时，应该有一个逐级升级的响应策略——不是简单地用相同参数重试，而是逐步升级到越来越激进的策略：调整参数 → 更换组件 → 启用新特性 → 放弃任务。

#### v1.0 做对的地方

编排器 + 修复器模式已经具有进化性：

```
Scout（尝试）→ 成功 → 上线
              → 失败 → Bugfixer（分类 + 修复）→ 重新排队 → Scout 再次尝试
                                                → 跳过（放弃）
```

#### v1.0 做错的地方：修复器只调整参数，从不更换组件

**改造前（v1.0）—— Bugfixer 只应用参数补丁：**

```python
# oxrl/swarm/bugfixer.py — 局限性：只修改配置值
def _fix_oom(config_path):
    changes["train.train_batch_size_per_gpu"] = max(1, train_bs // 2)
    changes["rollout.rollout_batch_size_per_gpu"] = max(1, rollout_bs // 2)
    return {"action": "adjust_config", "changes": changes}

def _fix_nan_loss(config_path):
    changes["train.lr"] = current_lr / 10.0
    changes["train.clip_grad_norm"] = 5.0
    return {"action": "adjust_config", "changes": changes}
```

修复器可以将批大小减半、降低学习率。但它无法：
- 当前算法不适用时切换算法变体
- 当系统在最小批大小时仍然 OOM 时启用内存高效适配器
- 当评分始终为零时更换评分函数
- 当资源紧张时添加内存优化技术

**改造后（v2.0）—— Bugfixer 可以锻造新齿轮（更换组件、启用适配器等）：**

```python
# oxrl/swarm/bugfixer.py v2.0 — 扩展的修复分类体系
def _fix_oom(config_path):
    cfg = _load_config_yaml(config_path)
    train_bs = cfg.get("train", {}).get("train_batch_size_per_gpu", 1)

    # 级别 1：减少批大小（参数调整）
    if train_bs > 1:
        return {"action": "adjust_config",
                "changes": {"train.train_batch_size_per_gpu": max(1, train_bs // 2)}}

    # 级别 2：启用适配器（锻造新齿轮——更换组件）
    if not cfg.get("lora", {}).get("enabled", False):
        return {"action": "adjust_config",
                "changes": {"lora.enabled": True, "lora.r": 16, "lora.lora_alpha": 32},
                "reason": "OOM at batch_size=1 — enabling adapter to reduce parameters"}

    # 级别 3：启用内存优化（锻造新齿轮）
    if not cfg.get("model", {}).get("gradient_checkpointing", False):
        return {"action": "adjust_config",
                "changes": {"model.gradient_checkpointing": True}}

    # 级别 4：CPU 卸载（已有）
    ...

    # 级别 5：放弃
    return {"action": "skip", "reason": "OOM with all mitigations exhausted"}

def _fix_score_zero():
    """v2.0 新增：当评分始终为 0 时更换评分函数。"""
    # 分析数据集元数据以确定正确的函数
    if "gsm8k" in dataset:
        return {"action": "adjust_config",
                "changes": {"reward.reward_func": "soft_math_reward_func"},
                "reason": "Score always 0 — switching to partial-credit function"}
    ...
```

v2.0 的修复器可以**锻造新齿轮**（启用适配器、切换评分函数）和**淘汰齿轮**（永久跳过），而不仅仅是调整参数。

---

### 第六章：多层反馈网络

> *微循环自修复步骤级错误。宏循环调整系统策略。*

原则：反馈应在多个时间尺度上运行。逐步检查能立即捕获灾难性故障。逐迭代检查能检测退化趋势。逐运行检查能处理基础设施故障。大多数系统只实现了最外层的循环。

#### v1.0 做错的地方：没有微级别的自修复

**改造前（v1.0）—— 训练循环盲目运行，没有健康监控：**

```python
# main_rl.py — 没有检测训练退化的机制
for tidx in range(number_of_training_steps_per_epoch):
    train_metrics = ray.get(train_futures)
    avg_loss = np.mean([m.get('loss_total', 0.0) for m in train_metrics])
    avg_kl_old = np.mean([m.get('kl_old', 0.0) for m in train_metrics])
    avg_clipfrac = np.mean([m.get('clipfrac', 0.0) for m in train_metrics])
    # 记录日志... 但从不对其采取行动。
    # 如果所有更新都被裁剪 → 系统什么都没学到 → 继续盲目运行
    # 如果散度指标在爆炸 → 系统不稳定 → 继续盲目运行
    # 如果 loss = NaN → 系统已死 → 继续盲目运行（直到崩溃）
```

**改造后（v2.0）—— 带有自动健康检测的微循环：**

```python
# oxrl/loops/train_phase.py v2.0 — 具有健康感知的训练
def run_training_steps(training_engines, replay_buffer, config, epoch):
    epoch_metrics = defaultdict(list)

    for tidx in range(config.train.train_steps_per_epoch):
        train_metrics = ray.get(train_futures)
        step_metrics = aggregate_metrics(train_metrics)

        # 微反馈循环：检测并标记退化
        if math.isnan(step_metrics['loss_total']):
            logger.error("[MICRO-LOOP] NaN loss at step %d — halting iteration early", tidx)
            return TrainResult(success=False, error="nan_loss", metrics=epoch_metrics)

        if step_metrics['clipfrac'] > 0.95:
            logger.warning("[MICRO-LOOP] clipfrac=%.2f at step %d — updates being fully clipped",
                         step_metrics['clipfrac'], tidx)

        if step_metrics['kl_old'] > 10.0:
            logger.warning("[MICRO-LOOP] kl_old=%.2f at step %d — significant drift detected",
                         step_metrics['kl_old'], tidx)

    return TrainResult(success=True, error=None, metrics=epoch_metrics)
```

```python
# oxrl/loops/health_check.py v2.0 — 新增：宏级健康监控
def health_check(rollout_result, train_result, epoch, history):
    """检测系统级退化模式。"""
    issues = []

    # 主要指标停滞：连续 N 轮迭代没有改善
    if len(history.primary_metric) >= 5:
        recent = history.primary_metric[-5:]
        if max(recent) - min(recent) < 0.01:
            issues.append(HealthIssue("metric_stalled",
                "Primary metric unchanged for 5 iterations — consider different parameters"))

    # 发散：不稳定指标逐迭代增长
    if len(history.divergence_metric) >= 3:
        if all(history.divergence_metric[i] < history.divergence_metric[i+1] for i in range(-3, -1)):
            issues.append(HealthIssue("diverging",
                "Instability metric increasing — consider reducing step size"))

    # 输出坍塌：系统产生退化输出
    if rollout_result.avg_response_len < 10:
        issues.append(HealthIssue("output_collapse",
            "Average output length < 10 tokens — system may have collapsed"))

    return HealthResult(issues=issues)
```

这构建了一个**多层反馈网络**：

```
┌────────────────────────────────────────────────────────┐
│  微循环（每个训练步骤）                                  │
│  ├── 检测到 NaN          → 提前终止迭代                 │
│  ├── 更新被完全裁剪      → 记录警告                     │
│  └── 散度突增            → 记录警告                     │
├────────────────────────────────────────────────────────┤
│  宏循环（每轮迭代）                                      │
│  ├── 指标停滞            → 建议调整参数                  │
│  ├── 发散趋势            → 建议减小步长                  │
│  └── 输出坍塌            → 建议调整配置                  │
├────────────────────────────────────────────────────────┤
│  系统循环（跨运行）— 已有                                │
│  ├── OOM → bugfixer 减少资源使用                        │
│  ├── NaN → bugfixer 将步长降低 10 倍                    │
│  └── 超时 → bugfixer 调整基础设施配置                    │
└────────────────────────────────────────────────────────┘
```

v1.0 只有系统循环（bugfixer）。v2.0 增加了微循环和宏循环，可以在训练**进行中**检测问题，而不只是在运行失败**之后**。

---

## 第三部分：工具化

**论点：** 在生成模型的概率世界中，确定性的围墙必须存在。大型软件系统不能在基础设施上掷骰子。

本部分讲述的是 AI 决策（路由、参数、策略）与必须确定性的部分（数学计算、I/O、验证）之间的边界。每个代码库都有这条边界——这里的模式帮助你清晰地划定它。

---

### 第七章：概率性与确定性

> *永远不要让模型以概率方式实现高风险的基础设施逻辑。*

#### 铁律

| 组件 | 本质 | 由谁实现 |
|---|---|---|
| "为这个工作负载选择变体 B" | 意图理解，路由 | LLM（概率性） |
| 变体 B 的公式本身 | 确定性数学 | 人类编写、测试、封装 |
| "为这个输入设置 step_size=5e-6" | 判断，参数选择 | LLM（概率性） |
| 配置校验（`extra='forbid'`） | Schema 强制执行 | Pydantic（确定性工具） |
| "启用适配器以修复 OOM" | 诊断，策略 | Bugfixer 代理（概率性） |
| 适配器权重合并 `delta = B @ A * scaling` | 线性代数 | 人类编写、测试、封装 |
| "用更小的批大小重试" | 恢复决策 | 编排器（概率性） |
| 状态保存的文件 I/O | 基础设施 | 人类编写、确定性工具 |

这条边界存在于每个系统中：Web 应用有 LLM 生成的 SQL（危险）与参数化查询（安全）之分。CI 系统有 LLM 选择的测试策略（灵活）与测试运行器本身（确定性）之分。划定界限。封装确定性的那一侧。

#### v1.0 做对的地方

on-policy 强制执行墙已经是确定性的：

```python
# 好的做法：硬错误，不是警告
if self.force_strict_on_policy:
    if self.temperature != 1.0:
        raise ValueError("Strict on-policy requires temperature = 1.0")
```

Pydantic 配置校验已经是严格的：

```python
# 好的做法：extra='forbid' 防止幻觉字段
class Train(BaseModel):
    model_config = ConfigDict(extra='forbid')
```

#### v1.0 做错的地方：确定性逻辑被埋在类里面

**改造前（v1.0）—— 权重合并是算法类的一个方法，不是独立工具：**

```python
# oxrl/algs/grpo.py — 权重合并在 GRPO 类内部
class GRPO(BaseAlgorithm):
    def _strip_lora_and_merge(self, state_dict):
        # 42 LOC 的精确线性代数
        # 使用 self.lora_config（与类状态耦合）
        for k in list(new_state_dict.keys()):
            scaling = alpha / r
            delta = (lb_w @ la_w) * scaling
            new_state_dict[k] = base_w + delta.to(base_w.dtype)
```

**问题：** PPO 也需要权重合并。在 v1.0 中，PPO 有一个**重复的**方法。如果在合并逻辑中发现了 bug，必须在两个地方同时修复。

**改造后（v2.0）—— 权重合并作为独立的确定性工具：**

```python
# oxrl/tools/lora_merge.py v2.0 — 封装、测试、可复用
def strip_lora_and_merge(state_dict, lora_alpha, lora_r):
    """将适配器权重合并到基础模型权重中。

    DETERMINISTIC TOOL CONTRACT:
        Input:  包含基础权重 + 适配器权重的 state_dict
        Output: 合并后的 state_dict，适配器键已移除
        Side effects: 无（返回新字典）
        Reproducibility: 保证（纯函数）

    Args:
        state_dict:  dict of {name: tensor} — 可能包含 PEFT 前缀
        lora_alpha:  int — 缩放因子的分子
        lora_r:      int — 秩（缩放因子的分母）

    Returns:
        dict of {name: tensor} — 合并后的权重，无适配器键
    """
    new_state_dict = {}
    lora_weights = {}

    for k, v in state_dict.items():
        clean_k = k.removeprefix("base_model.model.")
        if ".lora_A." in clean_k or ".lora_B." in clean_k:
            lora_weights[clean_k] = v
        elif ".base_layer." in clean_k:
            new_state_dict[clean_k.replace(".base_layer.", ".")] = v
        else:
            new_state_dict[clean_k] = v

    scaling = lora_alpha / lora_r
    for k in list(new_state_dict.keys()):
        prefix = k.rsplit(".", 1)[0]
        la = f"{prefix}.lora_A.default.weight"
        lb = f"{prefix}.lora_B.default.weight"
        if la in lora_weights and lb in lora_weights:
            delta = (lora_weights[lb] @ lora_weights[la]) * scaling
            if delta.shape == new_state_dict[k].shape:
                new_state_dict[k] = new_state_dict[k] + delta.to(new_state_dict[k].dtype)

    return new_state_dict
```

现在任何算法都可以调用 `strip_lora_and_merge(state_dict, alpha, r)` —— 一个单一的、经过测试的、确定性的工具。

---

### 第八章：基础设施即 LLM 工具

> *识别确定性组件，并将它们转化为超越模型认知突变的独立工具。*

原则：基础设施逻辑（文件 I/O、分布式协调、张量操作、配置提取）不应该嵌入业务逻辑类中。应将其提取为独立的工具，供任何类导入使用。

#### v1.0 做错的地方：状态持久化有两条代码路径埋在业务类中

**改造前（v1.0）—— `GRPO.save_checkpoint()` — 98 LOC，含 2 条代码路径：**

```python
class GRPO(BaseAlgorithm):
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None):
        if state_dict_ref is not None:
            # 快速路径：从对象存储写入预收集的 state dict
            if rank == 0:
                state_dict = state_dict_ref
                config_dict = state_dict.pop("__model_config_dict__", None)
                # 断开共享内存张量 ... 8 LOC
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                # 保存配置 ... 5 LOC
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            # 旧路径：分布式收集 + 写入
            self.policy_engine.save_16bit_model(output_dir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            # 如果使用适配器则在 rank 0 修复 state dict ... 20 LOC
            # 在 rank 0 保存配置 ... 5 LOC
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
```

这是基础设施逻辑（文件 I/O、分布式屏障、张量去重）嵌入在领域特定类中。LLM 修改算法时必须在持久化 I/O 代码中艰难跋涉。

**改造后（v2.0）—— 状态持久化作为独立工具：**

```python
# oxrl/tools/checkpoint.py v2.0 — 确定性持久化工具
def save_state_dict_to_disk(state_dict, output_dir, config_dict=None):
    """将 state dict 以 safetensors 格式保存到磁盘。

    TOOL CONTRACT:
        Input:  state_dict (dict), output_dir (str), 可选 config_dict
        Output: 文件写入 output_dir/
        Side effects: 在磁盘上创建文件
        Deterministic: 是 — 相同输入 → 相同文件
    """
    os.makedirs(output_dir, exist_ok=True)
    # 断开共享内存张量以兼容 safetensors
    state_dict = _dedup_shared_tensors(state_dict)
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    if config_dict is not None:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

def gather_and_save(engine, output_dir, tag, lora_config=None):
    """收集分布式权重并保存到磁盘。集体操作。"""
    state_dict = engine._zero3_consolidated_16bit_state_dict()
    torch.distributed.barrier()
    rank = torch.distributed.get_rank()
    if rank == 0 and state_dict is not None:
        if lora_config and lora_config.enabled:
            state_dict = strip_lora_and_merge(state_dict, lora_config.lora_alpha, lora_config.r)
        config_dict = _extract_model_config(engine)
        save_state_dict_to_disk(state_dict, output_dir, config_dict)
    torch.distributed.barrier()
```

现在算法的保存方法变成了 5 行的委托调用：

```python
class GRPO(BaseAlgorithm):
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None):
        if state_dict_ref is not None:
            save_state_dict_to_disk(state_dict_ref, output_dir, config_dict)
        else:
            gather_and_save(self.policy_engine, output_dir, tag, self.lora_config)
```

---

#### 完整的 v2.0 工具清单

**改造前（v1.0）—— 工具埋在类里面：**

```
GRPO 类（665 LOC）:
  ├── _strip_lora_and_merge()     → 权重合并
  ├── gather_state_dict()          → 分布式收集
  ├── save_checkpoint()            → 文件 I/O（2 条路径）
  └── _get_base_model_config()     → 配置提取

VLLMRolloutEngine 类（565 LOC）:
  ├── normalize_rewards()          → z-score 归一化
  ├── extract_logprobs()           → 输出解析
  ├── score_response()             → 评分计算
  └── make_sampling_params()       → 参数校验

Config 类（483 LOC）:
  ├── _sync_batch_sizes()          → 配置同步
  ├── _sync_dtype()                → 数据类型映射
  ├── _sync_optimizer()            → 优化器配置
  └── _sync_scheduler()            → 调度器配置
```

**改造后（v2.0）—— 工具提取为独立模块：**

```
oxrl/tools/                        # 新增：独立确定性工具
├── lora_merge.py                  # strip_lora_and_merge()              ~50 LOC
├── checkpoint.py                  # save_state_dict_to_disk()           ~40 LOC
│                                  # gather_and_save()                   ~30 LOC
├── tensor_utils.py                # dedup_shared_tensors()              ~15 LOC
│                                  # ensure_1d(), pad_1d_to_length()     ~20 LOC
└── config_extract.py              # extract_model_config()              ~15 LOC

oxrl/rollouts/
├── normalization.py               # normalize_rewards()                 ~40 LOC
├── logprob_utils.py               # extract_logprobs()                  ~40 LOC
├── reward_scoring.py              # score_response()                    ~20 LOC
└── sampling.py                    # make_sampling_params()              ~50 LOC

oxrl/configs/
├── schema.py                      # 仅 Pydantic 模型                    ~200 LOC
├── sync.py                        # sync_deepspeed_config() 及相关函数  ~150 LOC
└── loader.py                      # load_and_verify()                   ~50 LOC
```

每个工具都是：
- **独立的**：可以独立导入和调用
- **经过测试的**：有清晰的输入、输出和不变量
- **确定性的**：相同输入 → 相同输出
- **小巧的**：在 LLM 的最优智能窗口范围内

---

### 第九章：零幻觉契约

> *工具必须保证稳定的 I/O、可复现性和零副作用。*

原则：在系统边界（用户输入、配置文件、外部 API 响应）处，要进行积极的验证。使用 Schema 强制执行来防止 LLM——或任何调用者——传入无意义的值。静默失败是最大的敌人。

#### v1.0 做对的地方

评分函数已经有稳定的契约：

```python
def reward_func(prompt_ids, response_ids, finish_reason, metadata=None) -> Tuple[Tensor, bool]
```

Pydantic 已经拒绝未知字段：

```python
class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')  # ← 零幻觉防线
```

#### v1.0 做错的地方：配置中的静默失败

**改造前（v1.0）—— 静默默认值掩盖了配置错误：**

```python
class Rollout(BaseModel):
    temperature: float = 1.0          # 默认值：合理
    max_tokens: int = 512             # 默认值：合理
    n_samples: int = 8                # 默认值：合理
    force_strict_on_policy: bool = True  # 默认值：强制执行

# 问题：如果有人设置 n_samples=0，系统静默产生空输出。
# 问题：如果有人设置 max_tokens=0，服务静默返回空结果。
```

**改造后（v2.0）—— Pydantic 验证器捕获一切：**

```python
class Rollout(BaseModel):
    model_config = ConfigDict(extra='forbid')
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32768)         # ← 必须 > 0
    n_samples: int = Field(default=8, ge=1, le=64)               # ← 必须 >= 1
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    gpu_memory_utilization: float = Field(default=0.5, gt=0.0, le=0.95)  # ← 不能为 1.0
    rollout_batch_size_per_gpu: int = Field(default=2, ge=1)     # ← 必须 >= 1
    force_strict_on_policy: bool = True

class Train(BaseModel):
    model_config = ConfigDict(extra='forbid')
    lr: float = Field(default=1e-5, gt=0, le=1.0)               # ← 不能为负数或 > 1
    clip_low: float = Field(default=0.2, ge=0.0, le=1.0)        # ← 有界
    clip_high: float = Field(default=0.2, ge=0.0, le=1.0)       # ← 有界
    total_number_of_epochs: int = Field(ge=1)                     # ← 必须 >= 1
    train_batch_size_per_gpu: int = Field(default=2, ge=1)       # ← 必须 >= 1
    gradient_accumulation_steps: int = Field(default=1, ge=1)    # ← 必须 >= 1
```

每个字段都有显式的边界。LLM 在生成配置时无法设置 `n_samples: 0` 或 `lr: -0.001`——Pydantic 会立即抛出 `ValidationError`。工具剥夺了 LLM 在基础设施参数上的即兴发挥自由。

#### v1.0 做错的地方：外部调用会静默崩溃

**改造前（v1.0）—— 裸 except 吞掉所有错误：**

```python
# oxrl/rewards/code.py
def code_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
    ...
    try:
        result = subprocess.run(["python", f.name], capture_output=True, timeout=5)
        if result.returncode == 0:
            r[-1] = 1.0
    except Exception:
        pass     # ← 静默失败！可能是超时、权限错误、OOM 或任何问题。
    return r, is_per_token
```

**改造后（v2.0）—— 结构化错误报告：**

```python
# oxrl/rewards/code.py v2.0 — 报告错误，而不是吞掉它们
def code_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
    ...
    try:
        result = subprocess.run(["python", f.name], capture_output=True, timeout=5)
        if result.returncode == 0:
            r[-1] = 1.0
        # returncode != 0 意味着测试失败 → 分数保持 0.0（正确行为）
    except subprocess.TimeoutExpired:
        pass  # 超时 → 视为失败（分数 = 0.0）— 这是有意的
    except Exception as e:
        # 记录意外错误以便诊断
        import logging
        logging.getLogger(__name__).warning("code_reward unexpected error: %s", e)
    return r, is_per_token
```

---

## 尾声

### 从代码编写者到系统牧羊人

#### 开发者的新角色

掌握这些模式的工程师不再是 if-else 劳工。我们变成了：

1. **原子齿轮的提供者** —— 编写小巧的、单一用途的模块，能够放入 LLM 的注意力窗口
2. **目标-反馈系统的设计者** —— 构建 尝试 → 评估 → 诊断 → 修复 → 重试 的循环
3. **确定性边界的架构师** —— 定义什么是封装的（工具）、什么是灵活的（路由）

无论你是在构建训练框架、Web 应用、数据管道还是 CLI 工具，这些角色都是相同的。这些模式具有普适性。

#### oxRL v1.0 → v2.0 转型总结

| 指标 | v1.0 | v2.0 | 改进 |
|---|---|---|---|
| 最大的类（GRPO） | 665 LOC | ~250 LOC | 缩小 2.7 倍 |
| 算法变体（在一个方法中） | 3 个交织在一起 | 3 个独立文件 | 零交叉污染 |
| main_rl.py | 595 LOC，8 个关注点 | 120 LOC，纯编排 | 缩小 5 倍 |
| VLLMRolloutEngine | 565 LOC，5 个关注点 | 200 LOC，2 个关注点 | 缩小 2.8 倍 |
| 配置文件 | 483 LOC，3 个关注点 | 3 个文件，每个约 130 LOC | 更专注 |
| 权重合并 | 在 2 个类中重复 | 单一工具，可复用 | DRY |
| 状态持久化 | 98 LOC 嵌入算法类 | 70 LOC 独立工具 | 解耦 |
| 健康监控 | 无 | 微循环 + 宏循环 + 系统循环 | 三层反馈 |
| Bugfixer 策略 | 仅参数补丁 | 参数 + 适配器 + 组件更换 | 更丰富的进化能力 |
| 模块黑箱规范 | 无（必须读源码） | 每个模块都有调用规范 | 约 65 tokens 即可理解整个管道 |

#### 终极愿景

用人类的智慧设计**规则**（Pydantic 验证器、强制执行墙）、**反馈回路**（尝试 → 诊断 → 修复 → 重试）和**工具边界**（什么是确定性的 vs 什么是概率性的）。

让 AI 的算力去**动态组装**微齿轮、在确定性骨架的约束内**重构**代码，并通过反馈回路**进化**——无需人类干预。

---

## 附录：完整的前后文件对照表

### 改造前（v1.0）—— 当前结构

```
oxrl/
├── algs/
│   ├── base.py              41 LOC  ← 保持（已经是良好的微齿轮）
│   ├── grpo.py             665 LOC  ← 打碎（7 项职责）
│   ├── ppo.py              880 LOC  ← 打碎（类似问题）
│   ├── sft.py              145 LOC  ← 保持（已经足够专注）
│   ├── dpo.py              100 LOC  ← 保持
│   └── ... (11 more, 60-100 LOC each) ← 保持
├── configs/
│   └── load.py             483 LOC  ← 拆分（schema + sync + loader）
├── rollouts/
│   ├── vllm_engine.py      565 LOC  ← 提取（5 个关注点 → 1 个服务 + 4 个工具）
│   └── replay_buffer.py    208 LOC  ← 保持（已经足够专注）
├── rewards/
│   ├── base.py              62 LOC  ← 保持
│   ├── math.py              77 LOC  ← 保持
│   ├── code.py              37 LOC  ← 改进（错误报告）
│   └── ... (5 more)                 ← 保持
├── swarm/
│   ├── scout.py            898 LOC  ← 保持（已经有目标分解）
│   ├── bugfixer.py         777 LOC  ← 扩展（更丰富的修复策略）
│   └── orchestrator.py     545 LOC  ← 保持（已经具有进化性）
├── datasets/                        ← 保持（文件已经足够专注）
├── utils/                           ← 保持
├── main_rl.py              595 LOC  ← 打碎（→ 编排器 + 3 个阶段）
└── main_sl.py              ~600 LOC ← 打碎（与 main_rl 相同的模式）
```

### 改造后（v2.0）—— 建议的结构

```
oxrl/
├── algs/
│   ├── base.py              41 LOC  — 接口契约
│   ├── losses/                       — 新增：提取的算法变体
│   │   ├── sgrpo_loss.py    40 LOC  — 变体 A：token 级裁剪代理
│   │   ├── gspo_loss.py     35 LOC  — 变体 B：序列级（MoE）
│   │   ├── cispo_loss.py    30 LOC  — 变体 C：保守间接
│   │   └── ppo_loss.py      40 LOC  — 变体 D：策略 + 价值
│   ├── grpo.py             250 LOC  — 仅编排（原 665）
│   ├── ppo.py              400 LOC  — 仅编排（原 880）
│   ├── sft.py              145 LOC  — 未改动
│   ├── dpo.py              100 LOC  — 未改动
│   └── ...                          — 未改动
│
├── tools/                            — 新增：确定性工具
│   ├── lora_merge.py        50 LOC  — 权重合并（原在算法类中）
│   ├── checkpoint.py        70 LOC  — 保存/收集/去重（原在算法类中）
│   ├── tensor_utils.py      35 LOC  — ensure_1d, pad_1d（原在 utils 中）
│   └── config_extract.py    15 LOC  — 提取模型配置（原在算法类中）
│
├── configs/
│   ├── schema.py           200 LOC  — 纯 Pydantic 模型（原在 load.py 中）
│   ├── sync.py             150 LOC  — 配置同步逻辑（原在 load.py 中）
│   └── loader.py            50 LOC  — YAML 加载（原在 load.py 中）
│
├── rollouts/
│   ├── vllm_engine.py      200 LOC  — 服务：生命周期 + 核心处理（原 565）
│   ├── sampling.py          50 LOC  — 参数构建（原在服务中）
│   ├── logprob_utils.py     40 LOC  — 输出解析（原在服务中）
│   ├── reward_scoring.py    20 LOC  — score_response()（原在服务中）
│   ├── normalization.py     40 LOC  — z-score 归一化（原在服务中）
│   └── replay_buffer.py    208 LOC  — 未改动
│
├── loops/                            — 新增：训练循环阶段
│   ├── rollout_phase.py     60 LOC  — 数据生成（原在 main 中）
│   ├── train_phase.py       50 LOC  — 调度 + 指标（原在 main 中）
│   ├── checkpoint_phase.py  80 LOC  — 收集 + 保存 + 刷新（原在 main 中）
│   └── health_check.py      60 LOC  — 新增：微/宏级健康监控
│
├── setup/                            — 新增：初始化与循环分离
│   ├── ray_setup.py         30 LOC  — 基础设施初始化（原在 main 中）
│   ├── engine_factory.py    60 LOC  — 创建 Actor（原在 main 中）
│   └── dataloader_factory.py 30 LOC — 创建数据管道（原在 main 中）
│
├── rewards/                          — 保持 + 改进错误报告
├── swarm/                            — 保持 + 扩展 bugfixer 策略
├── datasets/                         — 保持
├── utils/                            — 保持
│
├── main_rl.py              120 LOC  — 纯编排（原 595）
└── main_sl.py              120 LOC  — 纯编排（原约 600）
```

### Token 预算对比

| 任务 | v1.0 Tokens | v2.0 Tokens（最大文件） | 缩减倍数 |
|---|---|---|---|
| 修改 GRPO 算法 | ~2,700 | ~1,000 | 2.7x |
| 修改某一算法变体 | ~2,700（必须全部读取） | ~140 | **19x** |
| 修改训练管道 | ~2,400 | ~480 | 5x |
| 调试状态持久化 | ~2,700（必须全部读取） | ~280 | **9.6x** |
| 添加配置字段 | ~1,950 | ~800 | 2.4x |
| 修改推理服务 | ~2,300 | ~800 | 2.9x |
| 修改归一化逻辑 | ~2,300（必须全部读取） | ~160 | **14x** |
| 理解整个管道（通过规范） | ~10,500 | ~260（仅规范） | **40x** |
