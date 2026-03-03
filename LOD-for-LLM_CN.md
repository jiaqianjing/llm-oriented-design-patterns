# 面向 LLM 的设计模式（LOD）

你写的代码，主要由 LLM 来阅读、修改和组装。遵循以下设计模式，可以最大程度提升 AI 的理解能力，降低幻觉风险。

所有示例均来自 oxRL（一个 LLM 后训练框架），但这些模式适用于任何代码库。

---

## 硬性规则

1. **文件不超过 800 LOC（约 4k-8k tokens）。** 每个文件必须控制在模型的最佳智能窗口内。超过就拆分。
2. **一个文件，一个职责。** 不要把 schema 定义和同步逻辑混在一起，也不要把核心算法和 checkpoint I/O 放在同一个文件。
3. **优先使用纯函数，而非方法。** 如果逻辑不需要 `self`，就提取为独立函数。
4. **扁平优于嵌套。** 继承链不超过 1 层。不要使用工厂的工厂模式。用 dict 分发比 Strategy 模式更好。
5. **显式优于隐式。** 每个模块开头都要有调用规格——输入、输出、副作用——让 LLM 无需阅读实现就能使用它。
6. **确定性逻辑是封闭的。** 数学公式、文件 I/O、张量操作、配置校验——这些是工具。由人类编写、测试，AI 永远不应重新生成。
7. **概率性逻辑是灵活的。** 路由决策、参数选择、变体选择——这些是 LLM 的领域。

---

## 模式 1：极致拆分

将臃肿的大类拆分为单一职责的微模块。

反模式——承担 7 个职责的上帝类：
```python
# oxRL 示例：GRPO 算法类 — 665 LOC
@ray.remote
class GRPO:
    def __init__(self, ...):              # engine setup
    def load_model(self):                 # model I/O
    def compute_kl_distance(self, ...):   # math utility
    def compute_policy_loss(self, ...):   # 3 algorithm variants behind if-else
    def train_step(self, ...):            # training orchestration
    def _strip_lora_and_merge(self, ...): # weight merging (duplicated in PPO)
    def save_checkpoint(self, ...):       # file I/O (2 code paths)
    # Must read all 665 LOC to modify any single concern
```

模式——拆成聚焦的文件：
```
algs/
├── losses/
│   ├── sgrpo.py       # 67 LOC — one algorithm variant per file
│   ├── gspo.py        # 75 LOC
│   └── cispo.py       # 66 LOC
├── grpo.py            # 477 LOC — orchestration only
tools/
├── lora_merge.py      # 61 LOC — pure function, no class state
├── checkpoint.py      # 84 LOC — save/gather/config extraction
```

修改某个变体：只需读 75 LOC，而不是 665 LOC。修复权重合并：只需读 61 LOC，且所有算法都能复用。

**核心原则**：如果一个类有 N 个职责，LLM 修改第 3 个职责时必须加载全部 N 个到上下文。拆分后，它只需加载第 3 个。

---

## 模式 2：调用规格即黑盒

每个模块顶部都有一份调用规格。LLM 读规格（约 20-30 tokens）而非读实现（500+ tokens）。

反模式——没有规格，必须读完整个源码才能使用：
```python
class SomeService:
    # What methods exist? What args do they take? What do they return?
    # Answer: read the entire file.
```

模式——每个模块顶部都有规格：
```python
"""SomeService — Brief description of purpose.

CALLING SPEC:
    svc = SomeService.remote(config=config)

    svc.process.remote(
        items=[{"id": int, "data": {...}}, ...],
        batch_id=int,
    ) -> List[Dict]:
        Each dict: result_id (int), score (float), metadata (dict)

    svc.refresh.remote(new_model_path, version) -> bool
"""
```

**递归压缩**——每一层只需读下一层的规格：
```
Level 0: variant_fn spec    -> "result = variant_fn(inputs)"        ~20 tokens
Level 1: engine.step spec   -> "metrics = engine.step(batch)"       ~20 tokens
Level 2: phase spec         -> "phase_result = run_phase(...)"      ~15 tokens
Level 3: main.py            -> "calls phase_1 -> phase_2 -> phase_3" ~10 tokens
Total: ~65 tokens to understand the full pipeline (vs ~10,000 tokens reading source)
```

---

## 模式 3：变体函数注册表

当多个实现共享相同接口时（算法变体、序列化格式、评分策略），使用签名一致的纯函数注册表。用 dict 分发，而非 if-else。

反模式——多个变体交织在一个方法中：
```python
def compute(self, data, ...):
    if self.variant == "A":
        # 30 LOC — variant A logic
    elif self.variant == "B":
        # 25 LOC — variant B logic
    elif self.variant == "C":
        # 20 LOC — variant C logic
    # Modifying one variant risks breaking others
```

模式——签名一致的纯函数注册表：
```python
# variants/__init__.py
REGISTRY = {
    "A": compute_variant_a,
    "B": compute_variant_b,
    "C": compute_variant_c,
}

def get_variant_fn(name):
    return REGISTRY[name]
```

```python
# variants/variant_b.py — single file, single responsibility
def compute_variant_b(data, config_a, config_b, ...):
    """Variant B: brief description of what makes it different."""
    # ... self-contained implementation ...
    return result, metrics_dict
```

新增一个变体 = 添加一个文件 + 一条注册表条目。对现有变体零风险。

**oxRL 示例**：三个损失函数（SGRPO、GSPO、CISPO）各自独占 `oxrl/algs/losses/` 下的一个文件，通过 `LOSS_REGISTRY` dict 分发。

---

## 模式 4：工具化

将确定性逻辑提取为独立的、经过测试的纯函数工具。如果两个类共用同一段工具代码，它必须是一个独立工具，而非重复的方法。

反模式——确定性逻辑埋在类里面：
```python
class AlgorithmA:
    def _merge_weights(self, state_dict):
        # 42 LOC of linear algebra using self.config
        # AlgorithmB duplicates this exact same method
```

模式——独立工具，不依赖类状态：
```python
# tools/weight_merge.py
def merge_weights(state_dict, alpha, rank):
    """Merge adapter weights into base weights.

    TOOL CONTRACT:
        Input:  state_dict with base + adapter weights, scaling params
        Output: new state_dict with merged weights, adapter keys removed
        Side effects: None
        Deterministic: same input -> same output
    """
    # ... pure function ...
    return new_state_dict
```

每个工具契约需声明：
- **输入与输出**：明确的类型和语义
- **确定性**：相同输入永远产生相同输出
- **无副作用**：不修改全局状态或类属性
- **可独立测试**：无需实例化任何类即可进行单元测试

**oxRL 示例**：LoRA 权重合并、checkpoint 保存、张量填充、配置提取、logprob 解析、奖励归一化——全部从类中提取到 `oxrl/tools/`。

---

## 模式 5：编排器文件即菜谱

顶层入口是一个精简的编排器，读起来像一份菜谱。所有逻辑都在它调用的各阶段中。

反模式——595 LOC 的主文件，8 种关注点交织在一起：
```python
def main(config_file):
    # initialization ... config loading ... resource creation ...
    # data preparation ... core processing ... metric aggregation ...
    # state saving ... resource refresh ... logging ... all in one function
```

模式——编排器调用命名阶段：
```python
# main.py — ~190 LOC, pure orchestration
from myapp.setup.init import setup_infrastructure
from myapp.setup.factory import create_engines
from myapp.loops.process_phase import run_processing
from myapp.loops.save_phase import save_and_refresh

def main(config_file):
    config = load_config(config_file)
    infra = setup_infrastructure(config)
    engines = create_engines(config, infra)

    for epoch in range(config.num_epochs):
        results = run_processing(engines, data, epoch)   # Phase 1
        metrics = run_optimization(engines, results)      # Phase 2
        save_and_refresh(engines, config, epoch)          # Phase 3
```

每个阶段是一个独立文件（60-100 LOC）。修改保存逻辑，只需读 `save_phase.py`。编排器用约 40 LOC 展示完整流水线。

**oxRL 示例**：`main_rl.py`（595 LOC）拆分为 `oxrl/setup/`（3 个文件）+ `oxrl/loops/`（3 个文件）+ 一个 190 LOC 的编排器。

---

## 模式 6：Schema 与逻辑分离

数据定义（有哪些字段）必须与行为（如何处理这些字段）分开。

反模式——483 LOC 文件混合了 schema + 逻辑 + I/O：
```python
# config.py — everything in one file
class DatabaseConfig(BaseModel): ...    # schema
class ServerConfig(BaseModel): ...      # schema
class AppConfig(BaseModel):             # schema
    def sync_settings(self, env):       # logic
        self._sync_pool_sizes(env)      # logic
def load_config(path, env): ...         # I/O
```

模式——分文件、分关注点：
```
configs/
├── schema.py    # Pydantic models only, zero logic
├── sync.py      # Config synchronization / derivation logic
└── loader.py    # File I/O + verification
```

添加配置字段：只需读 `schema.py`。调试同步逻辑：只需读 `sync.py`。每个关注点都控制在 800 LOC 以内。

**oxRL 示例**：`oxrl/configs/load.py`（483 LOC）拆分为 `schema.py`（288 LOC）+ `sync.py`（150 LOC）+ `loader.py`（65 LOC）。

---

## 模式 7：零幻觉契约

使用校验器防止无效配置。剥夺 LLM 在基础设施参数上的自由发挥空间。

反模式——静默默认值允许无效配置：
```python
class WorkerConfig(BaseModel):
    num_workers: int = 8     # What if someone sets 0? Silent failure.
    timeout_sec: int = 30    # What if negative? Undefined behavior.
```

模式——为每个字段设置显式边界：
```python
class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # reject unknown fields
    num_workers: int = Field(default=8, ge=1, le=64)
    timeout_sec: int = Field(default=30, gt=0, le=3600)
    memory_fraction: float = Field(default=0.5, gt=0.0, le=0.95)
```

- `extra='forbid'` 阻止幻觉字段名
- `Field(ge=1)` 阻止无意义的值
- Pydantic 会立即抛出 `ValidationError`——不会有下游的静默失败

---

## 模式 8：Dict 分发取代工厂模式

用普通 dict 做路由。不用 AbstractFactory，不用 StrategyProvider，不用 RegistryManager。

模式：
```python
ALGORITHMS = {
    "variant_a": ClassA, "variant_b": ClassA,
    "variant_c": ClassB,
}
alg = ALGORITHMS[config.algorithm_name.lower()]
```

```python
# Dynamic loading via config string
module = importlib.import_module("myapp.strategies")
strategy_fn = getattr(module, config.strategy_name)
```

LLM 读到 dict 就能立刻知道所有合法选项。无需追踪任何间接调用。

---

## 模式 9：带有可量化结果的反馈循环

每个处理阶段返回结构化结果。下游阶段可以检查并响应。永远不要只记录指标而不对异常采取行动。

反模式——处理循环盲目运行：
```python
for step in range(n_steps):
    metrics = compute(batch)
    logger.info("loss=%.4f", metrics['loss'])
    # Logged... but never acted on.
    # loss=NaN -> continues blindly until crash
    # accuracy=0.0 for 100 steps -> no warning
```

模式——检测并标记退化：
```python
for step in range(n_steps):
    metrics = compute(batch)

    if math.isnan(metrics['loss']):
        logger.error("NaN loss at step %d — halting", step)
        break

    if metrics['grad_norm'] > 100.0:
        logger.warning("grad_norm=%.1f — potential instability", metrics['grad_norm'])

    if step > 50 and metrics['primary_metric'] < threshold:
        logger.warning("Primary metric below threshold for %d steps", step)
```

在多个层级设计反馈：
```
MICRO (per step):   NaN detection, gradient explosion, metric collapse -> halt or warn
MACRO (per epoch):  Stagnation detection, divergence trends            -> suggest config changes
SYSTEM (per run):   OOM, timeout, crash                                -> auto-retry with adjusted params
```

---

## 总结检查清单

编写或审查代码时，逐项确认：

- [ ] 没有文件超过 800 LOC
- [ ] 每个文件只有一个职责
- [ ] 每个模块顶部都有调用规格（输入、输出、副作用）
- [ ] 算法变体使用纯函数注册表，而非 if-else 链
- [ ] 确定性逻辑已提取为带契约的独立工具
- [ ] 工具保证：确定性、无副作用、可独立测试
- [ ] 顶层入口是不超过 200 LOC 的精简编排器
- [ ] 数据 schema 与行为逻辑分离
- [ ] 配置字段有显式边界（`Field(ge=..., le=...)`）
- [ ] 所有配置模型都设置了 `extra='forbid'` 以拒绝未知字段
- [ ] 路由使用普通 dict 或 `importlib`，而非工厂/策略模式
- [ ] 处理循环能检测异常（NaN、发散、停滞）并采取行动
- [ ] 类之间没有重复逻辑——提取为共享工具
