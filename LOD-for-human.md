# LLM-Oriented Design Patterns

**Subtitle:** Abandon Human Readability, Reforge the Laws of Gears, Feedback Loops, and Tools

---

## Table of Contents

- [Prologue: A Species-Level Shift in Code Readers](#prologue)
- [Part I: Context Management — From Monoliths to Micro-Gears](#part-i-context-management)
  - [Chapter 1: Radical Fragmentation](#chapter-1-radical-fragmentation)
  - [Chapter 2: Dynamic Assembly & In-Context Emergence](#chapter-2-dynamic-assembly)
  - [Chapter 3: Black-Boxing & Semantic Deflation](#chapter-3-black-boxing)
- [Part II: Multi-Dimensional Feedback Loops](#part-ii-feedback-loops)
  - [Chapter 4: Top-Down Goal Decomposition](#chapter-4-goal-decomposition)
  - [Chapter 5: Evolutionary Gear Orchestration](#chapter-5-gear-orchestration)
  - [Chapter 6: Multi-Tier Feedback Network](#chapter-6-feedback-network)
- [Part III: Toolification — Strip Probability, Anchor Deterministic Boundaries](#part-iii-toolification)
  - [Chapter 7: Probabilistic vs Deterministic](#chapter-7-probabilistic-vs-deterministic)
  - [Chapter 8: Infrastructure as LLM Tools](#chapter-8-infrastructure-as-tools)
  - [Chapter 9: Zero-Hallucination Contracts](#chapter-9-zero-hallucination-contracts)
- [Epilogue: From Code Writer to System Shepherd](#epilogue)
- [Appendix: Full Before/After File Map](#appendix)

---

## Prologue

### A Species-Level Shift in Code Readers

In the near future, reading source code line by line will become a relic of the pre-industrial era. The standard development workflow will be:

> Drop a GitHub repo link to a large model → Describe business intent → Let AI autonomously understand and integrate.

The **first reader** of code officially shifts from carbon-based humans to silicon-based models.

### The "Context Poison" of Traditional Design

Object-Oriented Programming, SOLID principles, and microservices were designed around **human cognitive bandwidth** and short-term memory. They emphasize cohesion, deep inheritance, and human-readable naming. But in the eyes of large models, this creates:

- **Bloated modules** — Classes that exceed the model's optimal intelligence window
- **Deep dependency chains** — Must load 5+ files to understand a single method
- **Invisible indirection** — Factory, Registry, and Strategy patterns that obscure the actual call graph

These problems are universal. They appear in web frameworks, data pipelines, ML training systems, CLI tools, and enterprise backends alike. Any codebase with classes over 500 line-of-code (LOC), deep inheritance, or factory-of-factory patterns is hostile to LLM comprehension.

### The "Long-Context Intelligence Collapse"

Even with 1M-token context windows, the **"Lost in the Middle"** effect persists. Overly long, human-friendly code dilutes model attention, leading to degraded reasoning and increased hallucinations.

### The New Architectural Manifesto

> **The first principle of software design must shift — from Human-Friendly to LLM-Friendly.**

We must sacrifice intuitive human readability to maximize model parsing, assembly, and execution efficiency.

### The Case Study: oxRL

We are using ([https://github.com/warlockee/oxRL](https://github.com/warlockee/oxRL)) as a case study to demonstrate these principles. oxRL is a post-training framework for LLMs implementing 18 algorithms across RL and SL paths. It exhibits the same architectural debt found in any mature codebase: god classes, monolithic entry points, duplicated utilities, and coupled concerns. In its [v1.0 form](https://github.com/warlockee/oxRL/tree/LOD-example-v1.0), it already makes some LLM-friendly choices — but still carries significant debt that degrades AI comprehension.

Throughout this book, we show the [v1.0 form](https://github.com/warlockee/oxRL/tree/LOD-example-v1.0) as "Before" and provide [v2.0 form](https://github.com/warlockee/oxRL/tree/LOD-example-v2.0) as "After" for every principle. The patterns themselves apply to any codebase — oxRL just makes them concrete.

---

## Part I: Context Management

**Thesis:** Design must dance within AI's attention and compute limits. Context compression is the supreme priority.

### The Golden Constraint

> Each micro-gear's code and dependencies must strictly fit within the model's "optimal intelligence window" — **4k to 8k tokens** (~200–800 LOC) — ensuring full-lossless understanding.

This is not an RL-specific rule. It applies to any module in any system: a React component, a database migration tool, an API handler, or a training algorithm.

---

### Chapter 1: Radical Fragmentation

> *Shatter monolithic classes and deep call chains into absolutely single-responsibility, flat micro-gears.*

The core problem: a class with N responsibilities forces the LLM to load all N into context, even when it only needs to modify one. After splitting, modifying responsibility #3 means reading only the file that contains #3.

#### Violation 1: The God Class

Any class that mixes orchestration, I/O, math, and state management in a single file exceeding 500 LOC is a god class. The LLM must wade through irrelevant concerns to reach the one it needs.

**oxRL example — `oxrl/algs/grpo.py` — 665 LOC, 7 responsibilities:**

```python
@ray.remote
class GRPO(BaseAlgorithm):
    def __init__(self, ...):              # 80 LOC — stores 15+ attributes
        self.init_training_engine()       # calls load_model + LoRA + DeepSpeed + optimizer

    def init_training_engine(self):       # 72 LOC — responsibility: engine setup
        deepspeed.init_distributed()
        model, ref_model = self.load_model()
        # LoRA application (20 LOC)
        # Optimizer construction (10 LOC)
        # DeepSpeed initialize (15 LOC)
        # Ref model DeepSpeed initialize (10 LOC)

    def load_model(self):                 # 8 LOC — responsibility: model I/O

    def ref_forward(self, ...):           # 25 LOC — responsibility: reference inference

    def policy_forward(self, ...):        # 43 LOC — responsibility: policy inference

    def compute_kl_distance(self, ...):   # 12 LOC — responsibility: KL math

    def compute_policy_loss(self, ...):   # 106 LOC — responsibility: loss math (3 variants!)

    def train_step(self, ...):            # 72 LOC — responsibility: training orchestration

    def _get_base_model_config(self):     # 11 LOC — responsibility: config extraction

    def _strip_lora_and_merge(self, ...): # 42 LOC — responsibility: weight merging

    def gather_state_dict(self):          # 30 LOC — responsibility: distributed gathering

    def save_checkpoint(self, ...):       # 98 LOC — responsibility: checkpoint I/O
```

**What the LLM sees:** To understand how GRPO computes its loss, the model must load 665 LOC (~2,700 tokens) into context. But it only needs `compute_policy_loss` (106 LOC). The other 559 LOC are noise that dilutes attention.

**Worse:** The `compute_policy_loss` method contains **three completely different algorithms** (SGRPO, GSPO, CISPO) behind if-else branches. An LLM modifying one variant must hold all three in attention simultaneously.

**AFTER (v2.0) — Shatter into 6 micro-gears:**

```
oxrl/algs/
├── base.py                    # 41 LOC — interface contract (unchanged)
├── losses/
│   ├── sgrpo_loss.py          # ~40 LOC — variant A
│   ├── gspo_loss.py           # ~35 LOC — variant B
│   └── cispo_loss.py          # ~30 LOC — variant C
├── grpo.py                    # ~250 LOC — orchestration only
│   - __init__: store params, call setup_engine()
│   - train_step: data prep → forward → loss_fn() → backward
│   - policy_forward / ref_forward: model calls
├── engine_setup.py            # ~100 LOC — engine initialization
│   - setup_training_engine(model_path, ds_config, lora_config) → engine
│   - setup_ref_engine(ref_model_path, ds_ref_config) → engine
├── weight_tools.py            # ~80 LOC — checkpoint + weight merging
│   - strip_lora_and_merge(state_dict, lora_config) → state_dict
│   - gather_state_dict(engine) → state_dict
│   - save_checkpoint(state_dict, output_dir) → None
└── kl.py                      # ~15 LOC — divergence computation
    - compute_kl_distance(logprobs, ref_logprobs) → tensor
```

**Why this is better for LLMs:**

| Before (v1.0) | After (v2.0) | LLM Impact |
|---|---|---|
| Modify variant B → read 665 LOC | Modify variant B → read 35 LOC | **19x less context** |
| Add new variant → edit 106-LOC method | Add new variant → create new 35 LOC file | **Zero risk of breaking existing variants** |
| Debug checkpoint saving → read 665 LOC | Debug checkpoint saving → read 80 LOC | **8x less context** |
| Weight merging coupled to one class | Weight merging is a standalone tool | **Reusable by any algorithm** |

**Concrete v2.0 extracted variant — `gspo_loss.py`:**

```python
# oxrl/algs/losses/gspo_loss.py — 35 LOC, single responsibility
import torch

def gspo_loss(logprobs, old_logprobs, advantages, mask, clip_low, clip_high):
    """Sequence-level clipped surrogate loss for MoE models.

    Averages log-ratios across the sequence before clipping, so
    token-level MoE routing noise cancels out.

    Args:
        logprobs:      [B, T-1] current policy log-probs
        old_logprobs:  [B, T-1] old policy log-probs
        advantages:    [B, T-1] group-normalized advantages
        mask:          [B, T-1] response token mask
        clip_low:      float, lower clip bound (e.g. 0.2)
        clip_high:     float, upper clip bound (e.g. 0.2)

    Returns:
        loss:     scalar loss
        metrics:  dict with clipfrac, approx_kl
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

35 LOC. ~140 tokens. An LLM can read this, understand it, modify it, and verify its correctness — all within a single attention span.

---

#### Violation 2: The Monolithic Entry Point

Any main script that mixes initialization, processing, metric collection, state persistence, and cleanup in one file is hostile to LLMs. The model must hold 8+ concerns in context to modify any single one.

**oxRL example — `main_rl.py` — 595 LOC, 8+ concerns in one script:**

```python
# main_rl.py — a single file doing everything
def setup_ray(ray_address): ...              # 20 LOC — infrastructure init
def training_engine_setup(params, ...): ...  # 56 LOC — actor spawning
def rollout_engine_setup(params, ...): ...   # 52 LOC — inference actor spawning
def rollout_dataloader_setup(params, ...):   # 25 LOC — data loading
def collect_rollouts(...): ...               # 71 LOC — generation orchestration
def main(config_file, experiment_id):        # 330 LOC — processing loop + persistence + refresh
    # Everything interleaved:
    # - config loading
    # - infrastructure init
    # - engine creation
    # - tokenizer loading
    # - scoring function import
    # - data generation
    # - buffer management
    # - processing step dispatch
    # - metric aggregation and logging
    # - state saving (2 code paths!)
    # - engine refresh (2 code paths!)
    # - experiment tracking
```

**AFTER (v2.0) — Split into focused micro-gears:**

```
main_rl.py                    # ~120 LOC — pure orchestration
                              #   main() calls: setup → loop → cleanup

oxrl/setup/
├── ray_setup.py              # ~30 LOC — infrastructure initialization
├── engine_factory.py         # ~60 LOC — spawn processing actors
└── dataloader_factory.py     # ~30 LOC — create data pipeline

oxrl/loops/
├── rollout_phase.py          # ~60 LOC — data generation phase
├── train_phase.py            # ~50 LOC — dispatch processing, aggregate metrics
└── checkpoint_phase.py       # ~80 LOC — gather → save → refresh (both paths)
```

**v2.0 `main_rl.py` — Pure orchestration, ~120 LOC:**

```python
# main_rl.py v2.0 — the orchestrator reads like a recipe
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
        # Phase 1: Generate data
        rollout_stats = collect_rollouts(rollout_dataloader, rollout_engines, epoch, policy_version, replay_buffer)

        # Phase 2: Process data
        epoch_metrics = run_training_steps(training_engines, replay_buffer, config, epoch)

        # Phase 3: Save state + refresh engines
        policy_version = save_and_refresh(training_engines, rollout_engines, config, epoch, policy_version, tokenizer)

        # Reset for next epoch
        replay_buffer.reset()
```

An LLM reading this file sees the **entire pipeline** in 40 LOC. To modify any phase, it reads only the relevant 50-80 LOC file. No attention wasted on unrelated phases.

---

#### Violation 3: The Multi-Concern Service

Any service class that mixes lifecycle management, parameter construction, data parsing, core processing, and post-processing in one file is a multi-concern service. Each concern should be extractable as a pure function or a focused module.

**oxRL example — `oxrl/rollouts/vllm_engine.py` — 565 LOC, 5 responsibilities:**

```python
@ray.remote
class VLLMRolloutEngine:
    # Concern 1: Model lifecycle (load, refresh, refresh_from_state_dict)  ~140 LOC
    # Concern 2: Sampling parameter construction + validation              ~50 LOC
    # Concern 3: Output parsing (logprob extraction)                       ~40 LOC
    # Concern 4: Core processing (generation + sample construction)        ~120 LOC
    # Concern 5: Post-processing (scoring + normalization)                 ~45 LOC
```

**AFTER (v2.0) — Extract pure functions, keep the service thin:**

```
oxrl/rollouts/
├── vllm_engine.py             # ~200 LOC — service: lifecycle + core processing
├── sampling.py                # ~50 LOC  — make_sampling_params() + validation
├── logprob_utils.py           # ~40 LOC  — extract_logprobs() pure function
├── reward_scoring.py          # ~30 LOC  — score_response() pure function
├── normalization.py           # ~40 LOC  — normalize_rewards() pure function
└── replay_buffer.py           # ~208 LOC — unchanged (already focused)
```

**Key extraction — post-processing as a pure function:**

```python
# BEFORE: embedded inside VLLMRolloutEngine class (v1.0)
class VLLMRolloutEngine:
    def normalize_rewards(self, samples, stats, prompt_len, is_per_token):
        # Uses self.eps_reward_norm and self.reward_broadcast
        # Modifies samples in-place
        ...

# AFTER: standalone pure function (v2.0)
# oxrl/rollouts/normalization.py
def normalize_rewards(samples, stats, prompt_len, is_per_token, eps=1e-8, broadcast=False):
    """Z-score normalize rewards within a sample group.

    Args:
        samples:      list of result dicts for one input
        stats:        {"rewards": [float], "lengths": [int]}
        prompt_len:   int, length of the input prefix
        is_per_token: bool, whether scores are per-token
        eps:          float, normalization epsilon
        broadcast:    bool, whether to broadcast scalar score across all tokens

    Returns:
        None (modifies samples in-place)
    """
    ...
```

Now an LLM can test normalization in isolation, reuse it in a different service, or replace it — without touching the service class.

---

#### Violation 4: Schema + Logic + I/O Coupled

Any configuration file that mixes data definitions (schema), derivation logic (sync/compute), and file I/O (loading) in one module forces the LLM to read everything regardless of which concern it needs.

**oxRL example — `oxrl/configs/load.py` — 483 LOC, 3 distinct concerns:**

```python
# Concern 1: Config SCHEMA (Pydantic models) — ~275 LOC
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

# Concern 2: Config LOGIC (sync + derivation) — ~150 LOC
class Config(BaseModel):
    def sync_deepspeed_config(self, world_size):
        self._sync_batch_sizes(world_size)
        self._sync_gradient_clipping()
        self._sync_dtype()
        self._sync_optimizer()
        self._sync_scheduler()
        self._sync_zero_defaults()
        self._sync_ref_model_config()

# Concern 3: Config LOADING (file I/O) — ~50 LOC
def load_and_verify(method, input_yaml, experiment_id, world_size=None): ...
```

**AFTER (v2.0) — Separate schema from logic:**

```
oxrl/configs/
├── schema.py                 # ~200 LOC — pure Pydantic models, zero logic
│   class Run, Train, Model, Data, DeepSpeed, Lora, Reward, Rollout, Config
│
├── sync.py                   # ~150 LOC — config synchronization / derivation
│   def sync_deepspeed_config(config, world_size) → None
│   def sync_batch_sizes(config, world_size) → None
│   def sync_dtype(config) → None
│   ...
│
└── loader.py                 # ~50 LOC — YAML loading + verification
    def load_and_verify(method, input_yaml, experiment_id, world_size) → Config
```

**Why:** An LLM adding a new config field only needs `schema.py` (200 LOC). An LLM debugging sync logic only needs `sync.py` (150 LOC). Currently it must read all 483 LOC regardless of what it is doing.

---

### Chapter 2: Dynamic Assembly

> *Humans stop writing rigid glue code. The model gains dynamic assembly rights.*

The principle: use flat dispatch (dicts, importlib) instead of rigid wiring or factory patterns. When the full set of options is visible in a single dict, the LLM immediately knows what's available without tracing indirection.

#### What v1.0 Gets Right

oxRL v1.0 already uses a plain dict for algorithm dispatch instead of a Factory pattern:

```python
# main_rl.py — GOOD: explicit dictionary, zero indirection
RL_ALGORITHMS = {"sgrpo": GRPO, "cispo": GRPO, "gspo": GRPO, "rlhf": GRPO, "rlaif": GRPO, "ppo": PPO}
alg = RL_ALGORITHMS[alg_name]
```

And dynamic function loading via config:

```python
reward_module = importlib.import_module("oxrl.rewards")
reward_fnc = getattr(reward_module, config.reward.reward_func)
```

These are already LLM-native. **Keep them as-is.**

#### What v1.0 Gets Wrong: Rigid Engine Wiring

**BEFORE (v1.0) — Hardcoded parameter extraction in the caller:**

```python
# main_rl.py — training_engine_setup() builds kwargs manually
def training_engine_setup(params, alg, world_size, master_addr, master_port):
    kwargs = {
        'model_path': params.model.name,
        'ref_model_path': params.model.ref_model,
        'model_dtype': safe_string_to_torch_dtype(params.model.dtype),
        # ... 20+ kwargs manually extracted from config ...
        'loss_variant': params.train.alg_name.lower(),
        'lr': params.train.lr,
        'betas': params.train.betas,
    }
    # Different algorithms need different kwargs
    if params.train.alg_name.lower() == "ppo":
        kwargs['vf_clip'] = params.train.ppo_vf_clip
        kwargs['tau'] = params.train.ppo_tau
        kwargs['gamma'] = params.train.ppo_gamma
    ...
```

This is **rigid wiring**. Every time a new variant needs a new parameter, this function must be manually updated. The kwargs dict is an implicit contract that can silently break if the constructor changes.

**AFTER (v2.0) — Config-driven assembly:**

```python
# oxrl/setup/engine_factory.py v2.0 — config IS the contract
def create_training_engines(config, master_addr):
    """Each variant receives the full config and extracts what it needs."""
    alg_cls = RL_ALGORITHMS[config.train.alg_name.lower()]

    engines = []
    for rank in range(config.run.training_gpus):
        env_vars = _build_env_vars(master_addr, config.run.ray_master_port, rank, config.run.training_gpus)
        engine = alg_cls.options(num_gpus=1, runtime_env={"env_vars": env_vars}).remote(config=config)
        engines.append(engine)
    return engines
```

```python
# oxrl/algs/grpo.py v2.0 — each variant extracts what it needs from config
@ray.remote
class GRPO(BaseAlgorithm):
    def __init__(self, config):
        self.model_path = config.model.name
        self.loss_fn = LOSS_FUNCTIONS[config.train.alg_name.lower()]
        self.clip_low = config.train.clip_low
        self.clip_high = config.train.clip_high
        # ...extract only what's needed...
```

**Why:** In v1.0, adding a variant-specific parameter requires editing both the caller AND the callee. In v2.0, adding a parameter means editing only the callee — the config already carries it.

---

### Chapter 3: Black-Boxing & Semantic Deflation

> *Once a large gear is assembled and validated, immediately seal it as a black box with an ultra-minimal calling document.*

The principle: every module should have a **calling spec** at the top — a compact summary of its interface (methods, inputs, outputs, side effects) — so the LLM never needs to read the implementation to use it. This is semantic deflation: compressing 500+ LOC of implementation into ~30 lines of spec.

#### What v1.0 Gets Right

The `BaseAlgorithm` abstract class (41 LOC) is already a good black-box spec:

```python
class BaseAlgorithm(ABC):
    def is_ready(self) -> bool: ...
    def train_step(self, *args, **kwargs) -> Dict[str, float]: ...
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None): ...
    def gather_state_dict(self) -> Optional[dict]: ...
```

And scoring functions follow a uniform pure-function signature.

#### What v1.0 Gets Wrong: No Calling Specs for Services

The VLLMRolloutEngine has **no calling spec**. To use it, you must read 565 LOC to understand:
- What methods exist
- What arguments they take
- What they return
- What side effects they have

**BEFORE (v1.0) — No spec, must read source:**

```python
# To call VLLMRolloutEngine.generate(), you must figure out:
# - What format does `prompts` expect?
# - What does `current_iter` do?
# - What does `policy_version` do?
# - What is the return format?
# Answer: read 120 LOC of generate() + 40 LOC of helper methods = 160 LOC
```

**AFTER (v2.0) — Add explicit calling specs as module-level docstrings:**

```python
# oxrl/rollouts/vllm_engine.py v2.0 — Black-box spec at the top
"""VLLMRolloutEngine — Inference engine for generating data samples.

CALLING SPEC (for LLM agents):
    engine = VLLMRolloutEngine.remote(config=config, reward_func=fn, eos_id=id)

    # Refresh weights (2 methods):
    engine.refresh_model.remote(model_path, version)           → bool
    engine.refresh_model_from_state_dict.remote(sd, cfg, ver)  → bool

    # Generate samples:
    engine.generate.remote(
        prompts=[{"prompt_token_ids": [int, ...], "metadata": {...}}, ...],
        current_iter=int,
        policy_version=int,
    ) → List[Dict]:
        Each dict contains:
          "input_ids":          Tensor[T]    — input + output concatenated
          "rewards":            Tensor[T]    — per-token scores (0 on input)
          "pred_zscores":       Tensor[T]    — z-scored advantages
          "pred_masks":         Tensor[T]    — 1 on output predictions
          "pred_old_logprobs":  Tensor[T]    — prediction-aligned log-probs
          "response_len":       int          — number of output tokens

TOKEN BUDGET: ~200 LOC after refactoring (was 565 LOC).
"""
```

Now an LLM reads the spec (~30 lines, ~120 tokens) and knows exactly how to use the engine **without reading any implementation code**. This is semantic deflation — 565 LOC compressed to 120 tokens.

#### Adding Specs Everywhere

**v2.0 adds calling specs to every module:**

```python
# oxrl/rollouts/replay_buffer.py v2.0 — spec
"""ReplayBuffer — Stores generated samples for iterative processing.

CALLING SPEC:
    buffer = ReplayBuffer(pad_token_id=int, max_seq_len=int)
    buffer.add_batch_seqs(samples: List[Dict])  — add samples
    buffer.reset()                               — clear for next iteration
    len(buffer)                                  → int
    DataLoader(buffer, collate_fn=buffer.collate_fn)  → yields batches:
        {"input_ids": [B,T], "attn_mask": [B,T], "old_logprobs": [B,T],
         "mask": [B,T], "rewards": [B,T], "zscore": [B,T], "v_olds": [B,T]|None}
"""
```

```python
# oxrl/algs/losses/gspo_loss.py v2.0 — spec
"""GSPO Loss — Sequence-level clipped surrogate for MoE models.

CALLING SPEC:
    loss, metrics = gspo_loss(logprobs, old_logprobs, advantages, mask, clip_low, clip_high)
    # loss:    scalar tensor
    # metrics: {"clipfrac": float, "approx_kl": float}
"""
```

The **recursive compression** principle: at each level, the LLM reads only the spec from the level below:

```
Level 0: variant_fn spec       → "loss, metrics = variant_fn(inputs, ...)"     ~20 tokens
Level 1: engine.step spec      → "metrics = engine.train_step(eid, batches)"   ~20 tokens
Level 2: phase spec            → "epoch_metrics = run_processing_steps(...)"   ~15 tokens
Level 3: main.py               → "calls phase_1 → phase_2 → phase_3"          ~10 tokens
```

Total context to understand the entire pipeline: **~65 tokens** instead of **~10,500 tokens**.

---

## Part II: Feedback Loops

**Thesis:** Static code topology is dead. AI-native systems evolve through layered feedback like digital organisms.

This part is about systems that can detect their own failures and adapt — not just log metrics, but act on them. The patterns apply to any iterative system: training loops, CI/CD pipelines, data processing pipelines, or autonomous agents.

---

### Chapter 4: Goal Decomposition

> *Entry points are high-level "North Star" objectives. The system decomposes them into measurable, testable sub-goals.*

The principle: every pipeline phase should return a structured result with success/failure status and measurable metrics. Downstream phases can inspect these results and adapt. This transforms a blind sequential pipeline into a self-aware system.

#### What v1.0 Gets Right

The swarm system already implements goal decomposition:

```python
# oxrl/swarm/scout.py — GOOD: clear sub-goal decomposition
def onboard_model(model_id, entry):
    discover_info = step_discover(model_id, entry)    # Sub-goal 1: verify inputs
    step_preprocess(dataset, model_slug)               # Sub-goal 2: prepare data
    config_path = step_generate_config(...)             # Sub-goal 3: generate config
    log_path = step_train(config_path, ...)             # Sub-goal 4: process
    eval_result = step_evaluate(log_path)               # Sub-goal 5: evaluate
    step_archive(...)                                   # Sub-goal 6: archive
    step_gc(model_id)                                   # Sub-goal 7: cleanup
    step_update_manifest(...)                           # Sub-goal 8: update state
```

Each sub-goal is measurable (raises `RuntimeError` on failure), testable (can be run independently), and independent (skip if already done).

#### What v1.0 Gets Wrong: The Processing Loop Has No Goal Decomposition

**BEFORE (v1.0) — `main_rl.py` main loop — a flat sequence with no sub-goals:**

```python
for epoch in range(number_of_epochs):
    # Data generation ... 20 LOC of inline logic
    rollout_stats = collect_rollouts(...)

    # Data prep ... 30 LOC of inline logic
    train_batches = list(DataLoader(...))
    # Padding logic ... 10 LOC
    train_batches_padded = ...

    # Processing ... 40 LOC of inline dispatch + metric collection
    for tidx in range(number_of_training_steps_per_epoch):
        train_futures = []
        for eid, engine in enumerate(training_engine_runners):
            shard = train_batches_padded[eid::num_train_engines]
            train_futures.append(engine.train_step.remote(...))
        train_metrics = ray.get(train_futures)
        # 10 LOC of metric aggregation
        # 5 LOC of logging

    # State persistence ... 60 LOC with 2 different code paths
    try:
        gather_futures = [engine.gather_state_dict.remote() for engine in training_engine_runners]
        gather_results = ray.get(gather_futures)
        state_dict = next((r for r in gather_results if r is not None), None)
    except:
        state_dict = None

    if state_dict is not None:
        # Fast path: object store ... 30 LOC
    else:
        # Legacy path: disk-based ... 15 LOC
```

There are no named sub-goals. No measurable outcomes per phase. No ability for the system to detect "processing is failing" and adapt.

**AFTER (v2.0) — Named sub-goals with measurable outcomes:**

```python
# main_rl.py v2.0 — each phase returns a measurable result
for epoch in range(number_of_epochs):
    # Sub-goal 1: Generate data → measure output quality
    rollout_result = rollout_phase(rollout_dataloader, rollout_engines, epoch, policy_version, replay_buffer)
    assert rollout_result.total_samples > 0, "No samples generated"

    # Sub-goal 2: Process data → measure convergence
    train_result = train_phase(training_engines, replay_buffer, config, epoch)
    assert not math.isnan(train_result.avg_loss), "NaN loss detected"

    # Sub-goal 3: Save state + refresh → measure success
    checkpoint_result = checkpoint_phase(training_engines, rollout_engines, config, epoch, policy_version, tokenizer)
    assert checkpoint_result.success, f"Checkpoint failed: {checkpoint_result.error}"

    # Sub-goal 4: Health check → detect degradation
    health = health_check(rollout_result, train_result, epoch)
    if health.primary_metric_stalled:
        logger.warning("Primary metric stalled for 5 iterations — consider adjusting parameters")
    if health.divergence_detected:
        logger.warning("Divergence detected — consider reducing step size")

    policy_version += 1
    replay_buffer.reset()
```

Each sub-goal returns a **structured result** that downstream phases can inspect. The `health_check` is a new feedback mechanism that does not exist in v1.0.

---

### Chapter 5: Gear Orchestration

> *The system decides which gears to reuse, which to retire, and which new gears to forge.*

The principle: when a system encounters failure, it should have a graduated response — not just retry with the same parameters, but escalate through increasingly aggressive strategies: adjust parameters → swap components → enable new features → retire the task.

#### What v1.0 Gets Right

The orchestrator + bugfixer pattern is already evolutionary:

```
Scout (attempt) → success → Onboard
                → failure → Bugfixer (classify + fix) → Re-queue → Scout again
                                                      → Skip (retire)
```

#### What v1.0 Gets Wrong: The Bugfixer Only Adjusts Parameters, Never Components

**BEFORE (v1.0) — Bugfixer applies parameter patches only:**

```python
# oxrl/swarm/bugfixer.py — LIMITATION: only changes config values
def _fix_oom(config_path):
    changes["train.train_batch_size_per_gpu"] = max(1, train_bs // 2)
    changes["rollout.rollout_batch_size_per_gpu"] = max(1, rollout_bs // 2)
    return {"action": "adjust_config", "changes": changes}

def _fix_nan_loss(config_path):
    changes["train.lr"] = current_lr / 10.0
    changes["train.clip_grad_norm"] = 5.0
    return {"action": "adjust_config", "changes": changes}
```

The bugfixer can halve batch sizes and reduce step sizes. But it cannot:
- Switch algorithm variants when the current one is unsuitable
- Enable memory-efficient adapters when the system OOMs at minimum batch size
- Change scoring functions when scores are always zero
- Add memory optimization techniques when resources are tight

**AFTER (v2.0) — Bugfixer can forge new gears (swap components, enable adapters, etc.):**

```python
# oxrl/swarm/bugfixer.py v2.0 — expanded fix taxonomy
def _fix_oom(config_path):
    cfg = _load_config_yaml(config_path)
    train_bs = cfg.get("train", {}).get("train_batch_size_per_gpu", 1)

    # Level 1: Reduce batch sizes (parameter adjustment)
    if train_bs > 1:
        return {"action": "adjust_config",
                "changes": {"train.train_batch_size_per_gpu": max(1, train_bs // 2)}}

    # Level 2: Enable adapter (FORGE NEW GEAR — swap component)
    if not cfg.get("lora", {}).get("enabled", False):
        return {"action": "adjust_config",
                "changes": {"lora.enabled": True, "lora.r": 16, "lora.lora_alpha": 32},
                "reason": "OOM at batch_size=1 — enabling adapter to reduce parameters"}

    # Level 3: Enable memory optimization (FORGE NEW GEAR)
    if not cfg.get("model", {}).get("gradient_checkpointing", False):
        return {"action": "adjust_config",
                "changes": {"model.gradient_checkpointing": True}}

    # Level 4: CPU offload (existing)
    ...

    # Level 5: Retire
    return {"action": "skip", "reason": "OOM with all mitigations exhausted"}

def _fix_score_zero():
    """NEW in v2.0: Swap scoring function when scores are always 0."""
    # Analyze dataset metadata to determine correct function
    if "gsm8k" in dataset:
        return {"action": "adjust_config",
                "changes": {"reward.reward_func": "soft_math_reward_func"},
                "reason": "Score always 0 — switching to partial-credit function"}
    ...
```

The v2.0 bugfixer can **forge new gears** (enable adapters, switch scoring functions) and **retire gears** (skip permanently), not just adjust parameters.

---

### Chapter 6: Multi-Tier Feedback Network

> *Micro loops self-repair step-level errors. Macro loops adjust system strategy.*

The principle: feedback should operate at multiple time scales. Per-step checks catch catastrophic failures immediately. Per-iteration checks detect degradation trends. Per-run checks handle infrastructure failures. Most systems only implement the outermost loop.

#### What v1.0 Gets Wrong: No Micro-Level Self-Repair

**BEFORE (v1.0) — Processing loop runs blindly, no health monitoring:**

```python
# main_rl.py — no detection of degraded processing
for tidx in range(number_of_training_steps_per_epoch):
    train_metrics = ray.get(train_futures)
    avg_loss = np.mean([m.get('loss_total', 0.0) for m in train_metrics])
    avg_kl_old = np.mean([m.get('kl_old', 0.0) for m in train_metrics])
    avg_clipfrac = np.mean([m.get('clipfrac', 0.0) for m in train_metrics])
    # Log it... but never ACT on it.
    # If all updates are being clipped → system learning nothing → continues blindly
    # If divergence metric is exploding → system is unstable → continues blindly
    # If loss = NaN → system is dead → continues blindly (until crash)
```

**AFTER (v2.0) — Micro-loop with automatic health detection:**

```python
# oxrl/loops/train_phase.py v2.0 — health-aware processing
def run_training_steps(training_engines, replay_buffer, config, epoch):
    epoch_metrics = defaultdict(list)

    for tidx in range(config.train.train_steps_per_epoch):
        train_metrics = ray.get(train_futures)
        step_metrics = aggregate_metrics(train_metrics)

        # MICRO FEEDBACK LOOP: detect and flag degradation
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
# oxrl/loops/health_check.py v2.0 — NEW: macro health monitoring
def health_check(rollout_result, train_result, epoch, history):
    """Detect system-level degradation patterns."""
    issues = []

    # Primary metric stagnation: hasn't improved in N iterations
    if len(history.primary_metric) >= 5:
        recent = history.primary_metric[-5:]
        if max(recent) - min(recent) < 0.01:
            issues.append(HealthIssue("metric_stalled",
                "Primary metric unchanged for 5 iterations — consider different parameters"))

    # Divergence: instability metric growing iteration-over-iteration
    if len(history.divergence_metric) >= 3:
        if all(history.divergence_metric[i] < history.divergence_metric[i+1] for i in range(-3, -1)):
            issues.append(HealthIssue("diverging",
                "Instability metric increasing — consider reducing step size"))

    # Output collapse: system producing degenerate outputs
    if rollout_result.avg_response_len < 10:
        issues.append(HealthIssue("output_collapse",
            "Average output length < 10 tokens — system may have collapsed"))

    return HealthResult(issues=issues)
```

This creates a **multi-tier feedback network**:

```
┌────────────────────────────────────────────────────────┐
│  MICRO LOOP (per processing step)                      │
│  ├── NaN detected          → halt iteration early      │
│  ├── Updates fully clipped → log warning               │
│  └── Divergence spike      → log warning               │
├────────────────────────────────────────────────────────┤
│  MACRO LOOP (per iteration)                            │
│  ├── Metric stalled        → suggest parameter change  │
│  ├── Divergence trend      → suggest step size change  │
│  └── Output collapse       → suggest config adjustment │
├────────────────────────────────────────────────────────┤
│  SYSTEM LOOP (across runs) — EXISTING                  │
│  ├── OOM → bugfixer reduces resource usage             │
│  ├── NaN → bugfixer reduces step size 10x              │
│  └── Timeout → bugfixer adjusts infrastructure config  │
└────────────────────────────────────────────────────────┘
```

v1.0 only has the system loop (bugfixer). v2.0 adds micro and macro loops that detect problems **during** processing, not just **after** a run fails.

---

## Part III: Toolification

**Thesis:** In a probabilistic world of generative models, deterministic walls must exist. Large software systems cannot roll dice on infrastructure.

This part is about the boundary between what AI decides (routing, parameters, strategies) and what must be deterministic (math, I/O, validation). Every codebase has this boundary — the patterns here help you draw it clearly.

---

### Chapter 7: Probabilistic vs Deterministic

> *Never allow the model to probabilistically implement high-risk foundational infrastructure logic.*

#### The Iron Law

| Component | Nature | Who Implements It |
|---|---|---|
| "Choose variant B for this workload" | Intent understanding, routing | LLM (probabilistic) |
| The variant B formula itself | Deterministic math | Human-written, tested, sealed |
| "Set step_size=5e-6 for this input" | Judgment, parameter selection | LLM (probabilistic) |
| Config validation (`extra='forbid'`) | Schema enforcement | Pydantic (deterministic tool) |
| "Enable adapter to fix OOM" | Diagnosis, strategy | Bugfixer agent (probabilistic) |
| Adapter weight merging `delta = B @ A * scaling` | Linear algebra | Human-written, tested, sealed |
| "Retry with smaller batch" | Recovery decision | Orchestrator (probabilistic) |
| File I/O for saving state | Infrastructure | Human-written, deterministic tool |

This boundary exists in every system: a web app has LLM-generated SQL (dangerous) vs parameterized queries (safe). A CI system has LLM-chosen test strategies (flexible) vs the test runner itself (deterministic). Draw the line. Seal the deterministic side.

#### What v1.0 Gets Right

The on-policy enforcement wall is already deterministic:

```python
# GOOD: Hard error, not a warning
if self.force_strict_on_policy:
    if self.temperature != 1.0:
        raise ValueError("Strict on-policy requires temperature = 1.0")
```

Pydantic config validation is already strict:

```python
# GOOD: extra='forbid' prevents hallucinated fields
class Train(BaseModel):
    model_config = ConfigDict(extra='forbid')
```

#### What v1.0 Gets Wrong: Deterministic Logic Buried Inside Classes

**BEFORE (v1.0) — Weight merging is a method of one algorithm class, not a standalone tool:**

```python
# oxrl/algs/grpo.py — weight merging is INSIDE the GRPO class
class GRPO(BaseAlgorithm):
    def _strip_lora_and_merge(self, state_dict):
        # 42 LOC of exact linear algebra
        # Uses self.lora_config (coupling to class state)
        for k in list(new_state_dict.keys()):
            scaling = alpha / r
            delta = (lb_w @ la_w) * scaling
            new_state_dict[k] = base_w + delta.to(base_w.dtype)
```

**Problem:** PPO also needs weight merging. In v1.0, PPO has a **duplicate** method. If a bug is found in the merging logic, it must be fixed in both places.

**AFTER (v2.0) — Weight merging as a standalone deterministic tool:**

```python
# oxrl/tools/lora_merge.py v2.0 — sealed, tested, reusable
def strip_lora_and_merge(state_dict, lora_alpha, lora_r):
    """Merge adapter weights into base model weights.

    DETERMINISTIC TOOL CONTRACT:
        Input:  state_dict with base weights + adapter weights
        Output: state_dict with merged weights, adapter keys removed
        Side effects: None (returns new dict)
        Reproducibility: Guaranteed (pure function)

    Args:
        state_dict:  dict of {name: tensor} — may contain PEFT prefixes
        lora_alpha:  int — scaling factor numerator
        lora_r:      int — rank (scaling factor denominator)

    Returns:
        dict of {name: tensor} — merged weights, no adapter keys
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

Now any algorithm calls `strip_lora_and_merge(state_dict, alpha, r)` — a single, tested, deterministic tool.

---

### Chapter 8: Infrastructure as LLM Tools

> *Identify deterministic components and convert them into independent Tools beyond the model's cognitive mutation.*

The principle: infrastructure logic (file I/O, distributed coordination, tensor manipulation, config extraction) should never be embedded inside business logic classes. Extract it into standalone tools that any class can import.

#### What v1.0 Gets Wrong: State Persistence Has Two Code Paths Buried in a Business Class

**BEFORE (v1.0) — `GRPO.save_checkpoint()` — 98 LOC with 2 code paths:**

```python
class GRPO(BaseAlgorithm):
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None):
        if state_dict_ref is not None:
            # Fast path: write pre-gathered state dict from object store
            if rank == 0:
                state_dict = state_dict_ref
                config_dict = state_dict.pop("__model_config_dict__", None)
                # Break shared-memory tensors ... 8 LOC
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                # Save config ... 5 LOC
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            # Legacy path: distributed gather + write
            self.policy_engine.save_16bit_model(output_dir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            # Fix state dict on rank 0 if using adapters ... 20 LOC
            # Save config on rank 0 ... 5 LOC
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
```

This is infrastructure logic (file I/O, distributed barriers, tensor deduplication) embedded inside a domain-specific class. An LLM modifying the algorithm must wade through persistence I/O code.

**AFTER (v2.0) — State persistence as a standalone tool:**

```python
# oxrl/tools/checkpoint.py v2.0 — deterministic persistence tool
def save_state_dict_to_disk(state_dict, output_dir, config_dict=None):
    """Save a state dict to disk as safetensors.

    TOOL CONTRACT:
        Input:  state_dict (dict), output_dir (str), optional config_dict
        Output: files written to output_dir/
        Side effects: creates files on disk
        Deterministic: yes — same input → same files
    """
    os.makedirs(output_dir, exist_ok=True)
    # Break shared-memory tensors for safetensors compatibility
    state_dict = _dedup_shared_tensors(state_dict)
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    if config_dict is not None:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

def gather_and_save(engine, output_dir, tag, lora_config=None):
    """Gather distributed weights and save to disk. Collective operation."""
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

Now the algorithm's save method becomes a 5-LOC delegation:

```python
class GRPO(BaseAlgorithm):
    def save_checkpoint(self, output_dir, tag, state_dict_ref=None):
        if state_dict_ref is not None:
            save_state_dict_to_disk(state_dict_ref, output_dir, config_dict)
        else:
            gather_and_save(self.policy_engine, output_dir, tag, self.lora_config)
```

---

#### The Complete v2.0 Tool Inventory

**BEFORE (v1.0) — Tools buried inside classes:**

```
GRPO class (665 LOC):
  ├── _strip_lora_and_merge()     → weight merging
  ├── gather_state_dict()          → distributed gathering
  ├── save_checkpoint()            → file I/O (2 paths)
  └── _get_base_model_config()     → config extraction

VLLMRolloutEngine class (565 LOC):
  ├── normalize_rewards()          → z-score normalization
  ├── extract_logprobs()           → output parsing
  ├── score_response()             → scoring computation
  └── make_sampling_params()       → parameter validation

Config class (483 LOC):
  ├── _sync_batch_sizes()          → config sync
  ├── _sync_dtype()                → dtype mapping
  ├── _sync_optimizer()            → optimizer config
  └── _sync_scheduler()            → scheduler config
```

**AFTER (v2.0) — Tools extracted as standalone modules:**

```
oxrl/tools/                        # NEW: standalone deterministic tools
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
├── schema.py                      # Pydantic models only                ~200 LOC
├── sync.py                        # sync_deepspeed_config() and friends ~150 LOC
└── loader.py                      # load_and_verify()                   ~50 LOC
```

Every tool is:
- **Standalone**: can be imported and called independently
- **Tested**: has clear inputs, outputs, and invariants
- **Deterministic**: same input → same output
- **Small**: fits within the LLM's optimal intelligence window

---

### Chapter 9: Zero-Hallucination Contracts

> *Tools must guarantee stable I/O, reproducibility, and zero side effects.*

The principle: at system boundaries (user input, config files, external API responses), validate aggressively. Use schema enforcement to prevent the LLM — or any caller — from passing nonsensical values. Silent failures are the enemy.

#### What v1.0 Gets Right

Scoring functions already have a stable contract:

```python
def reward_func(prompt_ids, response_ids, finish_reason, metadata=None) -> Tuple[Tensor, bool]
```

Pydantic already rejects unknown fields:

```python
class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')  # ← Zero-hallucination wall
```

#### What v1.0 Gets Wrong: Silent Failures in Config

**BEFORE (v1.0) — Silent defaults mask configuration errors:**

```python
class Rollout(BaseModel):
    temperature: float = 1.0          # Default: reasonable
    max_tokens: int = 512             # Default: reasonable
    n_samples: int = 8                # Default: reasonable
    force_strict_on_policy: bool = True  # Default: enforced

# Problem: If someone sets n_samples=0, system silently produces empty output.
# Problem: If someone sets max_tokens=0, service returns empty results silently.
```

**AFTER (v2.0) — Pydantic validators catch everything:**

```python
class Rollout(BaseModel):
    model_config = ConfigDict(extra='forbid')
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32768)         # ← must be > 0
    n_samples: int = Field(default=8, ge=1, le=64)               # ← must be >= 1
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    gpu_memory_utilization: float = Field(default=0.5, gt=0.0, le=0.95)  # ← can't be 1.0
    rollout_batch_size_per_gpu: int = Field(default=2, ge=1)     # ← must be >= 1
    force_strict_on_policy: bool = True

class Train(BaseModel):
    model_config = ConfigDict(extra='forbid')
    lr: float = Field(default=1e-5, gt=0, le=1.0)               # ← can't be negative or > 1
    clip_low: float = Field(default=0.2, ge=0.0, le=1.0)        # ← bounded
    clip_high: float = Field(default=0.2, ge=0.0, le=1.0)       # ← bounded
    total_number_of_epochs: int = Field(ge=1)                     # ← must be >= 1
    train_batch_size_per_gpu: int = Field(default=2, ge=1)       # ← must be >= 1
    gradient_accumulation_steps: int = Field(default=1, ge=1)    # ← must be >= 1
```

Every field has explicit bounds. An LLM generating a config cannot set `n_samples: 0` or `lr: -0.001` — Pydantic raises an immediate `ValidationError`. The tool strips the LLM of improvisational freedom on infrastructure parameters.

#### What v1.0 Gets Wrong: External Calls Can Crash Silently

**BEFORE (v1.0) — Bare except swallows all errors:**

```python
# oxrl/rewards/code.py
def code_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
    ...
    try:
        result = subprocess.run(["python", f.name], capture_output=True, timeout=5)
        if result.returncode == 0:
            r[-1] = 1.0
    except Exception:
        pass     # ← Silent failure! Could be timeout, permission error, OOM, anything.
    return r, is_per_token
```

**AFTER (v2.0) — Structured error reporting:**

```python
# oxrl/rewards/code.py v2.0 — report errors, don't swallow them
def code_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
    ...
    try:
        result = subprocess.run(["python", f.name], capture_output=True, timeout=5)
        if result.returncode == 0:
            r[-1] = 1.0
        # returncode != 0 means test failed → score stays 0.0 (correct behavior)
    except subprocess.TimeoutExpired:
        pass  # Timeout → treat as failed (score = 0.0) — this is intentional
    except Exception as e:
        # Log unexpected errors so they can be diagnosed
        import logging
        logging.getLogger(__name__).warning("code_reward unexpected error: %s", e)
    return r, is_per_token
```

---

## Epilogue

### From Code Writer to System Shepherd

#### The Developer's New Role

Engineers mastering these patterns are no longer if-else laborers. We become:

1. **Providers of atomic gears** — Write small, single-purpose modules that fit in an LLM's attention window
2. **Designers of goal-feedback systems** — Build attempt → evaluate → diagnose → fix → retry loops
3. **Architects of deterministic boundaries** — Define what is sealed (tools) vs what is flexible (routing)

These roles are the same whether you're building a training framework, a web application, a data pipeline, or a CLI tool. The patterns are universal.

#### The oxRL v1.0 → v2.0 Transformation Summary

| Metric | v1.0 | v2.0 | Improvement |
|---|---|---|---|
| Largest class (GRPO) | 665 LOC | ~250 LOC | 2.7x smaller |
| Algorithm variants (in one method) | 3 interleaved | 3 separate files | Zero cross-contamination |
| main_rl.py | 595 LOC, 8 concerns | 120 LOC, pure orchestration | 5x smaller |
| VLLMRolloutEngine | 565 LOC, 5 concerns | 200 LOC, 2 concerns | 2.8x smaller |
| Config file | 483 LOC, 3 concerns | 3 files, ~130 LOC each | Focused |
| Weight merging | Duplicated in 2 classes | Single tool, reusable | DRY |
| State persistence | 98 LOC inside algorithm class | 70 LOC standalone tool | Decoupled |
| Health monitoring | None | Micro + macro + system loops | 3-tier feedback |
| Bugfixer strategies | Parameter patches only | Params + adapters + component swap | Richer evolution |
| Module black-box specs | None (read source) | Calling specs on every module | ~65 tokens to understand pipeline |

#### The Ultimate Vision

Use human wisdom to design **rules** (Pydantic validators, enforcement walls), **feedback loops** (attempt → diagnose → fix → retry), and **tool boundaries** (what is deterministic vs probabilistic).

Let AI compute power **assemble** micro-gears dynamically, **refactor** within the constraints of the deterministic backbone, and **evolve** through feedback loops without human intervention.

---

## Appendix: Full Before/After File Map

### BEFORE (v1.0) — Current Structure

```
oxrl/
├── algs/
│   ├── base.py              41 LOC  ← KEEP (already a good micro-gear)
│   ├── grpo.py             665 LOC  ← SHATTER (7 responsibilities)
│   ├── ppo.py              880 LOC  ← SHATTER (similar issues)
│   ├── sft.py              145 LOC  ← KEEP (already focused)
│   ├── dpo.py              100 LOC  ← KEEP
│   └── ... (11 more, 60-100 LOC each) ← KEEP
├── configs/
│   └── load.py             483 LOC  ← SPLIT (schema + sync + loader)
├── rollouts/
│   ├── vllm_engine.py      565 LOC  ← EXTRACT (5 concerns → 1 service + 4 tools)
│   └── replay_buffer.py    208 LOC  ← KEEP (already focused)
├── rewards/
│   ├── base.py              62 LOC  ← KEEP
│   ├── math.py              77 LOC  ← KEEP
│   ├── code.py              37 LOC  ← IMPROVE (error reporting)
│   └── ... (5 more)                 ← KEEP
├── swarm/
│   ├── scout.py            898 LOC  ← KEEP (already goal-decomposed)
│   ├── bugfixer.py         777 LOC  ← EXPAND (richer fix strategies)
│   └── orchestrator.py     545 LOC  ← KEEP (already evolutionary)
├── datasets/                        ← KEEP (already focused files)
├── utils/                           ← KEEP
├── main_rl.py              595 LOC  ← SHATTER (→ orchestrator + 3 phases)
└── main_sl.py              ~600 LOC ← SHATTER (same pattern as main_rl)
```

### AFTER (v2.0) — Proposed Structure

```
oxrl/
├── algs/
│   ├── base.py              41 LOC  — interface contract
│   ├── losses/                       — NEW: extracted algorithm variants
│   │   ├── sgrpo_loss.py    40 LOC  — variant A: token-level clipped surrogate
│   │   ├── gspo_loss.py     35 LOC  — variant B: sequence-level (MoE)
│   │   ├── cispo_loss.py    30 LOC  — variant C: conservative indirect
│   │   └── ppo_loss.py      40 LOC  — variant D: policy + value
│   ├── grpo.py             250 LOC  — orchestration only (was 665)
│   ├── ppo.py              400 LOC  — orchestration only (was 880)
│   ├── sft.py              145 LOC  — unchanged
│   ├── dpo.py              100 LOC  — unchanged
│   └── ...                          — unchanged
│
├── tools/                            — NEW: deterministic tools
│   ├── lora_merge.py        50 LOC  — weight merging (was in algorithm class)
│   ├── checkpoint.py        70 LOC  — save/gather/dedup (was in algorithm class)
│   ├── tensor_utils.py      35 LOC  — ensure_1d, pad_1d (was in utils)
│   └── config_extract.py    15 LOC  — extract model config (was in algorithm class)
│
├── configs/
│   ├── schema.py           200 LOC  — pure Pydantic models (was in load.py)
│   ├── sync.py             150 LOC  — config sync logic (was in load.py)
│   └── loader.py            50 LOC  — YAML loading (was in load.py)
│
├── rollouts/
│   ├── vllm_engine.py      200 LOC  — service: lifecycle + core processing (was 565)
│   ├── sampling.py          50 LOC  — parameter construction (was in service)
│   ├── logprob_utils.py     40 LOC  — output parsing (was in service)
│   ├── reward_scoring.py    20 LOC  — score_response() (was in service)
│   ├── normalization.py     40 LOC  — z-score normalization (was in service)
│   └── replay_buffer.py    208 LOC  — unchanged
│
├── loops/                            — NEW: processing loop phases
│   ├── rollout_phase.py     60 LOC  — data generation (was in main)
│   ├── train_phase.py       50 LOC  — dispatch + metrics (was in main)
│   ├── checkpoint_phase.py  80 LOC  — gather + save + refresh (was in main)
│   └── health_check.py      60 LOC  — NEW: micro/macro health monitoring
│
├── setup/                            — NEW: initialization separated from loop
│   ├── ray_setup.py         30 LOC  — infrastructure init (was in main)
│   ├── engine_factory.py    60 LOC  — spawn actors (was in main)
│   └── dataloader_factory.py 30 LOC — create data pipeline (was in main)
│
├── rewards/                          — KEEP + improve error reporting
├── swarm/                            — KEEP + expand bugfixer strategies
├── datasets/                         — KEEP
├── utils/                            — KEEP
│
├── main_rl.py              120 LOC  — pure orchestration (was 595)
└── main_sl.py              120 LOC  — pure orchestration (was ~600)
```

### Token Budget Comparison

| Task | v1.0 Tokens | v2.0 Tokens (largest file) | Reduction |
|---|---|---|---|
| Modify GRPO algorithm | ~2,700 | ~1,000 | 2.7x |
| Modify one algorithm variant | ~2,700 (must read all) | ~140 | **19x** |
| Modify processing pipeline | ~2,400 | ~480 | 5x |
| Debug state persistence | ~2,700 (must read all) | ~280 | **9.6x** |
| Add a config field | ~1,950 | ~800 | 2.4x |
| Modify inference service | ~2,300 | ~800 | 2.9x |
| Modify normalization | ~2,300 (must read all) | ~160 | **14x** |
| Understand entire pipeline (via specs) | ~10,500 | ~260 (specs only) | **40x** |
