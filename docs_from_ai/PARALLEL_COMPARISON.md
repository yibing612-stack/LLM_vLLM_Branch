# 并行方案对比：llama.cpp vs Transformers

## 问题：当前的并行方案能否直接用在 Transformers 方案里？

**简短回答：不能直接用，但可以改造。两种方案的并行方式完全不同。**

---

## 1. 两种方案的根本区别

### llama.cpp 方案（当前实现）

**特点：外部进程调用**
```python
# 每次推理都启动一个独立的 llama-cli 进程
subprocess.run(['llama-cli', '-m', 'model.gguf', ...])
```

**并行方式：多进程**
```python
# 启动多个独立的 llama-cli 进程
with multiprocessing.Pool(8) as pool:
    answers = pool.map(worker_func, questions)
```

**为什么需要多进程？**
- llama-cli 是独立的 C++ 程序
- 每次调用都是一个新进程
- 无法在单个进程内批处理
- **必须**通过多进程实现并行

**内存模型：**
```
Process 1: [Python] → [llama-cli subprocess] → [加载模型 1.2GB]
Process 2: [Python] → [llama-cli subprocess] → [加载模型 1.2GB]
Process 3: [Python] → [llama-cli subprocess] → [加载模型 1.2GB]
...
Process 8: [Python] → [llama-cli subprocess] → [加载模型 1.2GB]

总内存: 8 × 1.2GB ≈ 10GB
```

---

### Transformers 方案

**特点：Python 内存中的模型**
```python
# 模型加载在 Python 内存中
model = AutoModelForCausalLM.from_pretrained(...)
```

**并行方式：批处理（Batching）**
```python
# 一次性处理多个输入（在同一个进程中）
inputs = tokenizer(questions, padding=True, return_tensors='pt')
outputs = model.generate(**inputs)  # 内部并行
```

**为什么用批处理？**
- 模型已经在内存中
- PyTorch 支持张量批处理
- 权重共享，一次前向传播处理多个样本
- **不需要**多进程

**内存模型：**
```
Single Process:
  [Python]
    ↓
  [PyTorch Model 3.5GB]
    ↓
  [Batch processing: 4 samples in parallel]

总内存: 3.5GB (模型) + 少量 batch overhead
```

---

## 2. 为什么不能直接迁移？

### 问题 1：架构完全不同

**llama.cpp 并行方案的核心：**
```python
def worker(question):
    # 每个 worker 启动独立的 llama-cli 进程
    subprocess.run(['llama-cli', ...])
```

**如果直接用在 Transformers：**
```python
def worker(question):
    # ❌ 每个 worker 都会加载一次完整模型！
    model = AutoModelForCausalLM.from_pretrained(...)  # 3.5GB!
    output = model.generate(...)
```

**结果：**
- 8 个进程 × 3.5GB = **28GB 内存**（vs llama.cpp 的 10GB）
- 每个进程都要加载模型（**非常慢**，可能需要 30-60 秒）
- 完全没有意义！

### 问题 2：PyTorch 的多进程问题

PyTorch 模型在多进程中使用非常复杂：
```python
# ❌ 这样不行
with multiprocessing.Pool(8) as pool:
    # PyTorch 模型无法被 pickle（序列化）
    # CUDA context 无法跨进程共享
    pool.map(model.generate, questions)  # 会报错！
```

**PyTorch 多进程的正确方式：**
- 使用 `torch.multiprocessing`（不是标准 `multiprocessing`）
- 使用 `spawn` 方法
- 在每个进程中重新加载模型
- 非常复杂且低效

---

## 3. Transformers 方案的正确并行方式

### 方案 A：原生批处理（推荐）

**这是 Transformers 的标准做法：**

```python
class MyModel:
    def __init__(self):
        # 模型只加载一次
        self.model = AutoModelForCausalLM.from_pretrained(...)
        self.tokenizer = AutoTokenizer.from_pretrained(...)
        self.batch_size = 4  # 或 8
    
    def __call__(self, questions):
        answers = []
        # 分批处理
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            
            # 准备批量输入
            prompts = [self._format_prompt(q) for q in batch]
            inputs = self.tokenizer(
                prompts, 
                padding=True,           # 填充到相同长度
                return_tensors='pt'
            ).to(self.model.device)
            
            # 批量生成（关键！）
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 批量解码
            batch_answers = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            answers.extend(batch_answers)
        
        return answers
```

**优势：**
- ✅ 模型只加载一次（3.5GB）
- ✅ 权重共享，内存高效
- ✅ PyTorch 内部优化的并行
- ✅ 简单，不需要多进程

**性能：**
- Batch size = 4 → 2-3x 加速
- Batch size = 8 → 3-4x 加速

**限制：**
- 受内存限制（batch 越大，内存越多）
- 受 CPU/GPU 计算能力限制

---

### 方案 B：多进程 + 模型分片（高级，不推荐）

如果真的想用多进程：

```python
import torch.multiprocessing as mp

def worker_process(rank, questions, model_name, cache_dir, result_queue):
    """每个进程独立加载模型"""
    # 在子进程中加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # 处理分配给这个进程的问题
    for q in questions:
        answer = generate_answer(model, tokenizer, q)
        result_queue.put(answer)

def parallel_inference(all_questions, num_processes=4):
    # 分配问题给各个进程
    chunks = [all_questions[i::num_processes] for i in range(num_processes)]
    
    # 启动进程
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    for rank, chunk in enumerate(chunks):
        p = mp.Process(
            target=worker_process,
            args=(rank, chunk, model_name, cache_dir, result_queue)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    for p in processes:
        p.join()
    
    # ...
```

**问题：**
- ❌ 每个进程加载模型（4 × 3.5GB = 14GB）
- ❌ 加载时间长（每个进程 30-60 秒）
- ❌ 复杂，容易出错
- ❌ 不如批处理高效

**结论：对 Transformers 来说，这种方式没有意义！**

---

## 4. 性能对比

### llama.cpp + 多进程（当前方案）

```
配置: 8 进程 × 2 线程
内存: ~10GB
加载时间: 每个进程 ~2 秒（GGUF 加载快）
推理速度: 0.73s/问题
总体: ✅ 高效
```

### Transformers + 批处理（推荐）

```
配置: 1 进程，batch_size=4
内存: ~4GB
加载时间: 一次性 ~30 秒
推理速度: 预计 0.5-0.8s/问题（取决于 batch size）
总体: ✅ 高效
```

### Transformers + 多进程（不推荐）

```
配置: 4 进程
内存: ~14GB
加载时间: 4 × 30 秒 = 120 秒！
推理速度: 可能更慢（加载开销太大）
总体: ❌ 低效
```

---

## 5. 实际建议

### 对于 llama.cpp 方案（当前）

✅ **使用多进程并行**（已实现）
```bash
export PARALLEL_WORKERS=8  # 8 个并行进程
```

**原因：**
- llama-cli 是外部程序，必须用多进程
- GGUF 模型小（1.2GB），加载快
- 多进程开销可接受

---

### 对于 Transformers 方案

✅ **使用批处理**（不是多进程）
```python
self.batch_size = 4  # 或 8

# 批量处理
for batch in batched(questions, self.batch_size):
    outputs = self.model.generate(batch_inputs)
```

**原因：**
- 模型已在内存中，无需多进程
- PyTorch 原生支持批处理
- 简单、高效、稳定

❌ **不要用多进程**
- 内存浪费（每个进程都加载模型）
- 加载时间长
- 复杂且容易出错

---

## 6. 如何改造 Transformers 方案？

### 步骤 1：修改 `__call__` 方法

```python
def __call__(self, questions):
    """批处理版本"""
    answers = []
    
    # 分批处理
    for i in range(0, len(questions), self.batch_size):
        batch = questions[i:i + self.batch_size]
        batch_answers = self.get_answer_batch(batch)
        
        for q, a in zip(batch, batch_answers):
            answers.append({
                'questionID': q['questionID'],
                'answer': a
            })
    
    return answers
```

### 步骤 2：实现批量推理

```python
def get_answer_batch(self, questions):
    """批量生成答案"""
    # 格式化所有 prompts
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": q['question']}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # 批量 tokenize
    inputs = self.tokenizer(
        prompts,
        padding=True,
        return_tensors='pt'
    ).to(self.model.device)
    
    # 批量生成
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=64,
        pad_token_id=self.tokenizer.eos_token_id
    )
    
    # 批量解码
    answers = []
    for i, output in enumerate(outputs):
        # 只解码新生成的部分
        input_len = inputs['input_ids'][i].shape[0]
        answer = self.tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True
        )
        answers.append(answer.strip())
    
    return answers
```

### 步骤 3：配置 batch size

```python
def __init__(self):
    # 根据内存和 CPU 核心数设置
    # 经验值：每 4 个核心 → batch_size = 1
    # 16 核 → batch_size = 4
    self.batch_size = int(os.environ.get('BATCH_SIZE', '4'))
    self.get_model()
```

---

## 7. 总结对比表

| 特性 | llama.cpp 方案 | Transformers 方案 |
|------|---------------|------------------|
| **并行方式** | 多进程 | 批处理 |
| **实现复杂度** | 中等 | 简单 |
| **内存占用** | 进程数 × 1.2GB | 3.5GB + batch overhead |
| **加载时间** | 快（每进程 2s） | 慢（一次 30s） |
| **推理速度** | 0.73s/题 | 0.5-0.8s/题 |
| **CPU 利用率** | 高（多进程） | 中等（单进程批处理） |
| **适用场景** | 外部程序调用 | 内存中的模型 |

---

## 8. 结论

**不能直接迁移！** 两种方案的并行方式完全不同：

1. **llama.cpp**：必须用多进程（因为是外部程序）
2. **Transformers**：应该用批处理（因为模型在内存中）

**如果要优化 Transformers 方案：**
- ✅ 实现批处理（简单、高效）
- ❌ 不要用多进程（复杂、低效）

**当前 llama.cpp 的多进程方案：**
- ✅ 对 llama.cpp 是正确的选择
- ✅ 已经实现并验证有效（2.4x 加速）
- ✅ 继续使用当前方案

需要我帮你实现 Transformers 的批处理版本吗？

