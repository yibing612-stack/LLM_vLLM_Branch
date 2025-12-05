# Batch Processing 详解：Transformers vs llama.cpp

## 1. Context Size (上下文大小)

### Transformers 方案
查看之前的代码，**没有显式设置 context size**。使用的是：
- **模型默认值**：Qwen3-1.7B 的默认 context 是 **32,768 tokens**
- PyTorch/Transformers 会根据输入自动调整
- 每次推理都可以使用完整的 context window

### llama.cpp 方案
当前设置：
```bash
-c 2048  # 上下文限制为 2048 tokens
```

**为什么设置更小？**
1. **内存效率**：Context size 直接影响 KV cache 内存占用
   - KV cache 大小 ∝ context_size × batch_size × hidden_dim
   - 2048 对于简短问答已经足够（我们的问题通常 < 200 tokens）

2. **速度优化**：更小的 context 意味着：
   - 更少的内存访问
   - 更快的 attention 计算（O(n²) 复杂度）

**对比：**
| 方案 | Context Size | 说明 |
|------|-------------|------|
| Transformers | 32,768 (默认) | 未显式限制，使用模型最大值 |
| llama.cpp | 2048 (当前) | 手动优化，适合简短问答 |

---

## 2. Batch Size 的加速原理

### 什么是 Batch Processing？

**批处理** = 一次性处理多个样本，而不是逐个处理。

### 加速原理详解

#### 原理 1: **并行计算（GPU 优势明显，CPU 也受益）**

**串行处理**（batch_size = 1）：
```python
for question in questions:
    # 每次只处理 1 个问题
    answer = model.generate(question)
    # GPU/CPU 只处理单个样本，硬件未充分利用
```

**批处理**（batch_size = 4）：
```python
for batch in batched(questions, batch_size=4):
    # 一次处理 4 个问题
    answers = model.generate(batch)
    # GPU/CPU 并行处理多个样本
```

**为什么更快？**
```
假设每个问题处理耗时：
- 模型加载权重：80% 时间（内存带宽瓶颈）
- 实际计算：20% 时间

串行处理 4 个问题：
Question 1: [加载权重 80ms] [计算 20ms] = 100ms
Question 2: [加载权重 80ms] [计算 20ms] = 100ms
Question 3: [加载权重 80ms] [计算 20ms] = 100ms
Question 4: [加载权重 80ms] [计算 20ms] = 100ms
总时间：400ms

批处理（batch_size=4）：
Batch [1-4]: [加载权重 80ms] [并行计算 4×20ms ≈ 30ms] = 110ms
总时间：110ms

加速比：400ms / 110ms ≈ 3.6x
```

**关键点**：
- 权重只加载一次，被所有样本共享
- 计算可以并行化（矩阵运算的优势）
- 内存访问次数大幅减少

#### 原理 2: **内存访问优化**

LLM 推理的瓶颈主要是**内存带宽**，不是计算能力。

**串行处理**：
```
Iteration 1: RAM → Cache → Compute → Output
Iteration 2: RAM → Cache → Compute → Output  # 重复加载权重！
Iteration 3: RAM → Cache → Compute → Output
Iteration 4: RAM → Cache → Compute → Output
```

**批处理**：
```
Batch: RAM → Cache → [Compute×4 并行] → Output×4
```

权重只需要从 RAM 加载**一次**，然后被重复使用。

#### 原理 3: **向量化（SIMD）**

现代 CPU/GPU 支持 SIMD（Single Instruction, Multiple Data）：

```python
# 串行：需要 4 次独立操作
result1 = weights @ input1
result2 = weights @ input2
result3 = weights @ input3
result4 = weights @ input4

# 批处理：1 次矩阵乘法
# weights: [hidden, hidden]
# inputs:  [batch_size=4, hidden]
results = weights @ inputs.T  # 一次完成所有计算
```

CPU 的 AVX2/AVX512 指令集可以同时处理多个数据。

---

### 之前 Transformers 方案的 Batch Processing

查看备份代码：

```python
class MyModel:
    def __init__(self):
        # batch_size 根据 CPU 线程数自动设置
        default_bs = max(1, _target_threads // 4)  # 16线程 → batch_size=4
        self.batch_size = int(os.environ.get('INFERENCE_BATCH_SIZE', str(default_bs)))
    
    def __call__(self, questions):
        answers = []
        # 将问题分批处理
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            # 一次性生成整个 batch 的答案
            batch_answers = self.get_answer([q['question'] for q in batch])
            for q, a in zip(batch, batch_answers):
                answers.append({'questionID': q['questionID'], 'answer': a})
        return answers
    
    def get_answer(self, questions):
        # questions 是一个列表，包含多个问题
        for q in questions:
            messages = [{"role": "system", ...}, {"role": "user", "content": q}]
            prompts.append(self.tokenizer.apply_chat_template(...))
        
        # 一次性 tokenize 所有 prompts
        inputs = self.tokenizer(prompts, padding=True, return_tensors='pt')
        
        # 一次性生成所有答案
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 解码所有答案
        return [self.tokenizer.decode(...) for output in outputs]
```

**关键点**：
- `model.generate()` 接收 `[batch_size, seq_len]` 的张量
- PyTorch 内部并行处理所有样本
- 权重共享，一次前向传播处理多个输入

**理论加速比**：
- Batch size = 4 → 约 2-3x 加速
- Batch size = 8 → 约 3-4x 加速
- 但受内存限制（batch 越大，内存占用越高）

---

## 3. llama.cpp 的 Batch Processing

### llama.cpp 的批处理参数

查看当前代码：
```bash
-b 512         # Batch size for prompt processing (批处理大小)
-tb 16         # Threads for batch processing (批处理线程)
```

**重要区别**：llama.cpp 的 "batch" 含义不同！

### llama.cpp 的三种 "Batch"

#### 1. **Physical Batch**（物理批处理）- 多个 prompt 并行
这才是我们通常说的批处理：
```bash
llama-cli --parallel 4  # 同时处理 4 个不同的 prompt
```

但 `llama-cli` **不支持**这个模式！它一次只能处理一个 prompt。

要实现类似效果需要：
- 使用 `llama-server`（支持并发请求）
- 或启动多个 `llama-cli` 进程

#### 2. **Logical Batch**（逻辑批处理）- Prompt 处理优化
这是 `-b 512` 的含义：
```bash
-b 512  # 一次处理 512 个 tokens 的 prompt
```

**作用**：
- 将长 prompt 分块处理
- 优化内存访问模式
- 主要加速 prompt 处理阶段（不是生成阶段）

**示例**：
```
Prompt 有 1000 tokens：
- 不用 batch：逐个 token 处理，慢
- batch=512：分 2 批处理，[0:512] + [512:1000]，快
```

#### 3. **Continuous Batching**（连续批处理）
llama.cpp 内部的优化技术，自动进行。

---

### 当前 llama.cpp 方案能否像 Transformers 那样批处理？

**答案：可以，但需要改造！**

### 方案 A：多进程并行（推荐）

启动多个 `llama-cli` 进程，每个处理不同的问题：

```python
import multiprocessing
import subprocess

def process_question(question, model_path, bin_path):
    """单个进程处理一个问题"""
    # 创建临时 prompt 文件
    # 调用 llama-cli
    # 返回结果
    ...

def __call__(self, questions):
    # 使用进程池并行处理
    with multiprocessing.Pool(processes=4) as pool:
        answers = pool.starmap(
            process_question,
            [(q, self.gguf_path, self.llama_bin) for q in questions]
        )
    return answers
```

**优势**：
- 真正的并行处理
- 每个进程独立，稳定性好

**劣势**：
- 每个进程都加载模型到内存
- 内存占用：模型大小 × 并行数
- Qwen3-1.7B q4_K_M ≈ 1.2GB → 4 进程 ≈ 4.8GB

**评估**：
- 128GB 内存的服务器**完全可行**！
- 可以同时运行 8-16 个进程
- 理论加速：接近线性（8 进程 ≈ 8x）

### 方案 B：使用 llama-server（更优雅）

llama.cpp 提供的服务器模式，原生支持并发：

```bash
# 启动服务器（支持并发请求）
llama-server -m model.gguf -c 2048 -np 8  # 最多 8 个并行 slot
```

```python
import requests
import concurrent.futures

def query_llama_server(question):
    response = requests.post('http://localhost:8080/completion', 
                            json={'prompt': question, 'n_predict': 64})
    return response.json()['content']

def __call__(self, questions):
    # 并发发送请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        answers = list(executor.map(query_llama_server, questions))
    return answers
```

**优势**：
- 模型只加载一次
- 内部优化的并发处理
- 更高效的内存使用

**劣势**：
- 需要先启动服务器
- 稍微复杂一点的架构

---

## 4. 性能对比总结

### Transformers 方案（之前）
```
配置：
- Batch size: 4（自动）
- Context: 32,768（默认）
- 精度: bfloat16（完整精度）
- 内存: ~3.5GB（模型本身）

性能：
- 加速: 批处理带来 2-3x 加速
- 问题: 16 线程 100% 占用（aggressive threading）
```

### llama.cpp 方案（当前）
```
配置：
- Batch size: 1（串行处理）
- Context: 2,048（优化）
- 精度: q4_K_M（量化）
- 内存: ~1.2GB（模型）

性能：
- 速度: 1.76s/问题
- 问题: CPU 利用率波动（正常现象）
```

### llama.cpp + 多进程方案（建议）
```
配置：
- 并行进程: 8
- 每个进程: 2 线程
- Context: 2,048
- 精度: q4_K_M
- 内存: ~10GB（8×1.2GB + overhead）

预期性能：
- 理论加速: 6-8x
- 预计速度: ~0.25s/问题
- CPU: 16 核心充分利用
- 内存: 10GB < 128GB ✓
```

---

## 5. 实现建议

### 立即可行的优化（无需改代码）

**选项 1：使用更激进的量化**
```bash
export MODEL_QUANT=q4_0  # 比 q4_K_M 快 10-20%
```

### 中等改造（推荐）

**实现多进程并行**：
- 预计工作量：~50 行代码
- 加速比：6-8x
- 风险：低（每个进程独立）

我可以帮你实现这个！

### 高级方案

**使用 llama-server**：
- 需要在 loadPipeline() 时启动服务器
- 推理时发送并发请求
- 更优雅但稍复杂

---

## 6. 推荐方案

**针对 16 核 128GB 服务器，我推荐：**

1. **实现 8 进程并行**
   - 每进程 2 线程
   - 总共充分利用 16 核
   - 内存占用 ~10GB（安全）

2. **保持当前量化**
   - q4_K_M 平衡质量和速度
   - 或尝试 q4_0 如果需要极致速度

3. **预期效果**
   - 推理时间：93s → ~15s（50 题）
   - 500 题：~150s（2.5 分钟）
   - CPU 利用率：接近 100%（所有核心）

**需要我帮你实现多进程版本吗？**

