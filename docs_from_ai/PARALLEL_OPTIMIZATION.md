# 并行优化深度分析

## 1. 当前配置分析

### 你的机器
- **16 核心 24 线程**
- 这说明有 **超线程（Hyper-Threading/SMT）**
- 物理核心：16
- 逻辑核心：24

### 目标服务器（根据任务描述）
- **CPU**: AMD x86_64, 16 cores
- **问题**：任务描述只说 "16 cores"，没有明确是物理核心还是逻辑核心

---

## 2. AMD CPU 超线程技术

### AMD SMT (Simultaneous Multithreading)

**AMD 的情况：**
- AMD 从 Zen 架构开始支持 SMT
- 类似 Intel 的超线程（Hyper-Threading）
- 每个物理核心支持 2 个线程

**常见配置：**
```
AMD Ryzen/EPYC (Zen 架构):
- 8 物理核心 → 16 逻辑核心
- 16 物理核心 → 32 逻辑核心
- 32 物理核心 → 64 逻辑核心
```

### 任务描述中的 "16 cores" 可能是：

**情况 A：16 物理核心（更可能）**
```
物理核心: 16
逻辑核心: 32 (如果启用 SMT)
```

**情况 B：16 逻辑核心（不太可能）**
```
物理核心: 8
逻辑核心: 16
```

**根据服务器规格（128GB 内存），更可能是情况 A**
- 128GB 内存通常配 16+ 物理核心
- 所以目标服务器很可能是 **16 物理核心 32 逻辑核心**

---

## 3. 当前并行配置

### 默认配置
```python
PARALLEL_WORKERS = 8      # 8 个进程
每个进程线程 = 16 / 8 = 2  # 每进程 2 线程
总线程 = 8 × 2 = 16
```

### 问题分析

**问题 1：未充分利用超线程**
```
如果目标服务器有 32 逻辑核心：
当前使用: 16 线程
可用资源: 32 线程
利用率: 50%
```

**问题 2：进程数 vs 线程数的权衡**

当前配置（8 进程 × 2 线程）：
- ✅ 充分利用多核（8 个独立进程）
- ❌ 每个进程只用 2 线程（可能浪费 SMT）
- ❌ 内存占用：8 × 1.2GB ≈ 10GB

---

## 4. 优化方案对比

### 方案 1：增加进程数（推荐）

```bash
export PARALLEL_WORKERS=16
export INFERENCE_NUM_THREADS=32
# 结果：16 进程 × 2 线程 = 32 线程
```

**优势：**
- ✅ 充分利用 32 逻辑核心
- ✅ 更高的并行度（16 个独立任务）
- ✅ 更好的负载均衡

**劣势：**
- ❌ 内存占用增加：16 × 1.2GB ≈ 20GB（仍在 128GB 范围内）
- ❌ 进程启动开销略增

**预期加速：**
- 理论：2x（16 进程 vs 8 进程）
- 实际：1.5-1.8x（考虑 SMT 效率 ~75%）

---

### 方案 2：增加每进程线程数

```bash
export PARALLEL_WORKERS=8
export INFERENCE_NUM_THREADS=32
# 结果：8 进程 × 4 线程 = 32 线程
```

**优势：**
- ✅ 内存占用不变（8 × 1.2GB）
- ✅ 进程数少，管理简单

**劣势：**
- ❌ 并行度低（只有 8 个独立任务）
- ❌ llama.cpp 的线程扩展性有限（内存带宽瓶颈）
- ❌ 4 线程 vs 2 线程提升不明显

**预期加速：**
- 理论：1.3x
- 实际：1.1-1.2x（线程扩展性差）

---

### 方案 3：激进配置（最大化）

```bash
export PARALLEL_WORKERS=24
export INFERENCE_NUM_THREADS=32
# 结果：24 进程 × 1 线程 = 24 线程
```

**优势：**
- ✅ 最高并行度（24 个独立任务）
- ✅ 避免线程竞争

**劣势：**
- ❌ 内存占用：24 × 1.2GB ≈ 29GB
- ❌ 每进程只用 1 线程（可能不够高效）
- ❌ 进程启动开销大

**预期加速：**
- 理论：3x
- 实际：2-2.5x

---

### 方案 4：混合策略（平衡）

```bash
export PARALLEL_WORKERS=12
export INFERENCE_NUM_THREADS=32
# 结果：12 进程 × 2-3 线程 ≈ 32 线程
```

**优势：**
- ✅ 平衡并行度和线程效率
- ✅ 内存占用适中：12 × 1.2GB ≈ 15GB

**预期加速：**
- 理论：1.5x
- 实际：1.3-1.5x

---

## 5. 超线程（SMT）的效率

### 超线程不是真正的 2x 性能

**理论 vs 实际：**
```
物理核心: 16
逻辑核心: 32

理论性能: 32 / 16 = 2x
实际性能: 1.2-1.5x
```

**为什么？**
1. **共享资源**：
   - 两个逻辑核心共享同一个物理核心的：
     - ALU（算术逻辑单元）
     - FPU（浮点运算单元）
     - L1/L2 缓存
   
2. **内存带宽瓶颈**：
   - LLM 推理主要受限于内存带宽
   - 超线程无法增加内存带宽
   - 所以提升有限

3. **实际效率**：
   - CPU 密集型任务：1.2-1.3x
   - 内存密集型任务（LLM）：1.1-1.2x
   - 混合任务：1.3-1.5x

---

## 6. 实验建议

### 测试不同配置

```bash
# 测试 1：当前配置（基准）
export PARALLEL_WORKERS=8
python run.py  # 记录时间

# 测试 2：增加进程数
export PARALLEL_WORKERS=12
python run.py  # 对比时间

# 测试 3：最大进程数
export PARALLEL_WORKERS=16
python run.py  # 对比时间

# 测试 4：激进配置
export PARALLEL_WORKERS=20
python run.py  # 对比时间
```

### 如何选择最优配置？

**判断标准：**
1. **推理时间**（最重要）
2. **内存占用**（不超过 128GB）
3. **稳定性**（不要过度超配）

---

## 7. 推荐配置

### 保守配置（稳定优先）
```bash
export PARALLEL_WORKERS=12
export INFERENCE_NUM_THREADS=32
```
- 内存：~15GB
- 预期加速：1.3-1.5x
- 稳定性：高

### 激进配置（性能优先）
```bash
export PARALLEL_WORKERS=16
export INFERENCE_NUM_THREADS=32
```
- 内存：~20GB
- 预期加速：1.5-1.8x
- 稳定性：中

### 极限配置（实验性）
```bash
export PARALLEL_WORKERS=20
export INFERENCE_NUM_THREADS=32
```
- 内存：~25GB
- 预期加速：1.8-2.2x
- 稳定性：需要测试

---

## 8. 其他优化方向

### 优化 1：减少每进程线程数

当进程数增加时，减少每进程线程数：
```bash
# 16 进程配置
export PARALLEL_WORKERS=16
export INFERENCE_NUM_THREADS=16  # 每进程 1 线程
```

**原因：**
- llama.cpp 的多线程扩展性有限
- 1-2 线程通常是最优的
- 更多线程不一定更快（内存带宽瓶颈）

### 优化 2：使用更快的量化

你已经改为 q5_K_M（更高质量），如果需要速度：
```bash
export MODEL_QUANT=q4_0  # 比 q4_K_M 快 10-15%
```

### 优化 3：调整 batch size

llama.cpp 的 `-b` 参数：
```python
'-b', '1024',  # 增加到 1024（当前 512）
```

可能略微提升 prompt 处理速度。

---

## 9. 性能预测

### 当前性能（8 进程）
```
53 题: ~40-50s (并行)
500 题: ~400-500s (6-8 分钟)
```

### 优化后（16 进程）
```
53 题: ~25-30s (1.6-1.8x 加速)
500 题: ~250-300s (4-5 分钟)
```

### 极限优化（20 进程）
```
53 题: ~20-25s (2x 加速)
500 题: ~200-250s (3-4 分钟)
```

---

## 10. 实施建议

### 步骤 1：确认目标服务器配置

在目标服务器上运行：
```bash
# 查看 CPU 信息
lscpu | grep -E "^CPU\(s\)|Thread|Core"

# 输出示例：
# CPU(s):              32      ← 逻辑核心
# Thread(s) per core:  2       ← 超线程
# Core(s) per socket:  16      ← 物理核心
```

### 步骤 2：本地测试不同配置

```bash
# 测试脚本
for workers in 8 12 16 20; do
    echo "Testing PARALLEL_WORKERS=$workers"
    export PARALLEL_WORKERS=$workers
    time python run.py > /dev/null
    echo ""
done
```

### 步骤 3：选择最优配置

根据测试结果选择：
- 时间最短
- 内存可接受
- 稳定性好

### 步骤 4：在代码中设置默认值

如果测试发现 16 进程最优：
```python
self.num_workers = int(os.environ.get('PARALLEL_WORKERS', '16'))  # 改为 16
```

---

## 11. 总结

### 回答你的问题：

**1. 当前并行程度还有优化空间吗？**
- ✅ **有！** 当前只用了 16 线程，目标服务器可能有 32 线程

**2. AMD 服务器有超线程吗？**
- ✅ **很可能有！** AMD Zen 架构支持 SMT（类似超线程）
- 16 核心的服务器通常指 16 物理核心
- 实际可能有 32 逻辑核心

**3. 最佳配置是什么？**
- **推荐**：`PARALLEL_WORKERS=16`（16 进程 × 2 线程）
- **激进**：`PARALLEL_WORKERS=20`（20 进程 × 1-2 线程）
- **预期加速**：1.5-2x

**4. 需要做什么？**
- 测试不同的 `PARALLEL_WORKERS` 值（12, 16, 20）
- 选择时间最短的配置
- 更新代码默认值

---

## 12. 快速测试命令

```bash
# 在你的机器上测试（16 核 24 线程）
cd /home/yz/Projects/huawei

# 测试 12 进程
export PARALLEL_WORKERS=12
time python run.py

# 测试 16 进程
export PARALLEL_WORKERS=16
time python run.py

# 测试 20 进程
export PARALLEL_WORKERS=20
time python run.py

# 对比结果，选择最快的
```

需要我帮你运行这些测试吗？

