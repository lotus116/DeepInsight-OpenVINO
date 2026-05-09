# DeepInsight 智能经营决策系统

> **基于 Intel OpenVINO™ 与 DeepSeek V3.2 的端云可切换混合部署架构**  
> 适用于中小企业、分支机构及对数据合规有严苛要求的行业场景（如金融、制造、政务）

---

## ✨ 核心特性

| 维度 | 指标 |
|------|------|
| G1 Baseline | 平均 7424.4 ms, P95 12169.0 ms |
| G2 RAG+KECA | 平均 9495.1 ms, P95 16686.3 ms |
| G3 Full System | 平均 12006.9 ms, P95 25379.1 ms |
| PyTorch FP32 | 3.4±0.1 ms, P95 4.4 ms, 292.8 QPS |
| OpenVINO FP32 | 1.0±0.1 ms, P95 1.6 ms, 1013.8 QPS |
| OpenVINO 加速比 | 3.46x（基于 10 轮平均结果） |

说明：`performance_summary.json` 可能保留单轮结果；对外引用时建议优先使用 `averaged_performance_results.json` 与 `evaluation_report.md`。

---

## 核心能力

| 模块 | 当前能力 |
|------|----------|
| 自然语言理解 | 支持中文查询，自动生成 SQL 与业务解释 |
| 检索增强 | 二阶段混合检索、KECA 术语/示例注入、外键依赖补全 |
| Agent 自愈 | 基于错误上下文的重试修复，可选切换 Reasoner |
| 可视化 | 自动生成交互式图表与商业洞察 |
| 导出 | 支持 CSV、Word、PDF 等导出路径 |
| 上下文能力 | 多轮对话与上下文记忆 |
| 性能优化 | OpenVINO 推理加速、查询缓存、硬件遥测 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化示例数据库

```bash
python tools/setup_northwind.py --user root --password <你的密码> --db northwind --sql "data/northwind.sql"
```

### 3. 启动应用

```bash
streamlit run app.py
```

### 4. 访问应用

打开浏览器访问: `http://localhost:8501`

---


## 系统要求

| 类型 | 要求 |
|------|------|
| **Python** | 3.8+ |
| **操作系统** | Windows 10/11, Linux, macOS 10.15+ |
| **内存** | 8GB RAM (推荐 16GB) |
| **CPU** |别太拉应该都行 |
| **GPU** (可选) | Intel Iris Xe / NVIDIA CUDA / AMD OpenCL |

---

## 项目结构

```text
DeepInsight-refine/
├── app.py
├── agent_core.py
├── rag_engine.py
├── visualization_engine.py
├── hardware/
├── context_memory/
├── ui/
├── tools/
├── data/
├── docs/

```

更完整的结构说明见 `docs/ARCHITECTURE.md`。

---

## 安全与约束

- SQL 安全检查以只读查询为目标，避免数据篡改风险。
- 敏感业务数据优先在本地处理，云端侧重模型推理与最小化上下文暴露。
- 当前系统主要面向分析型查询场景，不针对写入型数据库操作设计。

---

## 相关文档

| 文档 | 说明 |
|------|------|
| `docs/ARCHITECTURE.md` | 系统架构、模块划分与流程说明 |
| `docs/硬件优化技术文档.md` | OpenVINO、缓存与性能优化相关设计 |
| `docs/系统重构优化日志.md` | 重构与优化时间线、技术决策与最近更新 |

---

## 协作者

- 项目重构、改进：严秋实
- 项目负责人与核心开发：唐佳云、严秋实

