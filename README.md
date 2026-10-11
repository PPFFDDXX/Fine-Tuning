# 客服意图识别与订单号抽取：Qwen2.5 + LoRA

输入一条英文电商客服消息，输出五类意图之一和消息中**明确出现**的订单号。项目从固定版本的 Qwen2.5-1.5B-Instruct 出发，构建并审核训练数据，使用 LoRA 做监督微调（SFT），再以相同提示词和生成设置比较 Base 与 SFT。最终还用 50 条人工标注的真实客服推文检查跨数据来源的表现。

```text
输入：Where is order ORD-1234567890?
输出：{"intent":"track_order","order_id":"ORD-1234567890"}

输入：Can I cancel my order?
输出：{"intent":"cancel_order","order_id":null}
```

允许的意图是 `cancel_order`、`change_order`、`track_order`、`get_refund`、`track_refund`。输出必须是只有 `intent` 和 `order_id` 两个键的 JSON；没有明确订单号时使用 `null`。模型执行分类和字段抽取，不替客服回答或处理订单。

## 已取得的结果

| 数据集 | 模型 | JSON 合格率 | 意图准确率 | 意图 Macro-F1 | 订单号精确准确率 | 有订单号 F1 | 联合准确率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 合成独立测试集（120 条） | Base | 0.758 | 0.675 | 0.774 | 0.567 | 0.756 | 0.508 |
| 合成独立测试集（120 条） | SFT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 真实外部样本（50 条） | Base | 0.700 | 0.440 | 0.462 | 0.140 | 0.233 | 0.060 |
| 真实外部样本（50 条） | SFT | 1.000 | 0.840 | 0.823 | 0.900 | 0.800 | 0.740 |

**联合准确率**要求意图和订单号同时正确；无效 JSON 也算错误。外部集上，SFT 比 Base 多正确处理 34 条联合任务（3/50 → 37/50）。这些数字来自已保存的逐条预测和评分结果，见 `outputs/customer_support/{base_test,sft_test,base_external,sft_external}/`。模型、数据、配置与报告的 SHA-256 记录在 [`reports/customer_support_release/freeze.json`](reports/customer_support_release/freeze.json)。

外部集仍有 13 条联合错误：7 条退款进度查询被判为退款申请，1 条物流查询被判为退款进度查询；另外 5 条把 @账号数字或商品名当成了订单号。10 条明确给出 Amazon 格式订单号的外部消息均抽取正确。现有 API 只检查订单号是否出现在原文中，因此不能拦住上述“原文存在，但并非订单号”的错误。

## 数据与实验设计

- **训练来源**：[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)，固定修订版本 `430d1a89bd93bd1fa23c16f29dd53e73f0087443`。筛选五类英文消息，处理重复与冲突，人工审核后得到训练 560 条、验证 120 条、合成测试 120 条。数据审核记录在 `data/processed/customer_support/v2/`。来源清单记录许可为 `cdla-sharing-1.0`，实际复用请核对来源条款。
- **外部来源**：[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)。从客户发给 `@AmazonHelp` 的真实消息中人工选择五类各 10 条，保留原始 `tweet_id`；10 条含明确订单号，其余 40 条没有明确订单号。来源清单记录许可为 `CC BY-NC-SA 4.0`，实际复用请核对来源条款。原始 CSV 和含客户原文的评测产物位于 Git 忽略目录。
- **隔离方式**：训练与参数选择使用训练/验证集；合成测试集与外部集仅用于评估。外部 50 条已经用于错误分析，若以后据此修改模型，应把它们视为开发样本，另备新的独立测试集。

模型配置见 [`configs/inference.yaml`](configs/inference.yaml)，Base 修订版本固定为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。正式 SFT 配置见 [`configs/sft_customer_support_v2.yaml`](configs/sft_customer_support_v2.yaml)：LoRA `r=8`、`alpha=16`、`dropout=0.05`，训练 1 个 epoch，随机种子 42。正式导出的 adapter 位于 `outputs/customer_support/sft_v2/full-20261001T034406348108Z/adapter/`，独立重载检查已通过。项目未对客服任务运行 DPO，也未进行 LoRA 参数搜索。

## 运行与复核

先创建并安装项目隔离环境（本机默认 Python 3.14，请显式使用已验证的 Python 3.13）：

```powershell
D:\miniconda\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 包含数据准备、SFT/DPO、评估、推理及 API 依赖，固定核心库到已有训练环境版本，并安装 CUDA 12.8 版 PyTorch。后续命令中的 `D:\miniconda\python.exe` 可替换为 `.\.venv\Scripts\python.exe`。

以下命令在项目根目录的 PowerShell 中运行。示例使用本机 `D:\miniconda\python.exe`；其他环境请换成安装了兼容 CUDA 版 PyTorch 的 Python，并安装 [`requirements-ultrafeedback.txt`](requirements-ultrafeedback.txt) 与 [`requirements-api.txt`](requirements-api.txt)。本地推理还需要已缓存的固定版本 Base 权重和正式 SFT adapter；它们未提交到 Git。

先检查已冻结的模型、数据和评测文件：

```powershell
D:\miniconda\python.exe scripts/freeze_customer_support_release.py --verify
```

如需重新运行 50 条外部样本的 Base/SFT 评测，本地须已有原始 `data/external/customer_support_twitter/twcs.csv`；先用 `scripts/prepare_external_twitter_50.py` 重建选定样本，再运行：

```powershell
D:\miniconda\python.exe scripts/prepare_external_twitter_50.py
D:\miniconda\python.exe scripts/evaluate_customer_support.py run --kind base --split external --output outputs/customer_support/base_external --max-new-tokens 96
D:\miniconda\python.exe scripts/evaluate_customer_support.py run --kind sft --split external --training-run outputs/customer_support/sft_v2/full-20261001T034406348108Z --output outputs/customer_support/sft_external --max-new-tokens 96
D:\miniconda\python.exe scripts/compare_customer_support.py --base outputs/customer_support/base_external --sft outputs/customer_support/sft_external
```

评测脚本会校验已有输出的配置与样本，支持中断后续跑；`report.md` 是汇总，`scored_predictions.jsonl` 包含逐条预测与错误标记。要从头训练，请先用 `scripts/prepare_customer_support.py` 在**新的输出目录**准备并审核数据，完成 `publish` 后再运行训练检查及 `scripts/train_sft.py smoke/full`；现有冻结数据目录不可覆盖。当前冻结版本以 `freeze.json` 指向的产物为准。

## 本地 API

安装 API 依赖后，在一个 PowerShell 窗口启动**已训练的 SFT**（首次加载模型需要时间）：

```powershell
D:\miniconda\python.exe scripts/serve.py --kind sft --training-run outputs/customer_support/sft_v2/full-20261001T034406348108Z --config configs/inference.yaml --system-prompt prompts/system_customer_support.txt --customer-support --host 127.0.0.1 --port 8000
```

在另一个窗口发送请求：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze -ContentType 'application/json' -Body '{"message":"Where is order ORD-1234567890?"}'
```

`POST /analyze` 返回 `request_id`、`intent`、`order_id` 和生成耗时；`GET /health` 返回模型状态。接口实现见 [`src/bitext_support/inference/api.py`](src/bitext_support/inference/api.py)。这里提供的是本地服务启动方法；项目没有声称已部署公网服务。由于订单号类型误判尚存在，接口结果不应直接触发真实订单操作。

## 项目结构与限制

[`scripts/`](scripts/) 放运行入口；[`src/bitext_support/`](src/bitext_support/) 放数据处理、训练、评测和推理实现；[`configs/`](configs/) 与 [`prompts/`](prompts/) 固定实验设置；[`data/`](data/) 保存来源和分割；[`outputs/`](outputs/) 保存模型与逐条结果；[`reports/customer_support_release/`](reports/customer_support_release/) 保存冻结清单。

合成测试集包含模板化英文表达和受控订单号，不能替代真实流量。外部集只有 50 条、集中于一个品牌的推文，且五类意图经过人工平衡；74% 联合准确率仅描述这组样本，不代表生产环境准确率。真实消息中的退款状态表达和非订单号实体是当前主要薄弱点。
