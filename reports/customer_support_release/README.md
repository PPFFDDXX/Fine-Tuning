# 客服模型版本冻结记录

本目录的 `freeze.json` 固定了本项目已完成评测的版本，不代表重新训练或重新评测。基座模型为 `Qwen/Qwen2.5-1.5B-Instruct`，修订版本 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；SFT 为 `outputs/customer_support/sft_v2/full-20261001T034406348108Z` 中通过独立重载检查的 LoRA adapter。训练数据、验证集、独立测试集、50 条外部真实推文、提示词、推理配置、adapter 文件及六组评测产物均记录了 SHA-256。原始推文和模型权重仍保存在本地被 Git 忽略的目录；公开清单仅含路径、哈希与汇总分数。

| 指标 | Base 外部集 | SFT 外部集 |
|---|---:|---:|
| JSON 格式合格率 | 0.700 | 1.000 |
| 意图准确率 | 0.440 | 0.840 |
| 意图 Macro-F1 | 0.462 | 0.823 |
| 订单号精确准确率 | 0.140 | 0.900 |
| 有订单号 F1 | 0.233 | 0.800 |
| 联合准确率 | 0.060 | 0.740 |

在项目根目录复核冻结状态：

```powershell
D:\miniconda\python.exe scripts/freeze_customer_support_release.py --verify
```

脚本会校验本地文件、报告中的预测哈希、重新计算主要指标，并确认 Base/SFT 使用同一数据与推理配置、三个 SFT 评测使用同一 adapter。文件发生变化时复核失败；脚本不会自动覆盖已冻结清单。

外部样本只有 50 条，来自同一个公开客服推文数据集，且已用于错误分析。后续若根据这些错误改数据或调参，应把它们视为开发样本，另备独立测试集评价新模型。逐条错误见 `reports/customer_support_external_error_analysis.md`。
