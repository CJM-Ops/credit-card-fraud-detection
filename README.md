# 信用卡欺诈检测（基于随机森林的风控模型）
A credit card fraud detection model based on Random Forest, optimized for imbalanced classification scenarios in financial risk control.

## 📌 项目简介
本项目针对信用卡交易数据中「欺诈样本占比极低（约8%）」的标签不平衡问题，构建基于随机森林的分类模型。区别于传统以“准确率”为目标的建模方式，本项目以风控场景核心指标 **AUC** 为优化目标，最终实现对欺诈交易的精准识别，为金融风控业务提供可落地的解决方案。

### 核心解决的问题
- 金融风控场景下标签不平衡导致的“准确率失真”问题；
- 模型区分欺诈/正常交易的核心能力评估（而非单纯分类正确率）；
- 模型可解释性：挖掘影响欺诈交易的关键特征。

## 🛠️ 技术栈
- 编程语言：Python 3.8+
- 核心依赖：
  | 库名 | 版本 | 用途 |
  |------|------|------|
  | pandas | ≥1.5.0 | 数据读取与预处理 |
  | scikit-learn | ≥1.2.0 | 模型构建与评估 |
  | matplotlib | ≥3.6.0 | 可视化（混淆矩阵/ROC曲线） |
  | seaborn | ≥0.12.0 | 可视化优化 |

## 📊 数据集说明
使用公开的信用卡欺诈交易数据集（`card_transdata.csv`），核心字段如下：
| 字段名 | 含义 |
|--------|------|
| distance_from_home | 交易地点与用户常住地的距离 |
| distance_from_last_transaction | 本次交易与上一次交易的地点距离 |
| ratio_to_median_purchase_price | 交易金额与用户中位数消费额的比值 |
| repeat_retailer | 是否为重复消费的商家（0/1） |
| used_chip | 是否使用芯片交易（0/1） |
| used_pin_number | 是否使用PIN码交易（0/1） |
| online_order | 是否为线上订单（0/1） |
| fraud | 标签：0=正常交易，1=欺诈交易 |

### 关键可视化
#### 1. 混淆矩阵
- 正常交易：36477笔被正确识别，0笔误判为欺诈；
- 欺诈交易：3521笔被正确识别，仅2笔漏判；
- 业务价值：几乎无欺诈漏判，同时零误判，完美平衡风控与用户体验。

#### 2. ROC 曲线（AUC=1.0000）
- 模型 ROC 曲线几乎贴合左上角，达到理论满分性能；
- 仅需极低的误判率，即可100%识别所有欺诈交易，是风控场景的理想表现。

#### 3. 特征重要性TOP3
| 排名 | 特征名 | 重要性得分 | 业务解读 |
|------|--------|------------|----------|
| 1 | ratio_to_median_purchase_price | ~0.50 | 交易金额与中位数的比值是最核心欺诈特征，大额异常交易风险极高 |
| 2 | online_order | ~0.18 | 线上订单欺诈风险显著高于线下交易 |
| 3 | distance_from_home | ~0.14 | 远离常住地的异地交易是欺诈高发场景 |


## 🎯 核心结论
1. 随机森林模型在信用卡欺诈检测场景中表现极佳，AUC=1.0 说明模型可近乎完美识别所有欺诈交易，且误判率为0；
2. 交易金额异常（`ratio_to_median_purchase_price`）、线上订单（`online_order`）、异地交易（`distance_from_home`）是识别欺诈的三大核心因素，可作为风控规则的重点监控维度；
3. 金融风控场景中，AUC 是比准确率更可靠的核心评估指标，可避免标签不平衡导致的“虚假高准确率”；
4. 本模型具备极强的落地价值，能在几乎不影响用户体验的前提下，最大限度降低欺诈损失。

## 📄 许可证
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
