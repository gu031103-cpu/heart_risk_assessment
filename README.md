# ❤️ 心脏病(CHD/MI)风险评估系统

一套基于 **2024 BRFSS** 调查数据训练、以 **LightGBM** 为底层模型(AUC = 0.8376)、
通过 **Streamlit** 提供问卷式交互界面的心血管疾病风险预测辅助工具。
系统支持个体化 **SHAP** 归因解释、四级风险分层与可执行健康建议。

> ⚠️ **本系统仅为辅助参考工具,不是临床诊断依据。**

---

## 1. 目录结构

```
heart_risk_app/
├─ app.py                    # Streamlit 主应用 (UI + 状态机)
├─ risk_pipeline.py          # 推理管道:预处理 → 模型预测 → SHAP 解释
├─ questionnaire.py          # 35 题问卷配置(题干、选项、BRFSS 编码)
├─ test_pipeline.py          # 离线冒烟测试(用模拟产物验证管道)
├─ requirements.txt
└─ README.md (本文件)
```

## 2. 运行前准备

把训练阶段保存的 **7 个 `.pkl` 产物文件** 全部放到与 `app.py` 同一目录(或在
启动时通过侧边栏"文件路径配置"指向自定义目录):

| 文件名 | 由哪个脚本生成 | 作用 |
| --- | --- | --- |
| `model_LightGBM.pkl` | 模型训练.txt / 可解释性.txt | 最优 LightGBM 二分类模型 |
| `train_enc_columns.pkl` | 数据预处理.txt | 训练集独热编码后的列名顺序 |
| `scaler.pkl` | 数据预处理.txt | MinMaxScaler |
| `mode_imputer.pkl` | 数据预处理.txt | 低缺失字段众数插补器 |
| `iter_imputer.pkl` | 数据预处理.txt | 高缺失字段迭代插补器(可选) |
| `clip_bounds.pkl` | 数据预处理.txt | 高缺失列离散化截断边界 |
| `missing_cols_lists.pkl` | 数据预处理.txt | 低/高缺失列分类 |

## 3. 安装与启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Streamlit
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

> 也可以在启动前用环境变量指定产物路径:
> `ARTIFACTS_DIR=/path/to/pkls streamlit run app.py`

## 4. 推理流程(关键)

为防止训练-推理偏差,本系统对单个用户输入执行**与训练阶段完全相同**的处理:

```
原始问卷答案 (BRFSS 1-9 编码)
    ↓ MISSING_CODES 替换 (7,9 → NaN)
    ↓ mode_imputer / iter_imputer (复用训练拟合的对象)
    ↓ apply_feature_engineering_mappings (二元/有序/反转)
    ↓ pd.get_dummies + reindex(train_enc_columns)  ← 列对齐至关重要
    ↓ MinMaxScaler.transform
    ↓ 选取 48 个最终特征
    ↓ LightGBM.predict_proba  →  概率
    ↓ TreeExplainer.shap_values →  Top-K 个体化贡献
```

这一切都封装在 `risk_pipeline.HeartRiskPipeline.assess()` 中,你也可以在不启动
Streamlit 的情况下直接调用它做批量评估或 API 集成。

## 5. 风险分层规则

| 等级 | 概率区间 | 颜色 | 说明 |
| --- | --- | --- | --- |
| 低风险 | `< 0.0712` | 🟢 绿 | 训练集底 20% 分位 |
| 中风险 | `0.0712 – 0.486` | 🟡 黄 | 介于低分位与决策阈值之间 |
| 中-高风险 | `0.486 – 0.6487` | 🟠 橙 | ≥ 决策阈值,达到"预测患病"条件 |
| 高风险 | `≥ 0.6487` | 🔴 红 | 训练集顶 20% 分位,临床警戒区 |

> 决策阈值 0.486 来自 `模型训练.txt` 中"Accuracy ≥ 0.72 约束下最大化 Recall"
> 的阈值寻优结果;两个分位数来自 `可解释性.txt` 中的高/低风险人群划分。

## 6. 离线快速验证

如果你想在没有真实 `.pkl` 的环境下先验证代码:

```bash
python test_pipeline.py
```

脚本会用合成数据生成一份临时产物落到 `/tmp/heart_risk_test_artifacts/`,
然后跑过 3 个测试样例(默认/高风险/带缺失字段),输出预测概率与 SHAP top-K。

## 7. 程序化调用示例

```python
from risk_pipeline import HeartRiskPipeline

pipe = HeartRiskPipeline(artifacts_dir="./artifacts")

answers = {
    '_AGE_G': 6, 'SEXVAR': 1, 'CVDSTRK3': 1, 'DIABETE4': 1,
    '_SMOKER3': 1, '_BMI5CAT': 4,
    # ... 其他 29 项答案
}

result = pipe.assess(answers, top_k=8)
print(f"概率 = {result.probability:.4f}")
print(f"等级 = {result.risk_level}")
print(result.top_contributors[['display_name', 'shap_value']])
```

## 8. 已知限制 / 未来工作

- **模型仅在美国成年人群(BRFSS)上训练**,其它人群迁移性需进一步验证。
- 当前 SHAP 解释为单模型(LightGBM)版本;若以后切换到训练流水线最后产出的
  Stacking 模型(`stacking_meta_model.pkl`),需在 `risk_pipeline.py` 中改为
  `KernelExplainer` 或对各基模型分别求 SHAP 后加权聚合。
- UI 暂不支持多语言切换、用户账户与历史评估对比,后续可扩展。

## 9. 免责声明

本系统输出的风险概率属于**统计学预测**,不替代任何形式的医学检查与诊断。
任何健康决策请咨询执业医师。出现急性胸痛、气促等症状时请立即拨打急救电话。
