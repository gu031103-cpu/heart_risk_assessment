"""
快速冒烟测试：用一份小规模虚拟数据生成同款 .pkl 产物，
然后让 HeartRiskPipeline 端到端跑一遍，验证：
  - 列对齐正确
  - 独热编码符合训练命名
  - 单条样本 predict_proba & SHAP 解释能跑通
注意：本测试只为校验代码结构，模型与统计意义无关。
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from risk_pipeline import (
    HeartRiskPipeline, FEATURE_COLUMNS, NOMINAL_COLS, FINAL_FEATURES,
    apply_feature_engineering_mappings, MISSING_CODES,
)
from questionnaire import get_default_answers


TEST_DIR = "/tmp/heart_risk_test_artifacts"


def build_mock_artifacts(n: int = 2000) -> None:
    """伪造一份 BRFSS 风格的训练集，跑一遍训练流程的核心步骤，落盘 .pkl。"""
    os.makedirs(TEST_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # —— 1. 构造合成训练集（每个字段按合理候选编码采样）
    sampling_pool = {
        '_STATE': [1, 6, 12, 17, 36, 48, 53],
        'SEXVAR': [1, 2],
        'MEDCOST1': [1, 2], 'EXERANY2': [1, 2], 'CVDSTRK3': [1, 2],
        'CHCOCNC1': [1, 2], 'CHCCOPD3': [1, 2], 'ADDEPEV3': [1, 2],
        'CHCKDNY2': [1, 2],
        'DIABETE4': [1, 2, 3, 4],
        'MARITAL': [1, 2, 3, 4, 5, 6],
        'VETERAN3': [1, 2],
        'EMPLOY1': [1, 2, 3, 4, 5, 6, 7, 8],
        'DEAF': [1, 2], 'BLIND': [1, 2], 'DECIDE': [1, 2],
        'DIFFWALK': [1, 2], 'DIFFDRES': [1, 2], 'DIFFALON': [1, 2],
        'PNEUVAC4': [1, 2],
        '_IMPRACE': [1, 2, 3, 4, 5, 6],
        '_RFHLTH': [1, 2], '_PHYS14D': [1, 2, 3], '_MENT14D': [1, 2, 3],
        '_HLTHPL2': [1, 2], '_LTASTH1': [1, 2], '_DRDXAR2': [1, 2],
        '_AGE_G': [1, 2, 3, 4, 5, 6],
        '_BMI5CAT': [1, 2, 3, 4],
        '_EDUCAG': [1, 2, 3, 4],
        '_INCOMG1': [1, 2, 3, 4, 5, 6, 7],
        '_SMOKER3': [1, 2, 3, 4],
        '_CURECI3': [1, 2],
        '_RFBING6': [1, 2],
    }
    df = pd.DataFrame({c: rng.choice(sampling_pool[c], size=n)
                       for c in FEATURE_COLUMNS})
    # 故意制造 1% 缺失，触发 mode_imputer 路径
    for col in df.columns:
        mask = rng.random(n) < 0.01
        df.loc[mask, col] = np.nan
    # 合成一个有偏 label：年龄越大 + 吸烟 + 糖尿病 → 更高概率
    score = (df['_AGE_G'].fillna(3) - 3) + \
            (df['_SMOKER3'].fillna(4) <= 2).astype(int) * 1.5 + \
            (df['DIABETE4'].fillna(3) == 1).astype(int) * 1.2
    prob = 1 / (1 + np.exp(-(score - score.mean()) * 0.5))
    y = (rng.random(n) < prob).astype(int)

    # —— 2. 模仿训练脚本：先替换缺失编码 → 然后只用众数填充（简化）
    for col, miss in MISSING_CODES.items():
        if miss:
            df[col] = df[col].replace({v: np.nan for v in miss})

    low_cols = [c for c in df.columns if df[c].isnull().any()]
    high_cols: list = []  # 测试时不模拟高缺失分支
    mode_imp = SimpleImputer(strategy='most_frequent')
    if low_cols:
        df[low_cols] = mode_imp.fit_transform(df[low_cols])

    # —— 3. 数值映射 & 独热
    df = apply_feature_engineering_mappings(df)
    df_enc = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=False)
    df_enc = df_enc.astype(float)

    # —— 4. MinMax
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_enc),
                             columns=df_enc.columns, index=df_enc.index)

    # —— 5. 取最终 48 特征。如有些列不存在（合成数据未覆盖），补 0
    for col in FINAL_FEATURES:
        if col not in df_scaled.columns:
            df_scaled[col] = 0.0
    X_final = df_scaled[FINAL_FEATURES]

    # —— 6. 训练一棵小 LightGBM
    pos_w = (y == 0).sum() / max(1, (y == 1).sum())
    model = LGBMClassifier(n_estimators=80, num_leaves=15, learning_rate=0.1,
                           scale_pos_weight=pos_w, random_state=42,
                           verbose=-1, n_jobs=-1)
    model.fit(X_final, y)

    # —— 7. 落盘
    joblib.dump({'low': low_cols, 'high': high_cols},
                os.path.join(TEST_DIR, 'missing_cols_lists.pkl'))
    joblib.dump(mode_imp,             os.path.join(TEST_DIR, 'mode_imputer.pkl'))
    joblib.dump({},                   os.path.join(TEST_DIR, 'clip_bounds.pkl'))
    joblib.dump(list(df_enc.columns), os.path.join(TEST_DIR, 'train_enc_columns.pkl'))
    joblib.dump(scaler,               os.path.join(TEST_DIR, 'scaler.pkl'))
    joblib.dump(model,                os.path.join(TEST_DIR, 'model_LightGBM.pkl'))
    print(f"[mock] 模拟产物已落盘到 {TEST_DIR}/")


def run_pipeline_test() -> None:
    pipe = HeartRiskPipeline(artifacts_dir=TEST_DIR)
    print(f"[ok] 管道加载成功，最终特征数={len(FINAL_FEATURES)}")

    # 用默认答案跑一次
    answers = get_default_answers()
    res = pipe.assess(answers, top_k=5)
    print(f"[ok] 默认低风险样例: 概率={res.probability:.4f}, 等级={res.risk_level}")
    print("     Top contributors:")
    for _, row in res.top_contributors.iterrows():
        print(f"        {row['display_name']:<20s}  shap={row['shap_value']:+.4f}")

    # 高风险样例
    high_risk = {
        '_AGE_G': 6, 'SEXVAR': 1, '_IMPRACE': 1, 'MARITAL': 3,
        '_EDUCAG': 2, '_INCOMG1': 2, 'EMPLOY1': 7, '_BMI5CAT': 4,
        'VETERAN3': 1, '_STATE': 6,
        'EXERANY2': 2, '_SMOKER3': 1, '_CURECI3': 1, '_RFBING6': 1,
        'CVDSTRK3': 1, 'DIABETE4': 1, 'CHCOCNC1': 2, 'CHCCOPD3': 1,
        'CHCKDNY2': 1, 'ADDEPEV3': 1, '_DRDXAR2': 1, '_LTASTH1': 2,
        '_RFHLTH': 2, '_PHYS14D': 3, '_MENT14D': 2,
        'DEAF': 1, 'BLIND': 2, 'DECIDE': 1, 'DIFFWALK': 1,
        'DIFFDRES': 2, 'DIFFALON': 1,
        '_HLTHPL2': 1, 'MEDCOST1': 1, 'PNEUVAC4': 2,
    }
    res2 = pipe.assess(high_risk, top_k=5)
    print(f"[ok] 高风险样例:     概率={res2.probability:.4f}, 等级={res2.risk_level}")

    # 边界用例：故意留几个字段不填，触发 imputer
    partial = dict(high_risk)
    del partial['_BMI5CAT']
    del partial['_INCOMG1']
    res3 = pipe.assess(partial, top_k=5)
    print(f"[ok] 缺失字段样例:   概率={res3.probability:.4f}, 等级={res3.risk_level}")

    print("\n✅ 所有冒烟测试通过！")


if __name__ == "__main__":
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    build_mock_artifacts()
    run_pipeline_test()
