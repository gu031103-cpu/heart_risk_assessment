"""
=============================================================================
心脏病风险评估 - 推理管道 (Inference Pipeline)
=============================================================================
功能：
  把单个用户的原始 BRFSS 问卷答案，按照训练时的同一套规则
  (缺失编码替换 → 插补 → 数值映射 → 独热编码 → 列对齐 → 归一化)
  转换为模型可接受的 48 维特征向量，并使用已保存的 LightGBM
  最优模型给出风险概率、风险等级以及个体化 SHAP 归因解释。

  本管道严格复用训练时落盘的 .pkl 产物，杜绝训练-推理偏差。
  已增加自动拼接切片文件的功能，以便于云端部署。
=============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 全局常量：与训练脚本（数据预处理.txt / 特征筛选.txt）保持完全一致
# ----------------------------------------------------------------------------

# 训练阶段保留的 35 列原始字段顺序（target 放在最后）
ORIGINAL_COLUMNS: List[str] = [
    '_STATE', 'SEXVAR', 'MEDCOST1', 'EXERANY2', 'CVDSTRK3',
    'CHCOCNC1', 'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'DIABETE4',
    'MARITAL', 'VETERAN3', 'EMPLOY1', 'DEAF', 'BLIND',
    'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'PNEUVAC4',
    '_IMPRACE', '_RFHLTH', '_PHYS14D', '_MENT14D', '_HLTHPL2',
    '_LTASTH1', '_DRDXAR2', '_AGE_G', '_BMI5CAT',
    '_EDUCAG', '_INCOMG1', '_SMOKER3', '_CURECI3', '_RFBING6',
]
FEATURE_COLUMNS = [c for c in ORIGINAL_COLUMNS]  # 不含 _MICHD

# 缺失编码替换规则（与训练脚本完全相同）
MISSING_CODES: Dict[str, List[int]] = {
    "_STATE": [], "SEXVAR": [], "_IMPRACE": [], "_AGE_G": [],
    "MEDCOST1": [7, 9], "EXERANY2": [7, 9], "CVDSTRK3": [7, 9],
    "CHCOCNC1": [7, 9], "CHCCOPD3": [7, 9], "ADDEPEV3": [7, 9],
    "CHCKDNY2": [7, 9], "DIABETE4": [7, 9], "VETERAN3": [7, 9],
    "DEAF": [7, 9], "BLIND": [7, 9], "DECIDE": [7, 9],
    "DIFFWALK": [7, 9], "DIFFDRES": [7, 9], "DIFFALON": [7, 9],
    "PNEUVAC4": [7, 9],
    "MARITAL": [9], "EMPLOY1": [9],
    "_RFHLTH": [9], "_PHYS14D": [9], "_MENT14D": [9], "_HLTHPL2": [9],
    "_LTASTH1": [9], "_EDUCAG": [9], "_SMOKER3": [9], "_CURECI3": [9],
    "_RFBING6": [9], "_INCOMG1": [9],
    "_DRDXAR2": [], "_BMI5CAT": [],
}

# 名义型字段，会被独热编码
NOMINAL_COLS: List[str] = ['_STATE', 'MARITAL', 'EMPLOY1', '_IMPRACE', 'DIABETE4']

# 经特征筛选后保留的 48 个最终特征（顺序与训练完全一致）
FINAL_FEATURES: List[str] = [
    'SEXVAR', '_BMI5CAT', '_INCOMG1', '_SMOKER3', '_CURECI3',
    'DIABETE4_3.0', 'DIABETE4_1.0', '_IMPRACE_5.0', '_IMPRACE_2.0', '_IMPRACE_1.0',
    'EMPLOY1_8.0', 'EMPLOY1_7.0', 'EMPLOY1_6.0', 'EMPLOY1_4.0', 'EMPLOY1_2.0', 'EMPLOY1_1.0',
    'MARITAL_6.0', 'MEDCOST1', 'MARITAL_5.0', 'MARITAL_3.0', '_STATE_53.0',
    '_EDUCAG', '_RFBING6', '_AGE_G', 'CVDSTRK3', 'VETERAN3', 'DEAF', '_DRDXAR2',
    'BLIND', 'DECIDE', 'DIFFWALK', 'CHCCOPD3', 'CHCOCNC1', 'PNEUVAC4', 'ADDEPEV3',
    '_RFHLTH', '_PHYS14D', 'EXERANY2', '_MENT14D', '_HLTHPL2', '_LTASTH1',
    'CHCKDNY2', 'MARITAL_2.0', 'MARITAL_1.0', 'EMPLOY1_5.0', 'DIFFDRES',
    'DIFFALON', '_IMPRACE_6.0',
]

# 决策阈值（来自模型训练寻优结果）
OPTIMAL_THRESHOLD: float = 0.4860
# 高/低风险分位数（来自可解释性结果）
HIGH_RISK_QUANTILE: float = 0.6487
LOW_RISK_QUANTILE: float = 0.0712

# 中文可读名
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    'SEXVAR': '性别', '_BMI5CAT': 'BMI分类',
    '_INCOMG1': '收入等级', '_SMOKER3': '吸烟状态',
    '_CURECI3': '电子烟使用', 'DIABETE4_3.0': '无糖尿病',
    'DIABETE4_1.0': '确诊糖尿病', '_IMPRACE_5.0': '西班牙裔',
    '_IMPRACE_2.0': '非西裔黑人', '_IMPRACE_1.0': '非西裔白人',
    'EMPLOY1_8.0': '无法工作', 'EMPLOY1_7.0': '退休',
    'EMPLOY1_6.0': '学生', 'EMPLOY1_4.0': '失业<1年',
    'EMPLOY1_2.0': '自雇', 'EMPLOY1_1.0': '受雇就业',
    'MARITAL_6.0': '未婚同居', 'MEDCOST1': '因费用未就医',
    'MARITAL_5.0': '从未结婚', 'MARITAL_3.0': '丧偶',
    '_STATE_53.0': '居住地：华盛顿州', '_EDUCAG': '教育程度',
    '_RFBING6': '暴饮行为', '_AGE_G': '年龄分组',
    'CVDSTRK3': '中风史', 'VETERAN3': '退伍军人',
    'DEAF': '听力障碍', '_DRDXAR2': '关节炎',
    'BLIND': '视力障碍', 'DECIDE': '认知困难',
    'DIFFWALK': '行走困难', 'CHCCOPD3': '慢阻肺',
    'CHCOCNC1': '癌症史', 'PNEUVAC4': '肺炎疫苗',
    'ADDEPEV3': '抑郁症', '_RFHLTH': '自评健康良好',
    '_PHYS14D': '身体不适天数', 'EXERANY2': '体育锻炼',
    '_MENT14D': '心理不适天数', '_HLTHPL2': '医疗保险',
    '_LTASTH1': '哮喘史', 'CHCKDNY2': '肾病',
    'MARITAL_2.0': '离婚', 'MARITAL_1.0': '已婚',
    'EMPLOY1_5.0': '全职家务', 'DIFFDRES': '穿衣困难',
    'DIFFALON': '独立外出困难', '_IMPRACE_6.0': '其他种族',
}


# ----------------------------------------------------------------------------
# 特征工程映射函数：与训练脚本 apply_feature_engineering_mappings 完全一致
# ----------------------------------------------------------------------------
def apply_feature_engineering_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 应用与训练时一致的数值映射。"""
    df = df.copy()

    # 1. 常规二元映射 1=Yes, 2=No -> 1=Yes, 0=No
    binary_1_yes_2_no = [
        'SEXVAR', 'MEDCOST1', 'EXERANY2', 'CVDSTRK3', 'CHCOCNC1',
        'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'VETERAN3', 'DEAF',
        'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
        'PNEUVAC4', '_HLTHPL2', '_DRDXAR2',
    ]
    for col in binary_1_yes_2_no:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})

    # 2. 特殊二元反转
    special_flips = {
        '_LTASTH1': {1: 0, 2: 1},
        '_CURECI3': {1: 0, 2: 1},
        '_RFBING6': {1: 0, 2: 1},
        '_RFHLTH':  {1: 1, 2: 0},
    }
    for col, m in special_flips.items():
        if col in df.columns:
            df[col] = df[col].map(m)

    # 3. 三级有序映射
    for col in ['_PHYS14D', '_MENT14D']:
        if col in df.columns:
            df[col] = df[col].map({1: 0, 2: 1, 3: 2})

    # 4. 简单 -1 偏移
    for col in ['_AGE_G', '_BMI5CAT', '_EDUCAG', '_INCOMG1']:
        if col in df.columns:
            df[col] = df[col] - 1

    # 5. 吸烟状态反转：1(每天)→3(高风险), 4(从不)→0(低风险)
    if '_SMOKER3' in df.columns:
        df['_SMOKER3'] = df['_SMOKER3'].map({4: 0, 3: 1, 2: 2, 1: 3})

    return df


# ----------------------------------------------------------------------------
# 数据类
# ----------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    """单次风险评估的完整结果。"""
    probability: float                # 模型给出的患病概率 [0, 1]
    risk_level: str                   # '低风险' / '中风险' / '高风险'
    risk_color: str                   # 用于 UI 的颜色码
    threshold_used: float             # 决策阈值
    top_contributors: pd.DataFrame    # SHAP top-K 贡献因子（含中文名、值、SHAP）
    base_value: float                 # SHAP 基线（log-odds 空间）
    raw_input: Dict[str, float]       # 原始用户输入回显
    # 系统为「不确定/拒绝回答」字段智能推断后的原始 BRFSS 编码值，
    # 仅包含用户未作答的字段；用于结果页显示"系统推断出的具体选项"。
    imputed_values: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# 核心管道类
# ----------------------------------------------------------------------------
class HeartRiskPipeline:
    """
    端到端推理管道：
        from_user_input(dict) → preprocess → predict → SHAP-explain → RiskAssessment
    """

    def __init__(self, artifacts_dir: str = "."):
        """
        参数
        ----
        artifacts_dir : 存放 .pkl 产物的目录。
        """
        self.artifacts_dir = artifacts_dir
        self._load_artifacts()
        self._build_explainer()

    # ---------- 1. 加载落盘产物 (含自动拼接逻辑) ----------
    def _load_artifacts(self) -> None:
        p = lambda name: os.path.join(self.artifacts_dir, name)

        required = ['train_enc_columns.pkl', 'scaler.pkl',
                    'missing_cols_lists.pkl', 'model_LightGBM.pkl',
                    'clip_bounds.pkl']
        for f in required:
            if not os.path.exists(p(f)):
                raise FileNotFoundError(
                    f"缺少必备产物文件：{f}\n"
                    f"请确保所有 .pkl 文件都在 '{self.artifacts_dir}' 目录下。"
                )

        self.train_enc_columns: List[str] = joblib.load(p('train_enc_columns.pkl'))
        self.scaler                       = joblib.load(p('scaler.pkl'))
        self.clip_bounds: Dict            = joblib.load(p('clip_bounds.pkl'))
        self.missing_cols_lists           = joblib.load(p('missing_cols_lists.pkl'))
        self.low_missing_cols: List[str]  = self.missing_cols_lists.get('low', [])
        self.high_missing_cols: List[str] = self.missing_cols_lists.get('high', [])

        self.mode_imputer = joblib.load(p('mode_imputer.pkl')) \
            if os.path.exists(p('mode_imputer.pkl')) else None

        # ==== 核心修改：自动拼接 iter_imputer 切片逻辑 ====
        iter_path = p('iter_imputer.pkl')
        if os.path.exists(iter_path):
            # 如果存在完整的 pkl 文件，直接加载
            self.iter_imputer = joblib.load(iter_path)
        else:
            # 否则，寻找所有的 .part 切片文件
            chunk_files = sorted([f for f in os.listdir(self.artifacts_dir) if f.startswith('iter_imputer.pkl.part')])
            if chunk_files:
                # 提示正在拼接文件 (在控制台输出，帮助调试)
                print(f"正在从 {len(chunk_files)} 个切片中拼接 iter_imputer.pkl...")
                model_bytes = bytearray()
                for chunk_name in chunk_files:
                    with open(p(chunk_name), 'rb') as f:
                        model_bytes.extend(f.read())
                
                # 可选：如果你想把拼接好的文件保存下来以供下次使用，可以取消下面两行的注释
                # with open(iter_path, 'wb') as f:
                #     f.write(model_bytes)
                
                # 直接从内存中的字节流加载对象，无需再次写入硬盘，更适合云环境
                import io
                self.iter_imputer = joblib.load(io.BytesIO(model_bytes))
                print("拼接完成并成功加载！")
            else:
                self.iter_imputer = None
        # ===============================================

        self.model = joblib.load(p('model_LightGBM.pkl'))

    # ---------- 2. 构建 SHAP 解释器 ----------
    def _build_explainer(self) -> None:
        try:
            import shap
            self._shap = shap
            self.explainer = shap.TreeExplainer(self.model)
        except ImportError:
            self._shap = None
            self.explainer = None

    # ---------- 3. 把单条用户回答转换为 1×35 DataFrame ----------
    @staticmethod
    def user_input_to_dataframe(answers: Dict[str, float]) -> pd.DataFrame:
        """
        参数
        ----
        answers : 形如 {'SEXVAR': 1, '_AGE_G': None, ...} 的字典
        """
        # 安全地将前端传来的 None 直接视为 np.nan
        row = {col: (np.nan if answers.get(col) is None else answers.get(col)) 
               for col in FEATURE_COLUMNS}
        
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # 转换为 float 类型，如果全是 NaN 也会妥善保留为 np.nan 格式以供后续 Imputer 读取
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col, miss_vals in MISSING_CODES.items():
            if col in df.columns and miss_vals:
                df[col] = df[col].replace({v: np.nan for v in miss_vals})
        return df

    # ---------- 4. 执行训练同款预处理 ----------
    def _preprocess(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        返回
        ----
        X_scaled       : 48 维已归一化特征向量（供模型预测）
        imputed_values : {字段: 插补后原始 BRFSS 编码值} 仅包含原本为 NaN 的列
        """
        df = df_raw.copy()

        # 记录原本是 NaN（用户未作答或缺失编码）的列，用于追溯插补值
        originally_nan_cols: List[str] = df.columns[df.iloc[0].isna()].tolist()

        # 4.1 低缺失列：众数插补
        if self.low_missing_cols and self.mode_imputer is not None:
            df.loc[:, self.low_missing_cols] = self.mode_imputer.transform(
                df[self.low_missing_cols])

        # 4.2 高缺失列：迭代插补 + 训练时记录的 clip 边界
        if self.high_missing_cols and self.iter_imputer is not None:
            predictor_cols = self.low_missing_cols + self.high_missing_cols
            imputed = pd.DataFrame(
                self.iter_imputer.transform(df[predictor_cols]),
                columns=predictor_cols, index=df.index)
            for col in self.high_missing_cols:
                lo = self.clip_bounds[col]['min']
                hi = self.clip_bounds[col]['max']
                df[col] = np.clip(np.round(imputed[col]), lo, hi)

        # 4.3 兜底：若仍存在 NaN（极端情况），用 0 填充并发出警告
        if df.isnull().any().any():
            df = df.fillna(0)

        # ★★ 在数值映射/独热编码之前捕获插补后、保持原始 BRFSS 编码的数值 ★★
        imputed_values: Dict[str, float] = {
            col: float(df.iloc[0][col]) for col in originally_nan_cols
        }

        # 4.4 数值映射
        df = apply_feature_engineering_mappings(df)

        # 4.5 独热编码 (修复：强制转 float 以匹配训练时的 .0 列名后缀)
        nominal = [c for c in NOMINAL_COLS if c in df.columns]
        for col in nominal:
            df[col] = df[col].astype(float)
        df_enc = pd.get_dummies(df, columns=nominal, drop_first=False)

        # 4.6 列对齐
        df_enc = df_enc.reindex(columns=self.train_enc_columns, fill_value=0)
        df_enc = df_enc.astype(float)

        # 4.7 MinMax 归一化
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_enc),
            columns=df_enc.columns, index=df_enc.index)

        # 4.8 选取最终 48 维特征
        return df_scaled[FINAL_FEATURES], imputed_values

    # ---------- 5. 风险等级分层 ----------
    @staticmethod
    def _classify_risk(prob: float) -> Tuple[str, str]:
        """根据概率返回 (等级, 颜色)。"""
        if prob >= HIGH_RISK_QUANTILE:
            return '高风险', '#d62728'
        if prob >= OPTIMAL_THRESHOLD:
            return '中-高风险', '#ff7f0e'
        if prob >= LOW_RISK_QUANTILE:
            return '中风险', '#f1c40f'
        return '低风险', '#2ca02c'

    # ---------- 6. 主入口 ----------
    def assess(self,
               answers: Dict[str, float],
               top_k: int = 8) -> RiskAssessment:
        """
        对单个用户作风险评估，返回 RiskAssessment。
        """
        # 6.1 转 DataFrame + 完整预处理（同时拿到插补后的原始值）
        df_raw = self.user_input_to_dataframe(answers)
        X, imputed_values = self._preprocess(df_raw)

        # 6.2 模型概率 & 风险等级
        prob = float(self.model.predict_proba(X)[0, 1])
        level, color = self._classify_risk(prob)

        # 6.3 SHAP 个体化解释
        contributors = self._explain_individual(X, top_k=top_k)
        base_val = float(self.explainer.expected_value) if self.explainer else 0.0
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.array(base_val).ravel()[-1])

        return RiskAssessment(
            probability     = prob,
            risk_level      = level,
            risk_color      = color,
            threshold_used  = OPTIMAL_THRESHOLD,
            top_contributors= contributors,
            base_value      = base_val,
            raw_input       = answers,
            imputed_values  = imputed_values,
        )

    # ---------- 7. SHAP 解释 ----------
    def _explain_individual(self, X: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        返回 top-K 贡献因子表。
        """
        if self.explainer is None:
            # 退化方案：用模型自带 feature_importances_
            imp = getattr(self.model, 'feature_importances_', None)
            if imp is None:
                return pd.DataFrame()
            df = pd.DataFrame({
                'feature': FINAL_FEATURES,
                'display_name': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FINAL_FEATURES],
                'scaled_value': X.iloc[0].values,
                'shap_value': imp / (imp.max() + 1e-9),
                'direction': '未知',
            })
            return df.sort_values('shap_value', key=abs, ascending=False).head(top_k)

        sv_raw = self.explainer.shap_values(X)
        if isinstance(sv_raw, list):
            sv = sv_raw[1] if len(sv_raw) == 2 else sv_raw[0]
        else:
            sv = sv_raw
        sv = np.asarray(sv).reshape(-1)

        df = pd.DataFrame({
            'feature': FINAL_FEATURES,
            'display_name': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FINAL_FEATURES],
            'scaled_value': X.iloc[0].values.astype(float),
            'shap_value': sv.astype(float),
        })
        df['direction'] = np.where(df['shap_value'] > 0, '↑ 推高风险',
                          np.where(df['shap_value'] < 0, '↓ 降低风险', '— 无影响'))

        # 仅保留实际激活的特征
        onehot_active = ~df['feature'].str.contains(r'_\d+\.\d+$') | (df['scaled_value'] > 0)
        df = df[onehot_active]

        # 按 |SHAP| 排序取 top-K
        df = df.reindex(df['shap_value'].abs().sort_values(ascending=False).index)
        return df.head(top_k).reset_index(drop=True)
