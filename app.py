"""
=============================================================================
心脏病(CHD/MI)风险评估系统 - Streamlit 主应用
=============================================================================
"""

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# 同目录导入
sys.path.append(str(Path(__file__).parent))
from risk_pipeline import (
    HeartRiskPipeline, RiskAssessment,
    OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, LOW_RISK_QUANTILE,
)
from questionnaire import (
    QUESTIONS, MANDATORY_KEYS,
    build_value_to_label_map,
)

# ============================================================================
# 0. 页面基础配置 + 升级版 CSS
# ============================================================================
st.set_page_config(
    page_title="心脏病风险评估系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ==== 全局容器 ==== */
.main .block-container { padding-top: 1.4rem; max-width: 1200px; }

/* ==== 标题区(渐变文字) ==== */
.main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #ff6b6b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem; letter-spacing: -0.5px;
}
.subtitle {
    color: #7f8c8d; margin-bottom: 1.5rem;
    font-size: 0.98rem; font-weight: 400;
}

/* ==== 风险卡片(渐变 + 阴影) ==== */
.risk-card {
    padding: 1.8rem 1.5rem;
    border-radius: 16px; color: white; text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.risk-card .risk-prob {
    font-size: 1rem; opacity: 0.95;
    margin: 0.2rem 0; font-weight: 500;
}
.risk-card .risk-level {
    font-size: 2.6rem; font-weight: 800;
    margin: 0.4rem 0; line-height: 1;
}

/* ==== 必填星号(用于内文说明,而非 widget label) ==== */
.mandatory-star { color: #c0392b; font-weight: bold; margin-right: 4px; }

/* ==== 免责声明 ==== */
.disclaimer {
    background: #fff5e6; padding: 1rem 1.2rem;
    border-left: 4px solid #f39c12; border-radius: 8px;
    color: #6b4f00; font-size: 0.88rem; line-height: 1.6;
}

/* ==== 系统信息卡片(深色渐变) ==== */
.sys-info {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: #ecf0f1; padding: 0.9rem 1.1rem;
    border-radius: 10px; font-size: 0.86rem; line-height: 1.65;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.sys-info .metric-line {
    display: flex; justify-content: space-between;
    padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sys-info .metric-line:last-child { border-bottom: none; }
.sys-info .metric-key { color: #bdc3c7; font-size: 0.83rem; }
.sys-info .metric-val { color: #2ecc71; font-weight: 600; font-size: 0.85rem; }

/* ==== 进度条颜色 ==== */
.stProgress > div > div > div > div { background-color: #e74c3c; }

/* ==== 风险刻度图例 ==== */
.legend-box {
    border-radius: 8px; padding: 8px 6px; text-align: center;
    transition: all .2s ease;
}
.legend-box:hover { transform: translateY(-2px); }
.legend-name { font-weight: 700; font-size: 0.9rem; }
.legend-range { font-size: 0.78rem; color: #555; margin-top: 2px; }

/* ==== 关键因素双向柱状图 ==== */
.contrib-wrap {
    background: #ffffff; border: 1px solid #ecf0f1;
    border-radius: 12px; padding: 12px 18px; margin-top: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.contrib-row {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 6px; border-bottom: 1px solid #f0f2f5;
    transition: background .15s;
    flex-wrap: nowrap;
}
.contrib-row:hover { background: #fafbfc; border-radius: 6px; }
.contrib-row:last-child { border-bottom: none; }

.contrib-label { flex: 0 0 240px; min-width: 200px; }
.contrib-feature-name { font-weight: 700; color: #2c3e50; font-size: 0.96rem; }
.contrib-user-choice { font-size: 0.8rem; color: #7f8c8d; margin-top: 3px; }

.contrib-imputed-tag {
    display: inline-block;
    background: #ffe6e6; color: #c0392b;
    padding: 1px 7px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 700;
    margin-left: 6px;
}

.contrib-bar-container {
    flex: 1; display: flex; height: 28px; position: relative;
    min-width: 200px;
}
.contrib-bar-half { width: 50%; display: flex; align-items: center; }
.contrib-bar-half.left  { justify-content: flex-end; }
.contrib-bar-half.right { justify-content: flex-start; }
.contrib-bar-axis {
    position: absolute; left: 50%; top: -2px; bottom: -2px;
    width: 2px; background: #bdc3c7; border-radius: 1px;
    transform: translateX(-50%); opacity: 0.6;
    background-image: repeating-linear-gradient(0deg, #bdc3c7 0, #bdc3c7 4px, transparent 4px, transparent 7px);
}
.contrib-bar-fill {
    height: 22px; transition: width .3s ease;
}

.contrib-value {
    flex: 0 0 80px; text-align: center;
    font-weight: 700; font-size: 1.02rem;
    font-variant-numeric: tabular-nums;
}

/* ==== 移动端响应式: 竖屏手机适配 ==== */
@media (max-width: 640px) {
    /* 整体容器缩减内边距 */
    .main .block-container { padding-left: 0.6rem !important; padding-right: 0.6rem !important; }

    /* 关键因素: 标签独占一行, 柱状图+数值在第二行 */
    .contrib-wrap { padding: 8px 10px; }
    .contrib-row {
        flex-wrap: wrap;
        gap: 4px 8px;
        padding: 10px 4px;
    }
    .contrib-label {
        flex: 0 0 100%;   /* 标签独占整行 */
        min-width: 0;
    }
    .contrib-feature-name { font-size: 0.88rem; }
    .contrib-user-choice  { font-size: 0.76rem; }

    .contrib-bar-container {
        flex: 1;          /* 占满剩余宽度 */
        min-width: 0;
        height: 24px;
    }
    .contrib-bar-fill { height: 18px; }

    .contrib-value {
        flex: 0 0 64px;
        font-size: 0.9rem;
    }

    /* 风险刻度图例: 文字缩小 */
    .legend-name  { font-size: 0.78rem; }
    .legend-range { font-size: 0.68rem; }

    /* 风险卡片 */
    .risk-card .risk-level { font-size: 2rem; }
    .risk-card .risk-prob  { font-size: 0.88rem; }

    /* 系统推断卡片: 单列 */
    .imputed-list-wrap { grid-template-columns: 1fr; }

    /* 标题字号 */
    .main-title { font-size: 1.7rem; }
    .subtitle   { font-size: 0.85rem; }
}

/* ==== 系统推断清单 (新增) ==== */
.imputed-list-wrap {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; margin-top: 10px;
}
.imputed-card {
    background: #fff5f5; border: 1px dashed #fadbd8;
    border-radius: 10px; padding: 14px 16px;
    display: flex; flex-direction: column; justify-content: center;
    transition: transform 0.2s;
}
.imputed-card:hover { transform: translateY(-2px); border-style: solid; }
.imputed-card .q-title { font-size: 0.82rem; color: #7f8c8d; margin-bottom: 5px; font-weight: 500; }
.imputed-card .a-val { font-size: 1.02rem; font-weight: 700; color: #c0392b; }

/* ==== BMI 计算结果框 ==== */
.bmi-result {
    background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
    border: 1px solid #fadbd8;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.92rem; color: #2c3e50;
}
.bmi-result .bmi-num {
    font-weight: 800; font-size: 1.15rem; color: #c0392b;
}

/* ==== 健康建议小行 ==== */
.tip-line {
    padding: 8px 4px;
    border-bottom: 1px dashed #ecf0f1;
}
.tip-line:last-child { border-bottom: none; }

/* ==== Tab 美化 ==== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 1. 加载推理管道(默认使用脚本所在目录,无需手动配置路径)
# ============================================================================
DEFAULT_ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR") or str(Path(__file__).parent)


@st.cache_resource(show_spinner="🔧 正在加载模型与预处理管道...")
def load_pipeline(artifacts_dir: str) -> HeartRiskPipeline:
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)


# ============================================================================
# 2. 工具函数
# ============================================================================
def _pct(x: float) -> str:
    """阈值/概率统一显示为百分比。"""
    return f"{x * 100:.1f}%"


VALUE_LABEL_MAP = build_value_to_label_map()
_ONEHOT_RE = re.compile(r'^(.+?)_(\d+\.\d+)$')


def _parse_feature(feature: str):
    """把 final_feature 名解析为 (原始字段, onehot值或None)。"""
    m = _ONEHOT_RE.match(feature)
    if m:
        return m.group(1), float(m.group(2))
    return feature, None


def _get_user_choice_text(feature: str,
                          raw_input: dict,
                          imputed_values=None) -> tuple:
    """
    返回 (label_text, is_imputed)
      label_text  : 该 feature 对应的"用户实际选择"或"系统推断出的具体选项"
      is_imputed  : 是否因不确定/拒绝回答而被系统推断
    """
    orig, _ = _parse_feature(feature)

    # ---- BMI 特殊回显:展示用户实际计算出的 BMI 数值 ----
    if orig == '_BMI5CAT' and '_bmi_value' in raw_input:
        bmi = raw_input.get('_bmi_value', 0.0)
        cat_val = raw_input.get('_BMI5CAT')
        cat_name = VALUE_LABEL_MAP.get('_BMI5CAT', {}).get(cat_val, '')
        cat_short = cat_name.split(' (')[0] if cat_name else ''
        return f"BMI = {bmi:.1f} ({cat_short})", False

    raw_val = raw_input.get(orig)

    # ---- 用户已作答:常规回显 ----
    if raw_val is not None:
        label = VALUE_LABEL_MAP.get(orig, {}).get(raw_val, str(raw_val))
        short = re.split(r'\s*\(', label)[0].strip()
        return short, False

    # ---- 用户未作答:查系统推断出的具体值 ----
    if imputed_values:
        imp_val = imputed_values.get(orig)
        if imp_val is not None:
            # 插补结果可能是 float(1.0) 或 round 后的整数,统一归一到 int 查表
            key_int = int(round(float(imp_val)))
            label = VALUE_LABEL_MAP.get(orig, {}).get(key_int, f"推断值={imp_val:g}")
            short = re.split(r'\s*\(', label)[0].strip()
            return short, True

    # 极端兜底:没有 imputed_values 或该字段不在其中
    return "系统推断", True


# ============================================================================
# 3. 侧边栏 - 已删除"文件路径配置"部分,系统信息升级
# ============================================================================
with st.sidebar:
    st.markdown("## ❤️ 系统信息")
    st.markdown(f"""
    <div class="sys-info">
      <div style="margin-bottom:0.65rem;">
        <div style="color:#f39c12; font-weight:700; font-size:0.88rem; margin-bottom:0.45rem; letter-spacing:0.3px;">📖 系统简介</div>
        <div style="color:#bdc3c7; font-size:0.81rem; line-height:1.65;">
          本系统是您的心脏健康风险预测助手。基于美国疾控中心 (CDC) 2024 年超 45 万份权威数据打造。核心驱动采用先进的 LightGBM 机器学习模型。即使遇到不确定的问题，系统也能凭借大样本规律为您智能推断，提供专属的风险评估。
        </div>
      </div>
      <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:0.6rem; margin-top:0.1rem; margin-bottom:0.65rem;">
        <div style="color:#f39c12; font-weight:700; font-size:0.88rem; margin-bottom:0.45rem; letter-spacing:0.3px;">🎯 模型指标</div>
        <div class="metric-line"><span class="metric-key">综合可信度 (AUC)</span><span class="metric-val">0.8376</span></div>
        <div class="metric-line"><span class="metric-key">患病捕捉率 (敏感度)</span><span class="metric-val">80.68%</span></div>
        <div class="metric-line"><span class="metric-key">健康识别力 (特异度)</span><span class="metric-val">70.94%</span></div>
        <div class="metric-line"><span class="metric-key">预测准确率 (Accuracy)</span><span class="metric-val">71.85%</span></div>
      </div>
      <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:0.6rem; margin-bottom:0.65rem;">
        <div style="color:#f39c12; font-weight:700; font-size:0.88rem; margin-bottom:0.45rem; letter-spacing:0.3px;">📊 风险分层</div>
        <div class="metric-line">
          <span style="color:#2ecc71; font-size:0.83rem;">🟢 低风险</span>
          <span class="metric-val">0.0% – {_pct(LOW_RISK_QUANTILE)}</span>
        </div>
        <div class="metric-line">
          <span style="color:#f1c40f; font-size:0.83rem;">🟡 中风险</span>
          <span class="metric-val">{_pct(LOW_RISK_QUANTILE)} – {_pct(OPTIMAL_THRESHOLD)}</span>
        </div>
        <div class="metric-line">
          <span style="color:#e67e22; font-size:0.83rem;">🟠 中-高风险</span>
          <span class="metric-val">{_pct(OPTIMAL_THRESHOLD)} – {_pct(HIGH_RISK_QUANTILE)}</span>
        </div>
        <div class="metric-line">
          <span style="color:#e74c3c; font-size:0.83rem;">🔴 高风险</span>
          <span class="metric-val">{_pct(HIGH_RISK_QUANTILE)} – 100.0%</span>
        </div>
      </div>
      <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:0.6rem;">
        <div style="color:#f39c12; font-weight:700; font-size:0.88rem; margin-bottom:0.45rem; letter-spacing:0.3px;">💡 使用步骤</div>
        <div style="color:#bdc3c7; font-size:0.81rem; line-height:1.9;">
          <div>① <span style="color:#ecf0f1; font-weight:600;">填写问卷</span>：遇到不确定的题目可直接选"不确定/拒绝回答"。</div>
          <div>② <span style="color:#ecf0f1; font-weight:600;">一键评估</span>：极速演算您的风险概率。</div>
          <div>③ <span style="color:#ecf0f1; font-weight:600;">获取报告</span>：查看风险等级、关键因素及改善建议。</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚠️ 免责声明")
    st.markdown("""
    <div class="disclaimer">
    本系统是基于人群流行病学数据训练的 <b>风险预测辅助工具</b>,
    <b>不构成任何形式的临床诊断、医疗建议或治疗依据</b>。<br><br>
    评估结果仅反映模型基于您所填问卷给出的统计学风险概率。<br><br>
    若您出现胸痛、气促、持续乏力等症状,请立即就医。
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 4. 主区域逻辑
# ============================================================================
st.markdown('<div class="main-title">❤️ 心脏病 (CHD/MI) 风险评估系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">基于 2024 BRFSS 数据 · LightGBM 模型 · SHAP 个体化解释 · 支持缺失值智能推断</div>',
    unsafe_allow_html=True,
)

if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None


# ----------------------------------------------------------------------------
# 4.A 身高体重 → BMI 自动计算
# ----------------------------------------------------------------------------
def _render_bmi_input(answers: dict, key_prefix: str = "bmi") -> None:
    """让用户输入身高体重,自动计算 BMI 并写入 _BMI5CAT。"""
    h_col, w_col = st.columns(2)
    h_default = float(answers.get('_height') or 170.0)
    w_default = float(answers.get('_weight') or 65.0)

    height = h_col.number_input(
        "身高 (cm)",
        min_value=80.0, max_value=250.0,
        value=h_default, step=0.5,
        key=f"{key_prefix}_h",
    )
    weight = w_col.number_input(
        "体重 (kg)",
        min_value=20.0, max_value=300.0,
        value=w_default, step=0.5,
        key=f"{key_prefix}_w",
    )

    bmi = weight / ((height / 100.0) ** 2) if height > 0 else 0.0
    if bmi < 18.5:
        cat, cat_name = 1, "体重不足"
    elif bmi < 25:
        cat, cat_name = 2, "正常体重"
    elif bmi < 30:
        cat, cat_name = 3, "超重"
    else:
        cat, cat_name = 4, "肥胖"

    answers['_BMI5CAT'] = cat
    answers['_height'] = height
    answers['_weight'] = weight
    answers['_bmi_value'] = bmi

    st.markdown(
        f'<div class="bmi-result">📊 系统自动计算: '
        f'<span class="bmi-num">BMI = {bmi:.1f}</span> '
        f'&nbsp;→&nbsp; 分类: <b>{cat_name}</b></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 4.B 问卷页
# ----------------------------------------------------------------------------
def render_form() -> None:
    st.markdown(
        "请按实际情况完成问卷。标注 <span class='mandatory-star'>*</span> 的题目为必填;"
        "其余题目若不确定可选择 <b>「不确定 / 拒绝回答」</b>,系统将自动进行智能推断。",
        unsafe_allow_html=True,
    )

    section_names = list(QUESTIONS.keys())
    tabs = st.tabs(section_names)
    answers: dict = dict(st.session_state.answers)

    for tab, sec_name in zip(tabs, section_names):
        with tab:
            for q in QUESTIONS[sec_name]:
                key = q["key"]
                is_mandatory = key in MANDATORY_KEYS

                # ---- 特殊处理 ①: BMI 改为身高体重输入 ----
                if key == "_BMI5CAT":
                    st.markdown(
                        "<span class='mandatory-star'>*</span>"
                        "<b>您的身高与体重</b> &nbsp;<i>(用于自动计算 BMI)</i>",
                        unsafe_allow_html=True,
                    )
                    _render_bmi_input(answers)
                    continue

                # ---- 特殊处理 ②: 美国州改为下拉框 ----
                if key == "_STATE":
                    labels = [opt[0] for opt in q["options"]]
                    values = [opt[1] for opt in q["options"]]
                    if key in answers and answers[key] in values:
                        default_idx = values.index(answers[key])
                    else:
                        default_idx = 0
                    chosen_label = st.selectbox(
                        label=f"\\* {q['label']}",
                        options=labels,
                        index=default_idx,
                        key=f"q_{key}",
                        help=q.get("help"),
                    )
                    answers[key] = values[labels.index(chosen_label)]
                    continue

                # ---- 普通 radio 问题 ----
                labels = [opt[0] for opt in q["options"]]
                values = [opt[1] for opt in q["options"]]
                display_label = f"\\* {q['label']}" if is_mandatory else q['label']

                if key in answers:
                    try:
                        default_idx = values.index(answers[key])
                    except ValueError:
                        default_idx = 0 if is_mandatory else len(values) - 1
                else:
                    default_idx = 0 if is_mandatory else len(values) - 1

                chosen_label = st.radio(
                    label=display_label,
                    options=labels,
                    index=default_idx,
                    horizontal=True if len(labels) <= 4 else False,
                    key=f"q_{key}",
                    help=q.get("help"),
                )
                answers[key] = values[labels.index(chosen_label)]

    st.session_state.answers = answers
    st.markdown("---")
    if st.button("🔍 开始评估风险", type="primary", use_container_width=True):
        pipeline = load_pipeline(DEFAULT_ARTIFACTS_DIR)
        with st.spinner("⏳ 模型正在评估中..."):
            st.session_state.assessment = pipeline.assess(answers, top_k=8)
            st.session_state.stage = "result"
            st.rerun()


# ----------------------------------------------------------------------------
# 4.C 结果页
# ----------------------------------------------------------------------------
def render_result() -> None:
    res: RiskAssessment = st.session_state.assessment
    if res is None:
        st.session_state.stage = "form"
        st.rerun()
        return

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"""
            <div class="risk-card" style="background:linear-gradient(135deg,{res.risk_color}dd,{res.risk_color});">
                <p class="risk-prob">您的预测患病概率</p>
                <p class="risk-level">{_pct(res.probability)}</p>
                <p class="risk-prob">风险等级: <b>{res.risk_level}</b></p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("#### 📍 概率刻度尺")
        st.progress(min(1.0, max(0.0, res.probability)))
        st.caption(
            f"决策阈值 = **{_pct(OPTIMAL_THRESHOLD)}** &nbsp;|&nbsp; "
            f"低风险 ≤ **{_pct(LOW_RISK_QUANTILE)}** &nbsp;|&nbsp; "
            f"高风险 ≥ **{_pct(HIGH_RISK_QUANTILE)}**"
        )
        _render_risk_legend(res.probability)

    st.markdown("---")
    st.markdown("### 🎯 影响您此次评估的关键因素")
    st.caption(
        "下图展示对您本次评估贡献最大的特征。"
        "🔴 **红色(右侧)** 表示推高您的风险, "
        "🟢 **绿色(左侧)** 表示降低您的风险, "
        "数值为 SHAP 归因值(对数几率空间)。"
    )
    _render_contributors(res.top_contributors, res.raw_input, res.imputed_values)

    # ---- 新增部分: 无论是否出现在关键因素中,展示所有推断出的具体选择 ----
    _render_imputed_details(res)

    st.markdown("---")
    st.markdown("### 💡 基于评估结果的健康建议")
    _render_recommendations(res)

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("⬅️ 返回修改问卷", use_container_width=True):
        st.session_state.stage = "form"
        st.rerun()
    if c2.button("🆕 重新开始评估", use_container_width=True):
        st.session_state.answers = {}
        for key in list(st.session_state.keys()):
            if key.startswith("q_") or key.startswith("bmi_"):
                del st.session_state[key]
        st.session_state.assessment = None
        st.session_state.stage = "form"
        st.rerun()


# ============================================================================
# 5. 辅助渲染函数
# ============================================================================
def _render_risk_legend(prob: float) -> None:
    bands = [
        ("低风险", 0.0, LOW_RISK_QUANTILE, '#2ca02c'),
        ("中风险", LOW_RISK_QUANTILE, OPTIMAL_THRESHOLD, '#f1c40f'),
        ("中-高风险", OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, '#ff7f0e'),
        ("高风险", HIGH_RISK_QUANTILE, 1.0, '#d62728'),
    ]
    cols = st.columns(len(bands))
    for col, (name, lo, hi, color) in zip(cols, bands):
        active = lo <= prob < hi or (hi == 1.0 and prob >= lo)
        border = '3px solid #2c3e50' if active else '1px solid #ddd'
        col.markdown(
            f"<div class='legend-box' style='border:{border}; background:{color}25;'>"
            f"<div class='legend-name' style='color:{color};'>{name}</div>"
            f"<div class='legend-range'>{_pct(lo)} – {_pct(hi)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_contributors(df: pd.DataFrame,
                         raw_input: dict,
                         imputed_values=None) -> None:
    """以双向居中柱状图样式展示 SHAP 贡献因子,并标注用户选项(含系统推断值)。"""
    if df is None or df.empty:
        st.info("当前模型不支持 SHAP 解释。")
        return

    max_abs = float(df['shap_value'].abs().max() or 1e-9)

    has_imputed = False
    rows_html = []
    for _, row in df.iterrows():
        feature      = row['feature']
        display_name = row['display_name']
        shap_val     = float(row['shap_value'])
        bar_pct      = abs(shap_val) / max_abs * 100  # 占半边的百分比

        user_choice, is_imputed = _get_user_choice_text(feature, raw_input, imputed_values)
        if is_imputed:
            has_imputed = True

        if shap_val > 0:
            color = "#e74c3c"
            sign = "+"
            bar_left_html = ""
            bar_right_html = (
                f'<div class="contrib-bar-fill" '
                f'style="width:{bar_pct:.1f}%; '
                f'background:linear-gradient(90deg,{color}cc,{color}); '
                f'border-radius:0 6px 6px 0; '
                f'box-shadow:0 1px 3px rgba(231,76,60,0.3);"></div>'
            )
        else:
            color = "#27ae60"
            sign = "-"
            bar_left_html = (
                f'<div class="contrib-bar-fill" '
                f'style="width:{bar_pct:.1f}%; '
                f'background:linear-gradient(270deg,{color}cc,{color}); '
                f'border-radius:6px 0 0 6px; '
                f'box-shadow:0 1px 3px rgba(39,174,96,0.3);"></div>'
            )
            bar_right_html = ""

        imputed_tag = (
            '<span class="contrib-imputed-tag">⚠ 系统推断</span>' if is_imputed else ''
        )

        # 注意:此处 HTML 必须无前导空格,否则 Streamlit 的 Markdown 解析器
        # 会把"4 空格开头的行"识别为 indented code block,直接把源码渲染出来。
        rows_html.append(
            f'<div class="contrib-row">'
            f'<div class="contrib-label">'
            f'<div class="contrib-feature-name">{display_name}</div>'
            f'<div class="contrib-user-choice">您的选择: {user_choice}{imputed_tag}</div>'
            f'</div>'
            f'<div class="contrib-bar-container">'
            f'<div class="contrib-bar-half left">{bar_left_html}</div>'
            f'<div class="contrib-bar-axis"></div>'
            f'<div class="contrib-bar-half right">{bar_right_html}</div>'
            f'</div>'
            f'<div class="contrib-value" style="color:{color};">{sign}{abs(shap_val):.3f}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="contrib-wrap">{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )

    if has_imputed:
        st.info(
            "ℹ️ 标有 **⚠ 系统推断** 的选项是因您填写了「不确定 / 拒绝回答」而由模型"
            "自动推断出的取值,并非您的真实作答。 "
            "详细推断结果见下方列表。"
            "为确保评估结果更加准确，建议您返回问卷，尽可能填写确定的选项。"
        )


def _render_imputed_details(res: RiskAssessment) -> None:
    """展示因不确定/拒绝回答而由系统自动推断的所有字段明细。"""
    if not res.imputed_values:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 系统智能推断清单")
    st.caption("以下为您在问卷中选择「不确定/拒绝回答」的题目，系统基于大样本统计规律推断出的最可能取值：")

    # 建立 Key -> 题干文本的映射
    key_to_label = {}
    for sec in QUESTIONS.values():
        for q in sec:
            key_to_label[q['key']] = q['label']

    cards_html = []
    for key, val in res.imputed_values.items():
        q_text = key_to_label.get(key, key)
        
        # 转换并查表映射出推断的中文标签
        val_int = int(round(float(val)))
        raw_label = VALUE_LABEL_MAP.get(key, {}).get(val_int, f"推断值={val:g}")
        # 去除括号说明
        clean_label = re.split(r'\s*\(', raw_label)[0].strip()

        cards_html.append(
            f'<div class="imputed-card">'
            f'<div class="q-title">{q_text}</div>'
            f'<div class="a-val">{clean_label}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="imputed-list-wrap">{"".join(cards_html)}</div>',
        unsafe_allow_html=True
    )


def _render_recommendations(res: RiskAssessment) -> None:
    answers = res.raw_input
    tips = []

    if answers.get('_SMOKER3') in (1, 2):
        tips.append("🚭 **戒烟**:吸烟是冠心病最可干预的强风险因素。")
    if answers.get('_RFBING6') == 2:
        tips.append("🍷 **限酒**:避免短时间内大量饮酒。")
    if answers.get('EXERANY2') == 2:
        tips.append("🏃 **规律运动**:每周累计 ≥150 分钟中等强度有氧运动。")
    if answers.get('_BMI5CAT') in (3, 4):
        tips.append("⚖️ **控制体重**:建议结合饮食与运动减重 5-10%。")
    if answers.get('DIABETE4') == 1:
        tips.append("🩸 **管好血糖**:糖尿病显著加速冠脉硬化。目标通常 <7%。")
    if answers.get('CVDSTRK3') == 1:
        tips.append("⚠️ **二级预防**:既往中风者属心血管病高危人群。")
    if answers.get('_RFHLTH') == 2:
        tips.append("🌿 **改善健康**:建议做一次全面体检。")
    if answers.get('_PHYS14D') == 3 or answers.get('_MENT14D') == 3:
        tips.append("🧠 **关注身心**:关注长期身体或情绪不佳。")
    if answers.get('PNEUVAC4') == 2 and (answers.get('_AGE_G') or 0) >= 5:
        tips.append("💉 **肺炎疫苗**:65+ 老人接种可降低心血管事件风险。")
    tips.append("🥗 **健康饮食**:多蔬菜水果、全谷物,少加工肉。")

    if res.risk_level in ("高风险", "中-高风险"):
        st.warning("⚠️ 强烈建议尽快前往心内科检查。")
    elif res.risk_level == "中风险":
        st.info("ℹ️ 建议每年做一次心血管健康体检。")
    else:
        st.success("✅ 继续保持健康生活方式。")

    for tip in tips:
        st.markdown(f"<div class='tip-line'>{tip}</div>", unsafe_allow_html=True)


# ============================================================================
# 6. 主路由
# ============================================================================
if st.session_state.stage == "form":
    render_form()
else:
    render_result()
