"""
=============================================================================
心脏病(CHD/MI)风险评估系统 - Streamlit 主应用
=============================================================================
"""

from __future__ import annotations
import os
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
from questionnaire import QUESTIONS, MANDATORY_KEYS

# ============================================================================
# 0. 页面基础配置与原始 CSS
# ============================================================================
st.set_page_config(
    page_title="心脏病风险评估系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b; margin-bottom: 0.2rem; }
    .subtitle   { color: #666; margin-bottom: 1.5rem; }
    .risk-card  { padding: 1.2rem 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .risk-level { font-size: 2rem; font-weight: 700; margin: 0; }
    .risk-prob  { font-size: 1.1rem; opacity: 0.95; }
    .mandatory-star { color: #c0392b; font-weight: bold; margin-right: 4px; }
    .disclaimer { background: #fff5e6; padding: 1rem 1.2rem; border-left: 4px solid #f39c12; border-radius: 6px; color: #6b4f00; font-size: 0.92rem; }
    .stProgress > div > div > div > div { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. 加载推理管道
# ============================================================================
@st.cache_resource(show_spinner="🔧 正在加载模型与预处理管道...")
def load_pipeline(artifacts_dir: str) -> HeartRiskPipeline:
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)

# ============================================================================
# 2. 侧边栏
# ============================================================================
with st.sidebar:
    st.markdown("## ❤️ 系统信息")
    st.markdown("""
    **模型**: LightGBM (经 Optuna 调优)
    **AUC-ROC**: **0.8376**
    **决策阈值**: 0.486
    **训练数据**: BRFSS 2024 (45 万份问卷)
    **特征数**: 48 个
    """)

    st.markdown("---")
    st.markdown("## 📁 文件路径配置")
    artifacts_dir = st.text_input(
        "预处理产物 (.pkl) 所在目录",
        value=os.environ.get("ARTIFACTS_DIR", "."),
        help="该目录下需包含 model_LightGBM.pkl, scaler.pkl等文件。",
    )

    st.markdown("---")
    st.markdown("## ⚠️ 免责声明")
    st.markdown("""
    <div class="disclaimer">
    本系统是基于人群流行病学数据训练的 <b>风险预测辅助工具</b>，
    <b>不构成任何形式的临床诊断、医疗建议或治疗依据</b>。<br><br>
    评估结果仅反映模型基于您所填问卷给出的统计学风险概率。<br><br>
    若您出现胸痛、气促、持续乏力等症状，请立即就医。
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 3. 主区域逻辑
# ============================================================================
st.markdown('<div class="main-title">❤️ 心脏病 (CHD/MI) 风险评估系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于 2024 BRFSS 数据 · 支持缺失值智能推断</div>', unsafe_allow_html=True)

if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None

# ----------------------------------------------------------------------------
# 3.A 问卷页 
# ----------------------------------------------------------------------------
def render_form() -> None:
    st.markdown("请按实际情况完成问卷。标注 <span class='mandatory-star'>*</span> 的题目为必填；其余题目若不确定可保持默认选项，系统将自动进行推断。", unsafe_allow_html=True)

    section_names = list(QUESTIONS.keys())
    tabs = st.tabs(section_names)
    answers: dict = dict(st.session_state.answers)

    for tab, sec_name in zip(tabs, section_names):
        with tab:
            for q in QUESTIONS[sec_name]:
                key = q["key"]
                labels = [opt[0] for opt in q["options"]]
                values = [opt[1] for opt in q["options"]]

                # 必填项标识
                is_mandatory = key in MANDATORY_KEYS
                display_label = q["label"]
                if is_mandatory:
                    display_label = f"* {display_label}"

                # 索引逻辑：必填项默认第一个，可选项默认最后一个(None)
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
        pipeline = load_pipeline(artifacts_dir)
        with st.spinner("⏳ 模型正在评估中..."):
            st.session_state.assessment = pipeline.assess(answers, top_k=8)
            st.session_state.stage = "result"
            st.rerun()

# ----------------------------------------------------------------------------
# 3.B 结果页 
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
            <div class="risk-card" style="background:{res.risk_color};">
                <p class="risk-prob">您的预测患病概率</p>
                <p class="risk-level">{res.probability * 100:.1f}%</p>
                <p class="risk-prob">风险等级: <b>{res.risk_level}</b></p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("#### 概率刻度尺")
        st.progress(min(1.0, max(0.0, res.probability)))
        st.caption(f"📍 阈值 = {OPTIMAL_THRESHOLD} | 低风险 ≤ {LOW_RISK_QUANTILE} | 高风险 ≥ {HIGH_RISK_QUANTILE}")
        _render_risk_legend(res.probability)

    st.markdown("---")
    st.markdown("### 🎯 影响您此次评估的关键因素")
    _render_contributors(res.top_contributors)

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
        # 重置按钮保留底层清除逻辑，确保能回到初始状态
        for key in list(st.session_state.keys()):
            if key.startswith("q_"):
                del st.session_state[key]
        st.session_state.assessment = None
        st.session_state.stage = "form"
        st.rerun()

# ============================================================================
# 4. 辅助渲染函数 
# ============================================================================
def _render_risk_legend(prob: float) -> None:
    bands = [("低风险", 0.0, LOW_RISK_QUANTILE, '#2ca02c'), ("中风险", LOW_RISK_QUANTILE, OPTIMAL_THRESHOLD, '#f1c40f'), ("中-高风险", OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, '#ff7f0e'), ("高风险", HIGH_RISK_QUANTILE, 1.0, '#d62728')]
    cols = st.columns(len(bands))
    for col, (name, lo, hi, color) in zip(cols, bands):
        active = lo <= prob < hi or (hi == 1.0 and prob >= lo)
        border = '3px solid #2c3e50' if active else '1px solid #ddd'
        col.markdown(f"<div style='border:{border}; border-radius:6px; padding:6px; text-align:center; background:{color}30;'><div style='color:{color}; font-weight:700;'>{name}</div><div style='font-size:0.8rem;'>{lo:.3f} – {hi:.3f}</div></div>", unsafe_allow_html=True)

def _render_contributors(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("当前模型不支持 SHAP 解释。")
        return
    max_abs = float(df['shap_value'].abs().max() or 1e-9)
    for _, row in df.iterrows():
        shap_val = float(row['shap_value'])
        pct = abs(shap_val) / max_abs * 100
        color, side, arrow = ('#d62728', 'left', '↑') if shap_val > 0 else ('#2ca02c', 'right', '↓')
        bar_html = f"<div style='background:#f0f0f0; border-radius:4px; height:18px; position:relative; margin-top:4px;'><div style='background:{color}; width:{pct:.1f}%; height:100%; border-radius:4px; float:{side};'></div></div>"
        c1, c2, c3 = st.columns([3, 5, 2])
        c1.markdown(f"**{row['display_name']}**")
        c2.markdown(bar_html, unsafe_allow_html=True)
        c3.markdown(f"<span style='color:{color}; font-weight:600;'>{arrow} {abs(shap_val):.3f}</span>", unsafe_allow_html=True)

def _render_recommendations(res: RiskAssessment) -> None:
    answers = res.raw_input
    tips = []
    
    if answers.get('_SMOKER3') in (1, 2): tips.append("🚭 **戒烟**:吸烟是冠心病最可干预的强风险因素。")
    if answers.get('_RFBING6') == 2: tips.append("🍷 **限酒**:避免短时间内大量饮酒。")
    if answers.get('EXERANY2') == 2: tips.append("🏃 **规律运动**:每周累计 ≥150 分钟中等强度有氧运动。")
    if answers.get('_BMI5CAT') in (3, 4): tips.append("⚖️ **控制体重**:建议结合饮食与运动减重 5-10%。")
    if answers.get('DIABETE4') == 1: tips.append("🩸 **管好血糖**:糖尿病显著加速冠脉硬化。目标通常 <7%。")
    if answers.get('CVDSTRK3') == 1: tips.append("⚠️ **二级预防**:既往中风者属心血管病高危人群。")
    if answers.get('_RFHLTH') == 2: tips.append("🌿 **改善健康**:建议做一次全面体检。")
    if answers.get('_PHYS14D') == 3 or answers.get('_MENT14D') == 3: tips.append("🧠 **关注身心**:关注长期身体或情绪不佳。")
    if answers.get('PNEUVAC4') == 2 and (answers.get('_AGE_G') or 0) >= 5: tips.append("💉 **肺炎疫苗**:65+ 老人接种可降低心血管事件风险。")
    tips.append("🥗 **健康饮食**:多蔬菜水果、全谷物，少加工肉。")

    if res.risk_level in ("高风险", "中-高风险"): st.warning("⚠️ 强烈建议尽快前往心内科检查。")
    elif res.risk_level == "中风险": st.info("ℹ️ 建议每年做一次心血管健康体检。")
    else: st.success("✅ 继续保持健康生活方式。")
    for tip in tips: st.markdown(f"- {tip}")

# ============================================================================
# 5. 主路由
# ============================================================================
if st.session_state.stage == "form":
    render_form()
else:
    render_result()
