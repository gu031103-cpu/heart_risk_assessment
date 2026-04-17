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

sys.path.append(str(Path(__file__).parent))
from risk_pipeline import (HeartRiskPipeline, RiskAssessment, OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, LOW_RISK_QUANTILE)
from questionnaire import QUESTIONS, get_default_answers

# 页面基础配置及 CSS (保持不变)
st.set_page_config(page_title="心脏病风险评估系统", page_icon="❤️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b; margin-bottom: 0.2rem; }
    .subtitle   { color: #666; margin-bottom: 1.5rem; }
    .risk-card  { padding: 1.2rem 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .risk-level { font-size: 2rem; font-weight: 700; margin: 0; }
    .risk-prob  { font-size: 1.1rem; opacity: 0.95; }
    .disclaimer { background: #fff5e6; padding: 1rem 1.2rem; border-left: 4px solid #f39c12; border-radius: 6px; color: #6b4f00; font-size: 0.92rem; }
    .stProgress > div > div > div > div { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="🔧 正在加载模型与预处理管道...")
def load_pipeline(artifacts_dir: str) -> HeartRiskPipeline:
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)

with st.sidebar:
    st.markdown("## ❤️ 系统信息")
    st.markdown("**模型**: LightGBM\n\n**AUC-ROC**: 0.8376\n\n**决策阈值**: 0.486")
    st.markdown("---")
    artifacts_dir = st.text_input("预处理产物所在目录", value=os.environ.get("ARTIFACTS_DIR", "."))
    st.markdown("---")
    st.markdown("## ⚠️ 免责声明")
    st.markdown('<div class="disclaimer">本系统是基于人群流行病学数据训练的<b>辅助工具</b>，不构成任何形式的临床诊断。若您有心脏不适，请立即就医。</div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">❤️ 心脏病 (CHD/MI) 风险评估系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">支持缺失值智能推断插补 · LightGBM · SHAP 个体化解释</div>', unsafe_allow_html=True)

if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None

def render_form() -> None:
    st.markdown("请按实际情况完成下面的问卷。**如果遇到不确定或涉及隐私的问题，您可以保持默认的「跳过/不确定」选项，系统会自动根据人群画像特征为您插补缺失数据**。")

    col_a, col_b, _ = st.columns([1, 1, 6])
    if col_a.button("🎲 填入示例答案"):
        st.session_state.answers = _demo_high_risk_profile()
        st.rerun()
    if col_b.button("🧹 清空表单"):
        st.session_state.answers = {}
        st.rerun()

    section_names = list(QUESTIONS.keys())
    tabs = st.tabs(section_names)
    answers: dict = dict(st.session_state.answers)

    for tab, sec_name in zip(tabs, section_names):
        with tab:
            for q in QUESTIONS[sec_name]:
                key = q["key"]
                labels = [opt[0] for opt in q["options"]]
                values = [opt[1] for opt in q["options"]]

                # 默认值逻辑重构：未作答时，选中数组最后一个选项 ("不确定/拒绝回答")
                default_idx = len(values) - 1
                if key in answers:
                    try:
                        default_idx = values.index(answers[key])
                    except ValueError:
                        default_idx = len(values) - 1

                chosen_label = st.radio(
                    label=q["label"],
                    options=labels,
                    index=default_idx,
                    horizontal=True if len(labels) <= 4 else False,
                    key=f"q_{key}",
                    help=q.get("help"),
                )
                # 获取用户选定的 value (可能是 None)
                answers[key] = values[labels.index(chosen_label)]

    st.session_state.answers = answers
    st.markdown("---")
    if st.button("🔍 开始评估风险", type="primary", use_container_width=True):
        pipeline = load_pipeline(artifacts_dir)
        with st.spinner("⏳ 模型正在评估中... (若有缺失值，后台将自动激活插补引擎)"):
            st.session_state.assessment = pipeline.assess(answers, top_k=8)
            st.session_state.stage = "result"
            st.rerun()

def render_result() -> None:
    res: RiskAssessment = st.session_state.assessment
    if res is None:
        st.session_state.stage = "form"
        st.rerun()
        return

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="risk-card" style="background:{res.risk_color};"><p class="risk-prob">您的预测概率</p><p class="risk-level">{res.probability * 100:.1f}%</p><p class="risk-prob">风险等级: <b>{res.risk_level}</b></p></div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### 概率刻度尺")
        st.progress(min(1.0, max(0.0, res.probability)))
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
        st.session_state.assessment = None
        st.session_state.stage = "form"
        st.rerun()

def _render_risk_legend(prob: float) -> None:
    bands = [("低风险", 0.0, LOW_RISK_QUANTILE, '#2ca02c'), ("中风险", LOW_RISK_QUANTILE, OPTIMAL_THRESHOLD, '#f1c40f'), ("中-高风险", OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, '#ff7f0e'), ("高风险", HIGH_RISK_QUANTILE, 1.0, '#d62728')]
    cols = st.columns(len(bands))
    for col, (name, lo, hi, color) in zip(cols, bands):
        active = lo <= prob < hi or (hi == 1.0 and prob >= lo)
        border = '3px solid #2c3e50' if active else '1px solid #ddd'
        col.markdown(f"<div style='border:{border}; border-radius:6px; padding:6px; text-align:center; background:{color}30;'><div style='color:{color}; font-weight:700;'>{name}</div><div style='font-size:0.8rem;'>{lo:.3f} – {hi:.3f}</div></div>", unsafe_allow_html=True)

def _render_contributors(df: pd.DataFrame) -> None:
    max_abs = float(df['shap_value'].abs().max() or 1e-9)
    for _, row in df.iterrows():
        shap_val = float(row['shap_value'])
        pct, color, side, arrow = abs(shap_val) / max_abs * 100, '#d62728' if shap_val > 0 else '#2ca02c', 'left' if shap_val > 0 else 'right', '↑' if shap_val > 0 else '↓'
        c1, c2, c3 = st.columns([3, 5, 2])
        c1.markdown(f"**{row['display_name']}**")
        c2.markdown(f"<div style='background:#f0f0f0; border-radius:4px; height:18px; margin-top:4px;'><div style='background:{color}; width:{pct:.1f}%; height:100%; border-radius:4px; float:{side};'></div></div>", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:{color}; font-weight:600;'>{arrow} {abs(shap_val):.3f}</span>", unsafe_allow_html=True)

def _render_recommendations(res: RiskAssessment) -> None:
    answers = res.raw_input
    tips: list[str] = []

    # 逻辑修复：兼容 None 值比较
    if answers.get('_SMOKER3') in (1, 2): tips.append("🚭 **戒烟**:吸烟是冠心病最可干预的强风险因素。")
    if answers.get('_RFBING6') == 2: tips.append("🍷 **限酒**:避免短时间内大量饮酒。")
    if answers.get('EXERANY2') == 2: tips.append("🏃 **规律运动**:每周累计 ≥150 分钟中等强度有氧运动。")
    if answers.get('_BMI5CAT') in (3, 4): tips.append("⚖️ **控制体重**:建议结合饮食与运动减重 5-10%。")
    if answers.get('DIABETE4') == 1: tips.append("🩸 **管好血糖**:定期监测 HbA1c,目标通常 <7%。")
    if answers.get('CVDSTRK3') == 1: tips.append("⚠️ **二级预防**:既往中风者属心血管病高危人群,务必规律服药并复查。")
    if answers.get('_RFHLTH') == 2: tips.append("🌿 **改善整体健康**:自评健康欠佳常预示多重风险叠加,建议做一次全面体检。")
    if answers.get('_PHYS14D') == 3 or answers.get('_MENT14D') == 3: tips.append("🧠 **关注身心**:每月 ≥14 天身体或情绪不佳,建议同时评估慢性病与心理健康。")
    
    age_g = answers.get('_AGE_G')
    if answers.get('PNEUVAC4') == 2 and age_g is not None and age_g >= 5:
        tips.append("💉 **接种肺炎疫苗**:65+ 老年人接种可降低呼吸道感染诱发心血管事件的风险。")
    if answers.get('MEDCOST1') == 1 or answers.get('_HLTHPL2') == 2: tips.append("🏥 **保障医疗可及性**:解决医保障碍,确保慢病随访不中断。")

    tips.append("🥗 **DASH/地中海饮食**:多蔬菜水果、全谷物,少加工肉与含糖饮料。")

    if res.risk_level in ("高风险", "中-高风险"): st.warning("⚠️ 强烈建议尽快前往心内科做一次系统性检查。")
    elif res.risk_level == "中风险": st.info("ℹ️ 建议每年做一次心血管健康体检。")
    else: st.success("✅ 当前风险较低，继续保持健康生活方式。")

    for tip in tips: st.markdown(f"- {tip}")

def _demo_high_risk_profile() -> dict:
    return {'_AGE_G': 6, 'SEXVAR': 1, '_IMPRACE': 1, 'MARITAL': 3, '_EDUCAG': 2, '_INCOMG1': 2, 'EMPLOY1': 7, '_BMI5CAT': 4, 'VETERAN3': 1, '_STATE': 6, 'EXERANY2': 2, '_SMOKER3': 1, '_CURECI3': 1, '_RFBING6': 1, 'CVDSTRK3': 1, 'DIABETE4': 1, 'CHCOCNC1': 2, 'CHCCOPD3': 1, 'CHCKDNY2': 1, 'ADDEPEV3': 1, '_DRDXAR2': 1, '_LTASTH1': 2, '_RFHLTH': 2, '_PHYS14D': 3, '_MENT14D': 2, 'DEAF': 1, 'BLIND': 2, 'DECIDE': 1, 'DIFFWALK': 1, 'DIFFDRES': 2, 'DIFFALON': 1, '_HLTHPL2': 1, 'MEDCOST1': 1, 'PNEUVAC4': 2}

if st.session_state.stage == "form":
    render_form()
else:
    render_result()
