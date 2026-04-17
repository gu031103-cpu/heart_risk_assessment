"""
=============================================================================
心脏病(CHD/MI)风险评估系统 - Streamlit 主应用
=============================================================================
启动方式：
    streamlit run app.py
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
from questionnaire import QUESTIONS, get_default_answers


# ============================================================================
# 0. 页面基础配置
# ============================================================================
st.set_page_config(
    page_title="心脏病风险评估系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入自定义 CSS 提升观感
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b; margin-bottom: 0.2rem; }
    .subtitle   { color: #666; font-size: 1.1rem; margin-bottom: 1.5rem; letter-spacing: 0.5px; }
    .risk-card  { padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .risk-level { font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; letter-spacing: 2px; }
    .risk-prob  { font-size: 1.1rem; opacity: 0.95; font-weight: 500; }
    .disclaimer { background: #fdfaf6; padding: 1.2rem; border-left: 4px solid #d35400; border-radius: 6px; color: #5e4a3d; font-size: 0.9rem; line-height: 1.5; }
    .stProgress > div > div > div > div { background-color: #e74c3c; }
    .imputed-tag { font-size: 0.75rem; color: #e67e22; background: #fdf2e9; padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 1. 加载推理管道（缓存，避免每次刷新重复读盘）
# ============================================================================
@st.cache_resource(show_spinner="🔧 正在加载推演引擎与多重插补管道...")
def load_pipeline() -> HeartRiskPipeline:
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", ".")
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)


# ============================================================================
# 2. 侧边栏：系统信息 + 免责声明
# ============================================================================
with st.sidebar:
    st.markdown("## 🧠 临床级风险推演引擎")
    st.markdown("""
    **核心算法**: LightGBM (Optuna 寻优架构)
    **模型辨识度 (AUC-ROC)**: **0.8376**
    **灵敏度 / 特异性平衡**: 决策阈值 **48.6%**
    **循证基线**: BRFSS 2024 (45万+ 真实世界流行病学队列)
    **特征维度**: 48 维高鉴别力特征子集
    """)

    st.markdown("---")
    st.markdown("## ⚠️ 辅助决策声明")
    st.markdown("""
    <div class="disclaimer">
    本系统输出结果基于人群流行病学大数据归纳，属于<b>预防医学辅助预测工具</b>，
    <b>不可替代临床金标准</b>。<br><br>
    评估结果（概率与风险分层）旨在揭示潜在的心血管风险趋势。实际冠状动脉健康状况须由执业医师结合心电图(ECG)、超声心动图、肌钙蛋白等实验室与影像学指标综合定性。<br><br>
    <i>*如出现急性胸痛、放射性左臂痛或严重气促，请立即呼叫急救。</i>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 3. 主区域
# ============================================================================
st.markdown('<div class="main-title">心脏病 (CHD/MI) 风险推演评估系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于 2024 BRFSS 流行病学大队列 · 机器学习驱动 · 个体化 SHAP 归因释义</div>', unsafe_allow_html=True)

if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None


def render_form() -> None:
    st.markdown("请客观反馈下方体征与生活史信息。后台引擎将动态对齐基线数据并预测风险概率。(本表单数据严格本地/单会话销毁)")

    col_a, col_b, _ = st.columns([1, 1, 6])
    if col_a.button("🎲 填入演示病例 (高危)", help="一键填入一份高危心血管画像，用于功能体验。"):
        st.session_state.answers = _demo_high_risk_profile()
        st.rerun()
    if col_b.button("🧹 清除录入"):
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

                default_idx = 0
                if key in answers:
                    try:
                        default_idx = values.index(answers[key])
                    except ValueError:
                        default_idx = 0

                chosen_label = st.radio(
                    label=q["label"],
                    options=labels,
                    index=default_idx,
                    horizontal=True if len(labels) <= 4 else False,
                    key=f"q_{key}",
                    help=q.get("help"),
                )
                answers[key] = values[labels.index(chosen_label)]

    st.session_state.answers = answers

    st.markdown("---")
    submit_col, _ = st.columns([1, 5])
    if submit_col.button("⚡ 提交系统测算", type="primary", use_container_width=True):
        try:
            pipeline = load_pipeline()
        except Exception as e:
            st.error(f"❌ 推演引擎就绪失败: {type(e).__name__}: {e}")
            return

        with st.spinner("⏳ LightGBM 引擎评估中... 正在进行多重插补与特征映射..."):
            try:
                assessment = pipeline.assess(answers, top_k=8)
            except Exception as e:
                st.error(f"❌ 模型推演抛出异常: {type(e).__name__}: {e}")
                return

        st.session_state.assessment = assessment
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
        st.markdown(
            f"""
            <div class="risk-card" style="background:{res.risk_color};">
                <p class="risk-prob">预测患病概率 (Probability)</p>
                <p class="risk-level">{res.probability * 100:.1f}%</p>
                <p class="risk-prob">评估诊断: <b>{res.risk_level}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 风险定标带 (Risk Scale)")
        st.progress(min(1.0, max(0.0, res.probability)))
        st.caption(
            f"📍 临床决策阈值 = **{OPTIMAL_THRESHOLD * 100:.1f}%**　|　"
            f"低风险截断值 ≤ **{LOW_RISK_QUANTILE * 100:.1f}%**　|　"
            f"高风险截断值 ≥ **{HIGH_RISK_QUANTILE * 100:.1f}%**"
        )
        _render_risk_legend(res.probability)

    st.markdown("---")

    st.markdown("### 🎯 影响您此次评估的关键因素 (SHAP 释义)")
    st.caption("红色图条向右表示该项特征推高了患病预测概率；绿色图条向左表示该特征起到了保护或拉低风险的作用。")
    _render_contributors(res.top_contributors)

    st.markdown("---")

    st.markdown("### 💡 临床建议与生活干预")
    _render_recommendations(res)

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("⬅️ 修改特征并重新测算", use_container_width=True):
        st.session_state.stage = "form"
        st.rerun()
    if c2.button("🆕 清除并开始新评估", use_container_width=True):
        st.session_state.answers = {}
        st.session_state.assessment = None
        st.session_state.stage = "form"
        st.rerun()


def _render_risk_legend(prob: float) -> None:
    bands = [
        ("低风险",   0.0,                 LOW_RISK_QUANTILE,  '#2ca02c'),
        ("中风险",   LOW_RISK_QUANTILE,   OPTIMAL_THRESHOLD,  '#f1c40f'),
        ("中-高风险", OPTIMAL_THRESHOLD,  HIGH_RISK_QUANTILE, '#ff7f0e'),
        ("高风险",   HIGH_RISK_QUANTILE,  1.0,                '#d62728'),
    ]
    cols = st.columns(len(bands))
    for col, (name, lo, hi, color) in zip(cols, bands):
        active = lo <= prob < hi or (hi == 1.0 and prob >= lo)
        weight = '700' if active else '400'
        border = '2px solid #2c3e50' if active else '1px solid #eaeaea'
        bg = f"{color}30" if active else "#fafafa"
        col.markdown(
            f"<div style='border:{border}; border-radius:8px;"
            f"padding:8px; text-align:center; background:{bg};'>"
            f"<div style='color:{color}; font-weight:{weight}; font-size:1.05rem;'>{name}</div>"
            f"<div style='font-size:0.85rem; color:#666; margin-top:2px;'>"
            f"{lo * 100:.1f}% – {hi * 100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )


def _render_contributors(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("当前模型配置不支持可解释性输出。")
        return

    max_abs = float(df['shap_value'].abs().max() or 1e-9)

    html_blocks = ["<div style='margin-top: 1.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>"]

    for _, row in df.iterrows():
        shap_val = float(row['shap_value'])
        # 计算宽度比例，预留两侧显示文本的空间
        pct = (abs(shap_val) / max_abs) * 45 
        
        display_name = row['display_name']
        if row.get('is_imputed', False):
            display_name += " <span class='imputed-tag' title='因表单该项缺失/拒答，由系统基于其余特征推算填补'>填补</span>"

        if shap_val > 0:
            color = "#e87a71" # 截图中的红色
            text_color = "#e87a71"
            bar_html = f"""
            <div style='position: absolute; left: 50%; height: 24px; width: {pct}%; background-color: {color}; top: 50%; transform: translateY(-50%);'></div>
            <div style='position: absolute; left: {50 + pct}%; height: 100%; display: flex; align-items: center; padding-left: 10px; color: {text_color}; font-size: 0.9rem; font-weight: 500;'>+{shap_val:.3f}</div>
            """
        else:
            color = "#64a382" # 截图中的绿色
            text_color = "#64a382"
            bar_html = f"""
            <div style='position: absolute; right: 50%; height: 24px; width: {pct}%; background-color: {color}; top: 50%; transform: translateY(-50%);'></div>
            <div style='position: absolute; right: {50 + pct}%; height: 100%; display: flex; align-items: center; padding-right: 10px; color: {text_color}; font-size: 0.9rem; font-weight: 500;'>{shap_val:.3f}</div>
            """

        row_html = f"""
        <div style='display: flex; align-items: center; margin-bottom: 12px; min-height: 36px;'>
            <div style='width: 35%; text-align: right; padding-right: 15px; font-size: 0.95rem; color: #444; font-weight: 500;'>
                {display_name}
            </div>
            <div style='width: 65%; position: relative; height: 36px;'>
                <div style='position: absolute; left: 50%; top: -6px; bottom: -6px; border-left: 1px dashed #bbb; z-index: 0;'></div>
                {bar_html}
            </div>
        </div>
        """
        html_blocks.append(row_html)

    html_blocks.append("</div>")
    st.markdown("\n".join(html_blocks), unsafe_allow_html=True)


def _render_recommendations(res: RiskAssessment) -> None:
    answers = res.raw_input
    tips: list[str] = []

    if answers.get('_SMOKER3') in (1, 2):
        tips.append("🚭 **首要干预(戒烟)**: 尼古丁与焦油显著损伤血管内皮。建议立即启动戒烟干预，必要时寻求尼古丁替代疗法。")
    if answers.get('_RFBING6') == 2:
        tips.append("🍷 **限制酒精摄入**: 暴饮行为可诱发严重心律失常及心肌负荷陡增。建议严格限制单次酒精摄入量。")
    if answers.get('EXERANY2') == 2:
        tips.append("🏃 **规律心血管适能训练**: 每周累计 ≥150 分钟中等强度有氧运动（如快步走、游泳），改善心肺耐力。")
    if answers.get('_BMI5CAT') in (3, 4):
        tips.append("⚖️ **代谢负荷管理(减重)**: 当前 BMI 指数偏高，建议结合营养处方，目标在未来 6 个月内减轻基线体重的 5-10%。")
    if answers.get('DIABETE4') == 1:
        tips.append("🩸 **强化血糖控制**: 糖尿病是冠脉硬化的强等危症。请定期监测糖化血红蛋白(HbA1c)，控制在 7% 以下。")
    if answers.get('CVDSTRK3') == 1:
        tips.append("⚠️ **心脑血管二级预防**: 既往脑卒中病史表明血管病变广泛，务必严格遵医嘱服用抗血小板或降脂药物。")
    if answers.get('PNEUVAC4') == 2 and (answers.get('_AGE_G') or 0) >= 5:
        tips.append("💉 **肺炎疫苗接种**: 老年人群应接种肺炎球菌疫苗，降低因重症呼吸道感染诱发急性心衰的风险。")

    tips.append("🥗 **全因健康干预**: 建议依从 DASH (抗高血压) 或地中海饮食模式，提高膳食纤维与 Omega-3 摄入。")

    if res.risk_level in ("高风险", "中-高风险"):
        st.warning(
            f"⚠️ **临床预警**: 您的测算概率 ({res.probability * 100:.1f}%) 已越过决策阈值。建议尽快前往三甲医院心内科，完善静息/运动心电图、心脏多普勒超声及血脂全套检查。"
        )
    elif res.risk_level == "中风险":
        st.info(
            f"ℹ️ **风险提示**: 处于临界或中度风险区间 ({res.probability * 100:.1f}%)，虽未达极高危，但建议纳入年度心血管专科体检，并针对上述风险点进行干预。"
        )
    else:
        st.success(
            "✅ **平稳维持**: 当前处于低风险流行病学区间。继续保持良好的生理基线，每 1-2 年进行常规基础体检即可。"
        )

    for tip in tips:
        st.markdown(f"- {tip}")


def _demo_high_risk_profile() -> dict:
    return {
        '_AGE_G': 6, 'SEXVAR': 1, '_IMPRACE': 1, 'MARITAL': 3,
        '_EDUCAG': 2, '_INCOMG1': 2, 'EMPLOY1': 7, '_BMI5CAT': 4,
        'VETERAN3': 1, '_STATE': 6, 'EXERANY2': 2, '_SMOKER3': 1,
        '_CURECI3': 1, '_RFBING6': 1, 'CVDSTRK3': 1, 'DIABETE4': 1,
        'CHCOCNC1': 2, 'CHCCOPD3': 1, 'CHCKDNY2': 1, 'ADDEPEV3': 1,
        '_DRDXAR2': 1, '_LTASTH1': 2, '_RFHLTH': 2, '_PHYS14D': 3,
        '_MENT14D': 2, 'DEAF': 1, 'BLIND': 2, 'DECIDE': 1,
        'DIFFWALK': 1, 'DIFFDRES': 2, 'DIFFALON': 1, '_HLTHPL2': 1,
        'MEDCOST1': 1, 'PNEUVAC4': 2,
    }

if st.session_state.stage == "form":
    render_form()
else:
    render_result()
