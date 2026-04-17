"""
=============================================================================
心脏病(CHD/MI)风险评估系统 - 主程序
=============================================================================
功能优化：
1. 区分必填项与可选项。必填项（无缺失值指标）移除 "不确定" 选项。
2. 强化 UI 引导，说明系统如何利用后台模型进行缺失值推断。
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# 导入配置
sys.path.append(str(Path(__file__).parent))
from risk_pipeline import (HeartRiskPipeline, RiskAssessment, OPTIMAL_THRESHOLD, 
                           HIGH_RISK_QUANTILE, LOW_RISK_QUANTILE)
from questionnaire import QUESTIONS, MANDATORY_KEYS

# ----------------------------------------------------------------------------
# 1. 页面样式与基础配置
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="心脏病风险评估系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b; margin-bottom: 0.2rem; }
    .subtitle   { color: #666; margin-bottom: 1.5rem; }
    .risk-card  { padding: 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .risk-level { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .mandatory-label { color: #e74c3c; font-weight: bold; margin-right: 5px; }
    .disclaimer { background: #fff5e6; padding: 1rem; border-left: 5px solid #f39c12; border-radius: 5px; color: #6b4f00; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="🔧 正在初始化 AI 推理引擎...")
def load_pipeline(artifacts_dir: str) -> HeartRiskPipeline:
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)

# ----------------------------------------------------------------------------
# 2. 状态管理
# ----------------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None

# ----------------------------------------------------------------------------
# 3. 问卷页面渲染
# ----------------------------------------------------------------------------
def render_form():
    st.markdown('<div class="main-title">❤️ 心脏病 (CHD/MI) 风险评估系统</div>', unsafe_allow_html=True)
    st.markdown("本系统利用 2024 BRFSS 大规模人群数据训练。标注 <span class='mandatory-label'>*</span> 的题目为必选项，其余题目若不确定可保持默认。系统会自动为您推断缺失的信息。", unsafe_allow_html=True)
    
    # 顶部操作栏
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("🎲 载入示例数据", help="填入一份典型的高风险案例"):
        st.session_state.answers = _get_demo_data()
        st.rerun()
    if c2.button("🧹 清空所有选择"):
        st.session_state.answers = {}
        st.rerun()

    # 问卷板块渲染
    tabs = st.tabs(list(QUESTIONS.keys()))
    current_answers = dict(st.session_state.answers)

    for tab, (sec_name, questions) in zip(tabs, QUESTIONS.items()):
        with tab:
            for q in questions:
                key = q["key"]
                labels = [opt[0] for opt in q["options"]]
                values = [opt[1] for opt in q["options"]]
                
                is_mandatory = key in MANDATORY_KEYS
                display_label = f"{'* ' if is_mandatory else ''}{q['label']}"
                
                # 确定默认索引
                if key in current_answers:
                    try:
                        default_idx = values.index(current_answers[key])
                    except ValueError:
                        default_idx = 0 if is_mandatory else len(values) - 1
                else:
                    default_idx = 0 if is_mandatory else len(values) - 1

                chosen_label = st.radio(
                    label=display_label,
                    options=labels,
                    index=default_idx,
                    horizontal=True if len(labels) <= 4 else False,
                    key=f"radio_{key}"
                )
                current_answers[key] = values[labels.index(chosen_label)]

    st.session_state.answers = current_answers

    st.markdown("---")
    if st.button("🔍 开始多维风险评估", type="primary", use_container_width=True):
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", ".")
        pipeline = load_pipeline(artifacts_dir)
        with st.spinner("⏳ 后台模型正在进行多重插补与风险测算..."):
            st.session_state.assessment = pipeline.assess(current_answers)
            st.session_state.stage = "result"
            st.rerun()

# ----------------------------------------------------------------------------
# 4. 结果展示页面渲染
# ----------------------------------------------------------------------------
def render_result():
    res: RiskAssessment = st.session_state.assessment
    st.markdown(f"### 评估结果报告")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="risk-card" style="background:{res.risk_color};">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">预测患病概率</p>
            <p class="risk-level">{res.probability * 100:.1f}%</p>
            <p>风险等级：<strong>{res.risk_level}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 风险分布参考")
        st.progress(min(1.0, max(0.0, res.probability)))
        st.caption(f"决策阈值: {OPTIMAL_THRESHOLD} | 低风险线: {LOW_RISK_QUANTILE} | 高风险线: {HIGH_RISK_QUANTILE}")
        _draw_legend(res.probability)

    st.markdown("---")
    st.markdown("#### 🎯 个体化影响因子 (SHAP 贡献度分析)")
    _render_shap_bars(res.top_contributors)

    st.markdown("---")
    st.markdown("#### 💡 健康管理建议")
    _render_tips(res)

    st.markdown("---")
    if st.button("⬅️ 返回修改问卷", use_container_width=True):
        st.session_state.stage = "form"
        st.rerun()

# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------
def _draw_legend(prob):
    bands = [("低风险", 0, LOW_RISK_QUANTILE, '#2ca02c'), ("中风险", LOW_RISK_QUANTILE, OPTIMAL_THRESHOLD, '#f1c40f'), ("中高危", OPTIMAL_THRESHOLD, HIGH_RISK_QUANTILE, '#ff7f0e'), ("极高危", HIGH_RISK_QUANTILE, 1, '#d62728')]
    cols = st.columns(len(bands))
    for col, (name, lo, hi, color) in zip(cols, bands):
        is_active = lo <= prob < hi or (hi == 1 and prob >= lo)
        border = "3px solid #333" if is_active else "1px solid #eee"
        col.markdown(f"<div style='border:{border}; text-align:center; padding:5px; background:{color}22;'>{name}</div>", unsafe_allow_html=True)

def _render_shap_bars(df):
    if df is None or df.empty:
        st.info("数据完整度极高，所有因子贡献均衡。")
        return
    for _, row in df.iterrows():
        val = row['shap_value']
        color = "#e74c3c" if val > 0 else "#2ecc71"
        c1, c2, c3 = st.columns([3, 5, 2])
        c1.write(row['display_name'])
        c2.progress(min(1.0, abs(val) * 2)) # 示意性展示
        c3.markdown(f"<span style='color:{color}'>{'↑' if val > 0 else '↓'} {abs(val):.3f}</span>", unsafe_allow_html=True)

def _render_tips(res):
    ans = res.raw_input
    if res.risk_level in ["高风险", "中-高风险"]:
        st.warning("⚠️ 检测到高危信号。您的多项生理指标或病史与心脏病高度相关，建议尽快预约心内科医生进行专业检查。")
    
    tips = []
    if ans.get('_SMOKER3') in [1, 2]: tips.append("🚭 **戒烟计划**：吸烟是冠心病最强的单一可控因素。")
    if ans.get('EXERANY2') == 2: tips.append("🏃 **加强锻炼**：即使是每天 20 分钟的快走也能显著降低风险。")
    if ans.get('_BMI5CAT') == 4: tips.append("⚖️ **体重管理**：肥胖会增加心脏泵血压力，建议通过饮食控制 BMI。")
    
    if not tips:
        st.success("✨ 继续保持当前健康的生活方式！定期体检是关键。")
    else:
        for t in tips: st.markdown(f"- {t}")

def _get_demo_data():
    return {'_AGE_G': 6, 'SEXVAR': 1, '_IMPRACE': 1, '_STATE': 6, 'DIABETE4': 1, '_SMOKER3': 1, 'CVDSTRK3': 1}

# 运行
if st.session_state.stage == "form":
    render_form()
else:
    render_result()
