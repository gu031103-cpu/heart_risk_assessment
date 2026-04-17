"""
=============================================================================
心脏病(CHD/MI)风险评估系统 - Streamlit 主应用
=============================================================================
启动方式：
    streamlit run app.py

功能：
  1. 问卷式交互界面，分 5 个板块采集 35 项可自我报告指标；
  2. 对接训练好的 LightGBM 最优模型 (AUC=0.8376, 阈值=0.486)；
  3. 输出风险概率、4 级风险等级、个体化 SHAP 关键贡献因子；
  4. 给出基于贡献因子的可执行健康建议；
  5. 强调辅助参考定位，附醒目免责声明。
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

# 注入少量自定义 CSS 提升观感 (新增了 imputed-tag 的样式)
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b;
                  margin-bottom: 0.2rem; }
    .subtitle   { color: #666; margin-bottom: 1.5rem; }
    .risk-card  { padding: 1.2rem 1.5rem; border-radius: 12px;
                  color: white; text-align: center; }
    .risk-level { font-size: 2rem; font-weight: 700; margin: 0; }
    .risk-prob  { font-size: 1.1rem; opacity: 0.95; }
    .factor-row { padding: 0.4rem 0; border-bottom: 1px solid #eee; }
    .disclaimer { background: #fff5e6; padding: 1rem 1.2rem;
                  border-left: 4px solid #f39c12; border-radius: 6px;
                  color: #6b4f00; font-size: 0.92rem; }
    .stProgress > div > div > div > div { background-color: #e74c3c; }
    .imputed-tag { font-size: 0.75rem; color: #e67e22; background: #fdf2e9; padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 1. 加载推理管道（缓存，避免每次刷新重复读盘）
# ============================================================================
@st.cache_resource(show_spinner="🔧 正在加载模型与预处理管道...")
def load_pipeline() -> HeartRiskPipeline:
    # 移除前端路径配置，改为默认从环境变量或当前目录隐式获取
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", ".")
    return HeartRiskPipeline(artifacts_dir=artifacts_dir)


# ============================================================================
# 2. 侧边栏：系统信息 + 免责声明
# ============================================================================
with st.sidebar:
    # --- 修改点：系统信息更加高级专业化 ---
    st.markdown("## 🧠 临床级风险推演引擎")
    st.markdown("""
    **核心算法**: LightGBM (Optuna 寻优架构)
    **模型辨识度 (AUC-ROC)**: **0.8376**
    **灵敏度/特异性平衡阈值**: **48.6%**
    **循证基线**: BRFSS 2024 (45万+ 真实世界流行病学队列)
    **特征维度**: 48 维高鉴别力特征子集
    """)

    st.markdown("---")
    
    # --- 修改点：删除了文件路径配置部分 ---

    st.markdown("## ⚠️ 免责声明")
    st.markdown("""
    <div class="disclaimer">
    本系统是基于人群流行病学数据训练的 <b>风险预测辅助工具</b>，
    <b>不构成任何形式的临床诊断、医疗建议或治疗依据</b>。<br><br>
    评估结果仅反映模型基于您所填问卷给出的统计学风险概率，
    实际心血管健康状况须由<b>具有执业资质的医生</b>结合临床检查、
    实验室指标和影像学综合判断。<br><br>
    若您出现胸痛、气促、持续乏力等症状，请立即就医。
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 3. 主区域：标题 + 状态切换 (问卷页 ↔ 结果页)
# ============================================================================
st.markdown('<div class="main-title">❤️ 心脏病 (CHD/MI) 风险评估系统</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于 2024 BRFSS 数据 · LightGBM · '
            'SHAP 个体化解释</div>',
            unsafe_allow_html=True)

# 用 session_state 持久化页面状态
if "stage" not in st.session_state:
    st.session_state.stage = "form"          # form | result
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "assessment" not in st.session_state:
    st.session_state.assessment = None


# ----------------------------------------------------------------------------
# 3.A 问卷页
# ----------------------------------------------------------------------------
def render_form() -> None:
    st.markdown(
        "请按实际情况完成下面的简短问卷,所有信息仅在本会话内使用,**不会被保存或上传**。"
    )

    # 顶部实用按钮
    col_a, col_b, _ = st.columns([1, 1, 6])
    if col_a.button("🎲 填入示例答案", help="一键填入一份高风险样例,便于体验系统。"):
        st.session_state.answers = _demo_high_risk_profile()
        st.rerun()
    if col_b.button("🧹 清空答案"):
        st.session_state.answers = {}
        st.rerun()

    # 用 tabs 展示 5 个板块
    section_names = list(QUESTIONS.keys())
    tabs = st.tabs(section_names)

    answers: dict = dict(st.session_state.answers)

    for tab, sec_name in zip(tabs, section_names):
        with tab:
            for q in QUESTIONS[sec_name]:
                key = q["key"]
                labels = [opt[0] for opt in q["options"]]
                values = [opt[1] for opt in q["options"]]

                # 选取已存答案对应索引；没有则默认第一个
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
    if submit_col.button("🔍 开始评估风险", type="primary", use_container_width=True):
        try:
            pipeline = load_pipeline()
        except FileNotFoundError as e:
            st.error(f"❌ 加载模型失败:\n\n{e}")
            return
        except Exception as e:
            st.error(f"❌ 推理管道初始化异常: {type(e).__name__}: {e}")
            return

        with st.spinner("⏳ 模型正在评估中..."):
            try:
                assessment = pipeline.assess(answers, top_k=8)
            except Exception as e:
                st.error(f"❌ 评估过程出错: {type(e).__name__}: {e}")
                return

        st.session_state.assessment = assessment
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

    # ① 顶部：风险卡片 + 概率条
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""
            <div class="risk-card" style="background:{res.risk_color};">
                <p class="risk-prob">您的预测患病概率</p>
                <p class="risk-level">{res.probability * 100:.1f}%</p>
                <p class="risk-prob">风险等级: <b>{res.risk_level}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 概率刻度尺")
        st.progress(min(1.0, max(0.0, res.probability)))
        # --- 修改点：阈值门槛全部改为百分比 ---
        st.caption(
            f"📍 临床决策阈值 = **{OPTIMAL_THRESHOLD * 100:.1f}%**　|　"
            f"低风险分位 ≤ **{LOW_RISK_QUANTILE * 100:.1f}%**　|　"
            f"高风险分位 ≥ **{HIGH_RISK_QUANTILE * 100:.1f}%**"
        )
        _render_risk_legend(res.probability)

    st.markdown("---")

    # ② 关键贡献因子
    st.markdown("### 🎯 影响您此次评估的关键因素")
    st.caption("以下 SHAP 值代表该因素相对于平均水平把您的患病概率推高(↑)或拉低(↓)的程度。")
    _render_contributors(res.top_contributors)

    st.markdown("---")

    # ③ 个性化建议
    st.markdown("### 💡 基于评估结果的健康建议")
    _render_recommendations(res)

    st.markdown("---")

    # ④ 操作按钮
    c1, c2 = st.columns(2)
    if c1.button("⬅️ 返回修改问卷", use_container_width=True):
        st.session_state.stage = "form"
        st.rerun()
    if c2.button("🆕 重新开始评估", use_container_width=True):
        st.session_state.answers = {}
        st.session_state.assessment = None
        st.session_state.stage = "form"
        st.rerun()


# ----------------------------------------------------------------------------
# 4. 辅助渲染函数
# ----------------------------------------------------------------------------
def _render_risk_legend(prob: float) -> None:
    """概率刻度尺下方的等级图例。"""
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
        border = '3px solid #2c3e50' if active else '1px solid #ddd'
        # --- 修改点：图例门槛也全部改为百分比 ---
        col.markdown(
            f"<div style='border:{border}; border-radius:6px;"
            f"padding:6px; text-align:center; background:{color}30;'>"
            f"<div style='color:{color}; font-weight:{weight};'>{name}</div>"
            f"<div style='font-size:0.8rem; color:#555;'>"
            f"{lo * 100:.1f}% – {hi * 100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )


def _render_contributors(df: pd.DataFrame) -> None:
    """渲染 SHAP top-K 发散型条形图 (还原截图样式)。"""
    if df is None or df.empty:
        st.info("当前模型不支持 SHAP 解释,已跳过此模块。")
        return

    max_abs = float(df['shap_value'].abs().max() or 1e-9)
    
    # 纯 HTML/CSS 绘制的发散型图表容器
    html_blocks = ["<div style='margin-top: 1.5rem; font-family: sans-serif;'>"]

    for _, row in df.iterrows():
        shap_val = float(row['shap_value'])
        # 计算宽度比例，基准最大宽度占一半空间的 90% (留白给文字)
        pct = (abs(shap_val) / max_abs) * 45 
        
        display_name = row['display_name']
        # 处理插补标签
        if row.get('is_imputed', False):
            display_name += " <span class='imputed-tag' title='因您选择了不确定或未提供，该项由系统基于其他特征自动填补预测'>[填补]</span>"

        if shap_val > 0:
            color = "#e87a71" # 截图偏红颜色
            text_color = "#e87a71"
            bar_html = f"""
            <div style='position: absolute; left: 50%; height: 26px; width: {pct}%; background-color: {color}; top: 50%; transform: translateY(-50%);'></div>
            <div style='position: absolute; left: {50 + pct}%; height: 100%; display: flex; align-items: center; padding-left: 8px; color: {text_color}; font-size: 0.85rem;'>+{shap_val:.3f}</div>
            """
        else:
            color = "#64a382" # 截图偏绿颜色
            text_color = "#64a382"
            bar_html = f"""
            <div style='position: absolute; right: 50%; height: 26px; width: {pct}%; background-color: {color}; top: 50%; transform: translateY(-50%);'></div>
            <div style='position: absolute; right: {50 + pct}%; height: 100%; display: flex; align-items: center; padding-right: 8px; color: {text_color}; font-size: 0.85rem;'>{shap_val:.3f}</div>
            """

        row_html = f"""
        <div style='display: flex; align-items: center; margin-bottom: 8px; min-height: 36px;'>
            <div style='width: 35%; text-align: right; padding-right: 15px; font-size: 0.95rem; color: #333;'>
                {display_name}
            </div>
            <div style='width: 65%; position: relative; height: 36px;'>
                <div style='position: absolute; left: 50%; top: -4px; bottom: -4px; border-left: 1px dashed #ccc; z-index: 0;'></div>
                {bar_html}
            </div>
        </div>
        """
        html_blocks.append(row_html)

    html_blocks.append("</div>")
    st.markdown("\n".join(html_blocks), unsafe_allow_html=True)


def _render_recommendations(res: RiskAssessment) -> None:
    """根据 top 贡献因子+用户回答,生成可执行建议清单。"""
    answers = res.raw_input
    tips: list[str] = []

    # 按特征出现优先级生成建议
    if answers.get('_SMOKER3') in (1, 2):
        tips.append("🚭 **戒烟**:吸烟是冠心病最可干预的强风险因素。可咨询戒烟门诊或拨打戒烟热线 12320。")
    if answers.get('_RFBING6') == 2:
        tips.append("🍷 **限酒**:避免短时间内大量饮酒。男性单日 ≤2 杯,女性 ≤1 杯标准酒精饮品。")
    if answers.get('EXERANY2') == 2:
        tips.append("🏃 **规律运动**:每周累计 ≥150 分钟中等强度有氧运动(如快走、骑车),分 5 天进行。")
    if answers.get('_BMI5CAT') in (3, 4):
        tips.append("⚖️ **控制体重**:超重/肥胖增加心脏负担,建议结合饮食与运动减重 5-10%。")
    if answers.get('DIABETE4') == 1:
        tips.append("🩸 **管好血糖**:糖尿病显著加速冠脉硬化。定期监测 HbA1c,目标通常 <7%。")
    if answers.get('CVDSTRK3') == 1:
        tips.append("⚠️ **二级预防**:既往中风者属心血管病高危人群,务必规律服药并复查。")
    if answers.get('_RFHLTH') == 2:
        tips.append("🌿 **改善整体健康**:自评健康欠佳常预示多重风险叠加,建议做一次全面体检。")
    if answers.get('_PHYS14D') == 3 or answers.get('_MENT14D') == 3:
        tips.append("🧠 **关注身心**:每月 ≥14 天身体或情绪不佳,建议同时评估慢性病与心理健康。")
    if answers.get('PNEUVAC4') == 2 and (answers.get('_AGE_G') or 0) >= 5:
        tips.append("💉 **接种肺炎疫苗**:65+ 老年人接种可降低呼吸道感染诱发心血管事件的风险。")
    if answers.get('MEDCOST1') == 1 or answers.get('_HLTHPL2') == 2:
        tips.append("🏥 **保障医疗可及性**:解决医保/费用障碍,确保慢病随访不中断。")

    # 通用建议(无论评估结果)
    tips.append("🥗 **DASH/地中海饮食**:多蔬菜水果、全谷物、坚果鱼类,少加工肉与含糖饮料。")

    # 按风险等级追加优先建议
    if res.risk_level in ("高风险", "中-高风险"):
        st.warning(
            "⚠️ 您的预测概率已超过临床决策阈值,**强烈建议尽快前往心内科**做一次"
            "包含静息心电图、心脏超声、血脂血糖在内的系统性检查。"
        )
    elif res.risk_level == "中风险":
        st.info(
            "ℹ️ 您处于中等风险区间,虽未达高危,但**建议每年做一次心血管健康体检**,"
            "并积极改善下方提到的可控因素。"
        )
    else:
        st.success(
            "✅ 您当前处于较低风险区间。继续保持健康生活方式,**每 1-2 年常规体检**即可。"
        )

    for tip in tips:
        st.markdown(f"- {tip}")


def _demo_high_risk_profile() -> dict:
    """一组典型高风险样例,便于演示。"""
    return {
        '_AGE_G': 6,        # 65+
        'SEXVAR': 1,        # 男
        '_IMPRACE': 1,
        'MARITAL': 3,       # 丧偶
        '_EDUCAG': 2,       # 高中
        '_INCOMG1': 2,      # 低收入
        'EMPLOY1': 7,       # 退休
        '_BMI5CAT': 4,      # 肥胖
        'VETERAN3': 1,
        '_STATE': 6,
        'EXERANY2': 2,      # 不锻炼
        '_SMOKER3': 1,      # 每日吸烟
        '_CURECI3': 1,
        '_RFBING6': 1,
        'CVDSTRK3': 1,      # 中风史
        'DIABETE4': 1,      # 糖尿病
        'CHCOCNC1': 2,
        'CHCCOPD3': 1,      # COPD
        'CHCKDNY2': 1,      # 肾病
        'ADDEPEV3': 1,      # 抑郁
        '_DRDXAR2': 1,      # 关节炎
        '_LTASTH1': 2,      # 哮喘
        '_RFHLTH': 2,       # 健康差
        '_PHYS14D': 3,
        '_MENT14D': 2,
        'DEAF': 1,
        'BLIND': 2,
        'DECIDE': 1,
        'DIFFWALK': 1,
        'DIFFDRES': 2,
        'DIFFALON': 1,
        '_HLTHPL2': 1,
        'MEDCOST1': 1,
        'PNEUVAC4': 2,      # 没接种
    }


# ============================================================================
# 5. 主路由
# ============================================================================
if st.session_state.stage == "form":
    render_form()
else:
    render_result()
