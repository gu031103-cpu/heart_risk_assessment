"""
=============================================================================
心脏病风险评估 - 问卷配置 (Questionnaire Schema)
=============================================================================
集成逻辑：
1. MANDATORY_KEYS 定义了预处理阶段无缺失值的指标，这些项为必填。
2. 脚本会自动为非必填项追加 "不确定 / 拒绝回答" (None) 选项。
"""

from typing import Dict, List

# 数据预处理阶段无缺失值、未进入插补器的核心指标（必须有值）
MANDATORY_KEYS = ['_AGE_G', 'SEXVAR', '_IMPRACE', '_STATE']

# 完整的州列表配置
STATE_OPTIONS = [
    ("加利福尼亚 California", 6), ("纽约 New York", 36), ("德克萨斯 Texas", 48), 
    ("华盛顿 Washington", 53), ("佛罗里达 Florida", 12), ("伊利诺伊 Illinois", 17),
    ("阿拉巴马 Alabama", 1), ("阿拉斯加 Alaska", 2), ("亚利桑那 Arizona", 4), 
    ("阿肯色 Arkansas", 5), ("科罗拉多 Colorado", 8), ("康涅狄格 Connecticut", 9), 
    ("特拉华 Delaware", 10), ("佐治亚 Georgia", 13), ("夏威夷 Hawaii", 15), 
    ("爱达荷 Idaho", 16), ("印第安纳 Indiana", 18), ("爱荷华 Iowa", 19), 
    ("堪萨斯 Kansas", 20), ("肯塔基 Kentucky", 21), ("路易斯安那 Louisiana", 22), 
    ("缅因 Maine", 23), ("马里兰 Maryland", 24), ("马萨诸塞 Massachusetts", 25), 
    ("密歇根 Michigan", 26), ("明尼苏达 Minnesota", 27), ("密西西比 Mississippi", 28), 
    ("密苏里 Missouri", 29), ("蒙大拿 Montana", 30), ("内布拉斯加 Nebraska", 31), 
    ("内华达 Nevada", 32), ("新罕布什尔 New Hampshire", 33), ("新泽西 New Jersey", 34), 
    ("新墨西哥 New Mexico", 35), ("北卡罗来纳 North Carolina", 37), ("北达科他 North Dakota", 38), 
    ("俄亥俄 Ohio", 39), ("俄克拉何马 Oklahoma", 40), ("俄勒冈 Oregon", 41), 
    ("宾夕法尼亚 Pennsylvania", 42), ("罗得岛 Rhode Island", 44), ("南卡罗来纳 South Carolina", 45), 
    ("南达科他 South Dakota", 46), ("田纳西 Tennessee", 47), ("犹他 Utah", 49), 
    ("佛蒙特 Vermont", 50), ("弗吉尼亚 Virginia", 51), ("西弗吉尼亚 West Virginia", 54), 
    ("威斯康星 Wisconsin", 55), ("怀俄明 Wyoming", 56), ("其他/非美国地区", 99)
]

QUESTIONS: Dict[str, List[Dict]] = {
    "👤 基本人口学信息": [
        {"key": "_AGE_G", "label": "您的年龄分组",
         "options": [
             ("18 - 24 岁", 1), ("25 - 34 岁", 2), ("35 - 44 岁", 3),
             ("45 - 54 岁", 4), ("55 - 64 岁", 5), ("65 岁及以上", 6),
         ]},
        {"key": "SEXVAR", "label": "您的生理性别",
         "options": [("男性", 1), ("女性", 2)]},
        {"key": "_IMPRACE", "label": "您的种族/民族",
         "options": [
             ("非西裔白人 (White, Non-Hispanic)", 1),
             ("非西裔黑人 (Black, Non-Hispanic)", 2),
             ("非西裔亚裔 (Asian, Non-Hispanic)", 3),
             ("美洲原住民 (AI/AN, Non-Hispanic)", 4),
             ("西班牙裔 (Hispanic)", 5),
             ("其他/混合 (Other)", 6),
         ]},
        {"key": "MARITAL", "label": "您的婚姻状况",
         "options": [
             ("已婚 (Married)", 1), ("离婚 (Divorced)", 2),
             ("丧偶 (Widowed)", 3), ("分居 (Separated)", 4),
             ("从未结婚 (Never married)", 5),
             ("未婚同居 (Unmarried couple)", 6),
         ]},
        {"key": "_EDUCAG", "label": "您的最高学历",
         "options": [
             ("未完成高中", 1), ("高中毕业", 2),
             ("就读大学/技校", 3), ("大学/技校毕业", 4),
         ]},
        {"key": "_INCOMG1", "label": "家庭年收入(美元)",
         "options": [
             ("少于 $15,000", 1), ("$15,000 – $25,000", 2),
             ("$25,000 – $35,000", 3), ("$35,000 – $50,000", 4),
             ("$50,000 – $100,000", 5), ("$100,000 – $200,000", 6),
             ("$200,000 及以上", 7),
         ]},
        {"key": "EMPLOY1", "label": "您的就业状况",
         "options": [
             ("受雇就业 (Employed for wages)", 1),
             ("自雇 (Self-employed)", 2),
             ("失业 ≥1 年 (Out of work ≥1 year)", 3),
             ("失业 <1 年 (Out of work <1 year)", 4),
             ("家庭主妇/夫 (Homemaker)", 5),
             ("学生 (Student)", 6),
             ("退休 (Retired)", 7),
             ("无法工作 (Unable to work)", 8),
         ]},
        {"key": "_BMI5CAT", "label": "您的 BMI(体重指数)分类",
         "help": "BMI = 体重(kg) ÷ 身高(m)² 。<18.5 偏瘦,18.5-24.9 正常,25-29.9 超重,≥30 肥胖。",
         "options": [
             ("体重不足 (BMI < 18.5)", 1),
             ("正常体重 (18.5 ≤ BMI < 25)", 2),
             ("超重 (25 ≤ BMI < 30)", 3),
             ("肥胖 (BMI ≥ 30)", 4),
         ]},
        {"key": "VETERAN3", "label": "您是否为退伍军人",
         "options": [("是", 1), ("否", 2)]},
        {"key": "_STATE", "label": "您的居住地(美国州)",
         "options": STATE_OPTIONS},
    ],

    "🚬 生活方式": [
        {"key": "EXERANY2", "label": "过去 30 天您是否有过任何体育锻炼", "options": [("是", 1), ("否", 2)]},
        {"key": "_SMOKER3", "label": "您当前的吸烟状况",
         "options": [
             ("当前每日吸烟", 1), ("当前偶尔吸烟", 2), ("曾吸烟但已戒", 3), ("从不吸烟", 4),
         ]},
        {"key": "_CURECI3", "label": "您当前是否使用电子烟", "options": [("不使用", 1), ("使用", 2)]},
        {"key": "_RFBING6", "label": "您是否有暴饮行为", "options": [("否", 1), ("是", 2)]},
    ],

    "🩺 既往慢性病史": [
        {"key": "CVDSTRK3", "label": "您是否曾被诊断为中风", "options": [("是", 1), ("否", 2)]},
        {"key": "DIABETE4", "label": "您是否曾被诊断为糖尿病",
         "options": [
             ("是", 1), ("是,但仅在怀孕期间", 2), ("否", 3), ("糖尿病前期/临界", 4),
         ]},
        {"key": "CHCOCNC1", "label": "您是否曾被诊断为癌症(含黑色素瘤)", "options": [("是", 1), ("否", 2)]},
        {"key": "CHCCOPD3", "label": "您是否曾被诊断为慢阻肺/肺气肿/慢性支气管炎", "options": [("是", 1), ("否", 2)]},
        {"key": "CHCKDNY2", "label": "您是否曾被诊断为肾病", "options": [("是", 1), ("否", 2)]},
        {"key": "ADDEPEV3", "label": "您是否曾被诊断为抑郁症", "options": [("是", 1), ("否", 2)]},
        {"key": "_DRDXAR2", "label": "您是否曾被医生诊断为关节炎", "options": [("是", 1), ("否", 2)]},
        {"key": "_LTASTH1", "label": "您一生中是否曾患有哮喘", "options": [("否", 1), ("是", 2)]},
    ],

    "♿ 健康状况与功能状态": [
        {"key": "_RFHLTH", "label": "您对自身整体健康的评价", "options": [("良好或更好", 1), ("一般或较差", 2)]},
        {"key": "_PHYS14D", "label": "过去 30 天身体不适天数", "options": [("0 天", 1), ("1 - 13 天", 2), ("14 天及以上", 3)]},
        {"key": "_MENT14D", "label": "过去 30 天心理/情绪不佳天数", "options": [("0 天", 1), ("1 - 13 天", 2), ("14 天及以上", 3)]},
        {"key": "DEAF", "label": "您是否聋或有严重听力障碍", "options": [("是", 1), ("否", 2)]},
        {"key": "BLIND", "label": "您是否盲或有严重视力障碍", "options": [("是", 1), ("否", 2)]},
        {"key": "DECIDE", "label": "您是否有注意力集中或记忆困难", "options": [("是", 1), ("否", 2)]},
        {"key": "DIFFWALK", "label": "您是否有行走或爬楼梯困难", "options": [("是", 1), ("否", 2)]},
        {"key": "DIFFDRES", "label": "您是否有穿衣或洗澡困难", "options": [("是", 1), ("否", 2)]},
        {"key": "DIFFALON", "label": "您是否有独自外出办事的困难", "options": [("是", 1), ("否", 2)]},
    ],

    "🏥 医疗保健获取": [
        {"key": "_HLTHPL2", "label": "您是否拥有任何形式的医疗保险", "options": [("是", 1), ("否", 2)]},
        {"key": "MEDCOST1", "label": "过去一年是否曾因费用问题未能就医", "options": [("是", 1), ("否", 2)]},
        {"key": "PNEUVAC4", "label": "您是否曾接种肺炎疫苗", "options": [("是", 1), ("否", 2)]},
    ],
}

# 核心逻辑：自动补全 "不确定" 选项
for sec in QUESTIONS.values():
    for q in sec:
        if q['key'] not in MANDATORY_KEYS:
            q['options'].append(("不确定 / 拒绝回答", None))

def get_default_answers() -> Dict[str, any]:
    """生成默认答案：必填项选第一个，可选项选 None。"""
    defaults = {}
    for sec in QUESTIONS.values():
        for q in sec:
            if q['key'] in MANDATORY_KEYS:
                defaults[q['key']] = q['options'][0][1]
            else:
                defaults[q['key']] = None
    return defaults
