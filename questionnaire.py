"""
=============================================================================
心脏病风险评估 - 问卷配置 (Questionnaire Schema)
=============================================================================
按业务语义把 35 个 BRFSS 字段组织成 5 个面向用户的问卷板块；
已新增全美各州 FIPS 编码，并为每一题追加 "不确定 / 拒绝回答" (None) 选项，
以充分激活底层预处理阶段训练好的缺失值插补模型 (IterativeImputer / ModeImputer)。
"""

from typing import Dict, List

# 全美 50 州 + 特区 + 常用领地 FIPS 编码映射
STATE_OPTIONS = [
    ("阿拉巴马州 Alabama", 1), ("阿拉斯加州 Alaska", 2), ("亚利桑那州 Arizona", 4),
    ("阿肯色州 Arkansas", 5), ("加利福尼亚州 California", 6), ("科罗拉多州 Colorado", 8),
    ("康涅狄格州 Connecticut", 9), ("特拉华州 Delaware", 10), ("哥伦比亚特区 DC", 11),
    ("佛罗里达州 Florida", 12), ("佐治亚州 Georgia", 13), ("夏威夷州 Hawaii", 15),
    ("爱达荷州 Idaho", 16), ("伊利诺伊州 Illinois", 17), ("印第安纳州 Indiana", 18),
    ("爱荷华州 Iowa", 19), ("堪萨斯州 Kansas", 20), ("肯塔基州 Kentucky", 21),
    ("路易斯安那州 Louisiana", 22), ("缅因州 Maine", 23), ("马里兰州 Maryland", 24),
    ("马萨诸塞州 Massachusetts", 25), ("密歇根州 Michigan", 26), ("明尼苏达州 Minnesota", 27),
    ("密西西比州 Mississippi", 28), ("密苏里州 Missouri", 29), ("蒙大拿州 Montana", 30),
    ("内布拉斯加州 Nebraska", 31), ("内华达州 Nevada", 32), ("新罕布什尔州 New Hampshire", 33),
    ("新泽西州 New Jersey", 34), ("新墨西哥州 New Mexico", 35), ("纽约州 New York", 36),
    ("北卡罗来纳州 North Carolina", 37), ("北达科他州 North Dakota", 38), ("俄亥俄州 Ohio", 39),
    ("俄克拉何马州 Oklahoma", 40), ("俄勒冈州 Oregon", 41), ("宾夕法尼亚州 Pennsylvania", 42),
    ("罗得岛州 Rhode Island", 44), ("南卡罗来纳州 South Carolina", 45), ("南达科他州 South Dakota", 46),
    ("田纳西州 Tennessee", 47), ("德克萨斯州 Texas", 48), ("犹他州 Utah", 49),
    ("佛蒙特州 Vermont", 50), ("弗吉尼亚州 Virginia", 51), ("华盛顿州 Washington", 53),
    ("西弗吉尼亚州 West Virginia", 54), ("威斯康星州 Wisconsin", 55), ("怀俄明州 Wyoming", 56),
    ("关岛 Guam", 66), ("波多黎各 Puerto Rico", 72), ("非美国地区/其他", 99)
]

QUESTIONS: Dict[str, List[Dict]] = {

    "👤 基本人口学信息": [
        {"key": "_AGE_G", "label": "您的年龄分组",
         "options": [
             ("18 - 24 岁", 1), ("25 - 34 岁", 2), ("35 - 44 岁", 3),
             ("45 - 54 岁", 4), ("55 - 64 岁", 5), ("65 岁及以上", 6),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "SEXVAR", "label": "您的生理性别",
         "options": [("男性", 1), ("女性", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_IMPRACE", "label": "您的种族/民族",
         "options": [
             ("非西裔白人 (White, Non-Hispanic)", 1),
             ("非西裔黑人 (Black, Non-Hispanic)", 2),
             ("非西裔亚裔 (Asian, Non-Hispanic)", 3),
             ("美洲原住民 (AI/AN, Non-Hispanic)", 4),
             ("西班牙裔 (Hispanic)", 5),
             ("其他/混合 (Other)", 6),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "MARITAL", "label": "您的婚姻状况",
         "options": [
             ("已婚 (Married)", 1), ("离婚 (Divorced)", 2),
             ("丧偶 (Widowed)", 3), ("分居 (Separated)", 4),
             ("从未结婚 (Never married)", 5),
             ("未婚同居 (Unmarried couple)", 6),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "_EDUCAG", "label": "您的最高学历",
         "options": [
             ("未完成高中", 1), ("高中毕业", 2),
             ("就读大学/技校", 3), ("大学/技校毕业", 4),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "_INCOMG1", "label": "家庭年收入(美元)",
         "options": [
             ("少于 $15,000", 1), ("$15,000 – $25,000", 2),
             ("$25,000 – $35,000", 3), ("$35,000 – $50,000", 4),
             ("$50,000 – $100,000", 5), ("$100,000 – $200,000", 6),
             ("$200,000 及以上", 7), ("不确定 / 拒绝回答", None)
         ]},
        {"key": "EMPLOY1", "label": "您的就业状况",
         "options": [
             ("受雇就业 (Employed)", 1), ("自雇 (Self-employed)", 2),
             ("失业 ≥1 年 (Out of work ≥1 yr)", 3), ("失业 <1 年 (Out of work <1 yr)", 4),
             ("家庭主妇/夫 (Homemaker)", 5), ("学生 (Student)", 6),
             ("退休 (Retired)", 7), ("无法工作 (Unable to work)", 8),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "_BMI5CAT", "label": "您的 BMI(体重指数)分类",
         "help": "BMI = 体重(kg) ÷ 身高(m)² 。<18.5 偏瘦,18.5-24.9 正常,25-29.9 超重,≥30 肥胖。",
         "options": [
             ("体重不足 (BMI < 18.5)", 1), ("正常体重 (18.5 ≤ BMI < 25)", 2),
             ("超重 (25 ≤ BMI < 30)", 3), ("肥胖 (BMI ≥ 30)", 4),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "VETERAN3", "label": "您是否为退伍军人",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_STATE", "label": "您的居住地",
         "help": "支持全美 50 州。若不在美国,请选最后一项。",
         "options": STATE_OPTIONS + [("不确定 / 拒绝回答", None)]},
    ],

    "🚬 生活方式": [
        {"key": "EXERANY2", "label": "过去 30 天您是否有过任何体育锻炼",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_SMOKER3", "label": "您当前的吸烟状况",
         "options": [
             ("当前每日吸烟", 1), ("当前偶尔吸烟", 2),
             ("曾吸烟但已戒", 3), ("从不吸烟", 4),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "_CURECI3", "label": "您当前是否使用电子烟",
         "options": [("不使用", 1), ("使用", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_RFBING6", "label": "您是否有暴饮行为 (男性单次≥5杯/女性≥4杯)",
         "options": [("否", 1), ("是", 2), ("不确定 / 拒绝回答", None)]},
    ],

    "🩺 既往慢性病史": [
        {"key": "CVDSTRK3", "label": "您是否曾被诊断为中风",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "DIABETE4", "label": "您是否曾被诊断为糖尿病",
         "options": [
             ("是", 1), ("是,但仅在怀孕期间", 2),
             ("否", 3), ("糖尿病前期/临界", 4),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "CHCOCNC1", "label": "您是否曾被诊断为癌症(含黑色素瘤)",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "CHCCOPD3", "label": "您是否曾被诊断为慢阻肺/肺气肿/慢性支气管炎",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "CHCKDNY2", "label": "您是否曾被诊断为肾病",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "ADDEPEV3", "label": "您是否曾被诊断为抑郁症",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_DRDXAR2", "label": "您是否曾被医生诊断为关节炎",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "_LTASTH1", "label": "您一生中是否曾患有哮喘",
         "options": [("否", 1), ("是", 2), ("不确定 / 拒绝回答", None)]},
    ],

    "♿ 健康状况与功能状态": [
        {"key": "_RFHLTH", "label": "您对自身整体健康的评价",
         "options": [
             ("良好或更好 (Good/Excellent)", 1), ("一般或较差 (Fair/Poor)", 2),
             ("不确定 / 拒绝回答", None)
         ]},
        {"key": "_PHYS14D", "label": "过去 30 天身体不适天数",
         "options": [("0 天", 1), ("1 - 13 天", 2), ("14 天及以上", 3), ("不确定 / 拒绝回答", None)]},
        {"key": "_MENT14D", "label": "过去 30 天心理/情绪不佳天数",
         "options": [("0 天", 1), ("1 - 13 天", 2), ("14 天及以上", 3), ("不确定 / 拒绝回答", None)]},
        {"key": "DEAF", "label": "您是否聋或有严重听力障碍",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "BLIND", "label": "您是否盲或有严重视力障碍",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "DECIDE", "label": "您是否有注意力集中或记忆困难",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "DIFFWALK", "label": "您是否有行走或爬楼梯困难",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "DIFFDRES", "label": "您是否有穿衣或洗澡困难",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "DIFFALON", "label": "您是否有独自外出办事的困难",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
    ],

    "🏥 医疗保健获取": [
        {"key": "_HLTHPL2", "label": "您是否拥有任何形式的医疗保险",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "MEDCOST1", "label": "过去一年是否曾因费用问题未能就医",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
        {"key": "PNEUVAC4", "label": "您是否曾接种肺炎疫苗",
         "options": [("是", 1), ("否", 2), ("不确定 / 拒绝回答", None)]},
    ],
}

def get_default_answers() -> Dict[str, float]:
    """生成一份默认问卷答案(所有题目选 None，交由底层 Pipeline 插补)。"""
    return {q["key"]: None for sec in QUESTIONS.values() for q in sec}
