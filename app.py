# -*- coding: utf-8 -*-
"""
DiffDecision：多AI差异决策助手
别人给答案，我们给差异和代价。
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="DiffDecision - 多AI差异决策助手",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Mock 数据
# ============================================================

MODELS = {
    "DeepSeek": {
        "keywords": ["电商平台", "冷链物流", "品牌溢价", "社区团购"],
        "summary": "做高端苹果，走电商和社区团购",
        "color": "#4A90D9",
    },
    "GPT": {
        "keywords": ["城市批发市场", "电商", "品牌化", "供应链金融"],
        "summary": "先铺城市批发，再拿供应链金融放大",
        "color": "#10A37F",
    },
    "Claude": {
        "keywords": ["乡村集市", "电商", "合作社模式", "地理标志认证"],
        "summary": "用合作社整合农户，打地理标志，进乡村集市",
        "color": "#D97757",
    },
    "Gemini": {
        "keywords": ["城市零售", "电商", "冷链", "出口贸易"],
        "summary": "城市零售为主，尝试出口",
        "color": "#4285F4",
    },
    "Qwen": {
        "keywords": ["直播带货", "电商", "产地直发", "乡村旅游采摘"],
        "summary": "直播+产地直发，结合乡村旅游采摘",
        "color": "#615CED",
    },
}

ALL_KEYWORDS = [
    "电商", "冷链物流", "品牌溢价", "社区团购",
    "城市批发市场", "品牌化", "供应链金融",
    "乡村集市", "合作社模式", "地理标志认证",
    "城市零售", "冷链", "出口贸易",
    "直播带货", "产地直发", "乡村旅游采摘",
]

CONSENSUS_WORDS = ["电商"]
HIGH_CONSENSUS_WORDS = ["冷链物流", "品牌化", "冷链"]
DIFFERENCE_WORDS = ["乡村集市", "供应链金融", "出口贸易", "直播带货", "乡村旅游采摘"]

DIFF_RANKING = [
    {"keyword": "乡村集市", "source": "Claude", "weight": 0.87, "impact": "营业额+24%，利润率-1.5pp"},
    {"keyword": "直播带货", "source": "Qwen", "weight": 0.82, "impact": "营业额+18%，利润率+0.5pp"},
    {"keyword": "供应链金融", "source": "GPT", "weight": 0.75, "impact": "现金流改善，风险+"},
    {"keyword": "出口贸易", "source": "Gemini", "weight": 0.71, "impact": "营业额+30%，门槛高"},
    {"keyword": "乡村旅游采摘", "source": "Qwen", "weight": 0.66, "impact": "客单价+，季节性强"},
]

DIFF_PANELS = {
    "乡村集市": {
        "control_vars": "城市销售点 5 个、统一价、城市目标人群 10 万人不变",
        "rows": [
            ("城市销售点", "5 个", "5 个", "0"),
            ("产品定价", "统一价", "统一价", "0"),
            ("目标人群", "城市 10 万", "城市 10 万 + 乡村 3 万", "+3 万"),
            ("运输成本", "基准", "+15%", "+15%"),
            ("渠道建设费", "0", "2 万/月", "+2 万"),
            ("预计营业额", "50 万/月", "62 万/月", "+24%"),
            ("净利润率", "18%", "16.5%", "-1.5pp"),
            ("净利润", "9 万/月", "10.23 万/月", "+13.7%"),
        ],
        "conclusion": "如果目标是利润额，乡村渠道值得做；如果目标是利润率，要谨慎，或只在高购买力乡村试点。",
    },
    "直播带货": {
        "control_vars": "传统渠道不变，新增直播带货渠道",
        "rows": [
            ("传统渠道", "保持不变", "保持不变", "0"),
            ("产品定价", "统一价", "直播专属价（低5%）", "-5%"),
            ("目标人群", "城市 10 万", "城市 10 万 + 直播观众 5 万", "+5 万"),
            ("运输成本", "基准", "产地直发减少中间环节", "-5%"),
            ("渠道建设费", "0", "1 万/月（直播运营）", "+1 万"),
            ("预计营业额", "50 万/月", "59 万/月", "+18%"),
            ("净利润率", "18%", "18.5%", "+0.5pp"),
            ("净利润", "9 万/月", "10.92 万/月", "+21.3%"),
        ],
        "conclusion": "直播带货同时提升营业额和利润率，但需要持续内容运营能力。",
    },
    "供应链金融": {
        "control_vars": "销售渠道不变，新增供应链金融工具",
        "rows": [
            ("销售渠道", "保持不变", "保持不变", "0"),
            ("资金周转", "30 天", "15 天", "-15 天"),
            ("融资成本", "0", "年化 6%", "+6%"),
            ("风险敞口", "低", "中", "+1 级"),
            ("渠道建设费", "0", "0.5 万/月", "+0.5 万"),
            ("预计营业额", "50 万/月", "55 万/月", "+10%"),
            ("净利润率", "18%", "17%", "-1pp"),
            ("净利润", "9 万/月", "9.35 万/月", "+3.9%"),
        ],
        "conclusion": "供应链金融改善现金流，但增加融资风险，适合规模化阶段使用。",
    },
    "出口贸易": {
        "control_vars": "国内渠道不变，新增出口渠道",
        "rows": [
            ("国内渠道", "保持不变", "保持不变", "0"),
            ("产品定价", "国内价", "出口价（高20%）", "+20%"),
            ("目标人群", "城市 10 万", "城市 10 万 + 海外 2 万", "+2 万"),
            ("运输成本", "基准", "+30%（国际物流）", "+30%"),
            ("渠道建设费", "0", "3 万/月", "+3 万"),
            ("预计营业额", "50 万/月", "65 万/月", "+30%"),
            ("净利润率", "18%", "15%", "-3pp"),
            ("净利润", "9 万/月", "9.75 万/月", "+8.3%"),
        ],
        "conclusion": "出口贸易营业额增长显著，但门槛高、利润率下降，适合有外贸经验的团队。",
    },
    "乡村旅游采摘": {
        "control_vars": "传统销售不变，新增采摘体验项目",
        "rows": [
            ("传统销售", "保持不变", "保持不变", "0"),
            ("产品定价", "零售价", "体验价（高50%）", "+50%"),
            ("目标人群", "城市 10 万", "城市 10 万 + 游客 1 万", "+1 万"),
            ("运输成本", "基准", "减少（游客自采）", "-10%"),
            ("渠道建设费", "0", "1.5 万/月", "+1.5 万"),
            ("预计营业额", "50 万/月", "54 万/月", "+8%"),
            ("净利润率", "18%", "19%", "+1pp"),
            ("净利润", "9 万/月", "10.26 万/月", "+14%"),
        ],
        "conclusion": "采摘体验客单价高、利润率提升，但季节性强，适合作为补充渠道。",
    },
}

COMPARISON_DATA = {
    "指标": ["覆盖关键词数", "发现差异关键词", "量化影响的关键词", "可交互变量", "推荐方案可解释性"],
    "单模型": [4, 0, 0, 0, 1],
    "多模型投票": [6, 2, 0, 0, 2],
    "DiffDecision": [9, 5, 4, 4, 3],
}

RECOMMENDATION = """## 🎯 推荐方案

**核心策略：电商为基础 + 直播带货放大 + 乡村集市补充**

### 推荐理由

1. **高共识关键词（必选）**
   - **电商**：5 个模型一致认可，作为基础销售渠道
   - **冷链物流**：2 个模型提及，保鲜是苹果销售的核心保障
   - **品牌化**：2 个模型提及，提升溢价能力

2. **正向差异关键词（推荐加入）**
   - **直播带货**（Qwen 独有）：营业额 +18%，利润率 +0.5pp，同时提升两个指标
   - **乡村旅游采摘**（Qwen 独有）：客单价 +50%，利润率 +1pp，作为季节性补充

3. **需权衡的差异关键词（条件加入）**
   - **乡村集市**（Claude 独有）：营业额 +24% 但利润率 -1.5pp，建议在高购买力乡村试点
   - **供应链金融**（GPT 独有）：现金流改善但风险增加，适合规模化阶段

4. **暂不推荐**
   - **出口贸易**（Gemini 独有）：门槛高、利润率下降 3pp，初期不建议

### 方案总结
> 以电商为基础渠道，用直播带货放大销量，用乡村旅游采摘提升客单价，
> 在高购买力乡村试点集市销售，规模化后引入供应链金融。
> 预计综合营业额 70+ 万/月，净利润率 17%+，净利润 12+ 万/月。
"""

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
    .model-card {
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
    }
    .keyword-tag {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 13px;
    }
    .diff-tag {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 13px;
    }
    .consensus-tag {
        display: inline-block;
        background: #51cf66;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. 标题 + 输入区
# ============================================================
st.markdown("<h1 style='text-align:center;'>🍎 DiffDecision：多AI差异决策助手</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; font-size:18px;'>别人给答案，我们给差异和代价</p>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_btn = st.columns([4, 1])
with col_input:
    user_input = st.text_input("输入你的需求：", value="我想卖苹果，给我一个销售方案。")
with col_btn:
    st.button("🚀 生成方案", use_container_width=True)

st.markdown("---")

# ============================================================
# 2. 五张模型卡片
# ============================================================
st.markdown("### 📋 多模型输出")
st.markdown("系统同时调用 5 个模型，每个模型给出不同的关键词和方案摘要：")

cols = st.columns(5)
for i, (model_name, model_data) in enumerate(MODELS.items()):
    with cols[i]:
        color = model_data["color"]
        kw_html = ''.join(f'<span class="keyword-tag">{kw}</span>' for kw in model_data["keywords"])
        st.markdown(f"""
        <div class="model-card" style="border-left: 4px solid {color};">
            <h4 style="color:{color}; margin:0 0 8px 0;">{model_name}</h4>
            <div style="margin-bottom:8px;">{kw_html}</div>
            <p style="font-size:13px; color:#555; margin:0;">{model_data["summary"]}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 3. 关键词热力图
# ============================================================
st.markdown("### 🔥 关键词热力图（模型 × 关键词）")
st.markdown("颜色越深 = 该模型越强调；空白 = 未提出")

model_names = list(MODELS.keys())
heatmap_data = []
for model_name in model_names:
    row = []
    for kw in ALL_KEYWORDS:
        if kw in MODELS[model_name]["keywords"]:
            idx = MODELS[model_name]["keywords"].index(kw)
            row.append(round(1.0 - idx * 0.2, 1))
        else:
            row.append(0)
    heatmap_data.append(row)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=ALL_KEYWORDS,
    y=model_names,
    colorscale=[[0, '#ffffff'], [0.01, '#e8f5e9'], [0.5, '#66bb6a'], [1, '#1b5e20']],
    showscale=False,
    text=[[("✓" if v > 0 else "") for v in row] for row in heatmap_data],
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate="模型: %{y}<br>关键词: %{x}<br>提出: %{text}<extra></extra>",
))
fig_heat.update_layout(
    height=300,
    margin=dict(l=80, r=20, t=20, b=80),
    xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=12)),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("**关键词分类：**")
st.markdown(
    f'<span class="consensus-tag">共识词：{", ".join(CONSENSUS_WORDS)}</span>'
    f' <span class="keyword-tag" style="background:#a5d6a7;">高共识词：{", ".join(HIGH_CONSENSUS_WORDS)}</span>'
    f' <span class="diff-tag">差异词：{", ".join(DIFFERENCE_WORDS)}</span>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# 4. 差异关键词排序条形图
# ============================================================
st.markdown("### 📊 差异关键词排序")
st.markdown("按权重排序，点击下方按钮选择差异关键词查看变量对照面板：")

diff_df = pd.DataFrame(DIFF_RANKING)

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=diff_df["keyword"],
    x=diff_df["weight"],
    orientation='h',
    marker_color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db'],
    text=[f"权重 {w:.2f}" for w in diff_df["weight"]],
    textposition='inside',
    hovertemplate="关键词: %{y}<br>权重: %{x}<br>来源: %{customdata}<extra></extra>",
    customdata=diff_df["source"],
))
fig_bar.update_layout(
    height=280,
    margin=dict(l=100, r=20, t=20, b=20),
    xaxis=dict(title="权重", range=[0, 1]),
    yaxis=dict(title=""),
    showlegend=False,
)
fig_bar.update_yaxes(autorange="reversed")
st.plotly_chart(fig_bar, use_container_width=True)

# 差异词选择按钮
st.markdown("**选择差异关键词查看变量对照：**")
selected_cols = st.columns(5)
for i, row in enumerate(DIFF_RANKING):
    with selected_cols[i]:
        kw = row["keyword"]
        src = row["source"]
        if st.button(f"🔍 {kw}\n({src})", key=f"diff_{i}", use_container_width=True):
            st.session_state["selected_diff"] = kw

if "selected_diff" not in st.session_state:
    st.session_state["selected_diff"] = "乡村集市"

selected_diff = st.session_state["selected_diff"]
panel_data = DIFF_PANELS[selected_diff]

st.markdown("---")

# ============================================================
# 5. 变量对照面板
# ============================================================
st.markdown(f"### 🎛️ 变量对照面板 — {selected_diff}")
st.markdown(f"**控制变量：** {panel_data['control_vars']}")

# 对照表格
col_table, col_sliders = st.columns([1, 1])

with col_table:
    st.markdown("#### 📋 对照表")
    table_data = []
    for row in panel_data["rows"]:
        table_data.append({
            "变量": row[0],
            "对照组": row[1],
            "处理组": row[2],
            "差异": row[3],
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    st.info(f"💡 **结论：** {panel_data['conclusion']}")

with col_sliders:
    st.markdown("#### 🎚️ 交互滑块")
    st.markdown("拖动滑块，实时查看营业额、利润率、净利润变化：")

    slider_transport = st.slider("🚚 运输成本增加", 0, 40, 15, step=1, help="百分比")
    slider_rural = st.slider("🏘️ 乡村购买力（万人）", 1, 10, 3, step=1)
    slider_channel = st.slider("💰 渠道建设费（万/月）", 0, 10, 2, step=1)
    slider_competition = st.select_slider("🏙️ 城市竞争强度", options=["低", "中", "高"], value="中")

    # 实时计算
    base_revenue = 50.0  # 万/月
    base_profit_rate = 18.0  # %

    # 运输成本影响：每增加1%，利润率下降0.05pp
    profit_rate = base_profit_rate - slider_transport * 0.05

    # 乡村购买力影响：每1万人增加营业额4万/月
    rural_revenue = slider_rural * 4.0

    # 城市竞争影响
    comp_factor = {"低": 1.0, "中": 0.9, "高": 0.8}[slider_competition]

    # 总营业额
    total_revenue = (base_revenue + rural_revenue) * comp_factor

    # 净利润 = 营业额 * 利润率/100 - 渠道建设费
    net_profit = total_revenue * profit_rate / 100 - slider_channel

    # 实际利润率（扣除渠道费后）
    actual_profit_rate = (net_profit / total_revenue * 100) if total_revenue > 0 else 0

    # 显示结果
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("预计营业额", f"{total_revenue:.1f} 万/月", f"{((total_revenue/base_revenue)-1)*100:+.1f}%")
    with m2:
        st.metric("净利润率", f"{actual_profit_rate:.1f}%", f"{actual_profit_rate - base_profit_rate:+.1f}pp")
    with m3:
        st.metric("净利润", f"{net_profit:.2f} 万/月", f"{(net_profit/9-1)*100:+.1f}%")

    # 营业额构成图
    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Bar(
        x=["城市基础营业额", "乡村新增营业额", "竞争调整"],
        y=[base_revenue * comp_factor, rural_revenue * comp_factor, 0],
        marker_color=['#3498db', '#27ae60', '#e74c3c'],
        text=[f"{base_revenue * comp_factor:.1f}万", f"{rural_revenue * comp_factor:.1f}万", ""],
        textposition='inside',
    ))
    fig_gauge.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=False,
        yaxis=dict(title="万/月"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ============================================================
# 6. 推荐方案
# ============================================================
st.markdown("### 🎯 推荐方案")
st.markdown(RECOMMENDATION)

st.markdown("---")

# ============================================================
# 7. 对比实验
# ============================================================
st.markdown("### ⚖️ 对比实验：单模型 vs 多模型投票 vs DiffDecision")

comp_df = pd.DataFrame(COMPARISON_DATA)

# 表格
st.markdown("#### 📊 数据对比")
# 用文字替换可解释性数值
display_df = comp_df.copy()
display_df.loc[display_df["指标"] == "推荐方案可解释性", "单模型"] = "低"
display_df.loc[display_df["指标"] == "推荐方案可解释性", "多模型投票"] = "中"
display_df.loc[display_df["指标"] == "推荐方案可解释性", "DiffDecision"] = "高"
st.dataframe(display_df, use_container_width=True, hide_index=True)

# 条形图（数值化对比，排除可解释性行）
numeric_df = comp_df[comp_df["指标"] != "推荐方案可解释性"].copy()

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name="单模型",
    x=numeric_df["指标"],
    y=numeric_df["单模型"],
    marker_color='#95a5a6',
))
fig_comp.add_trace(go.Bar(
    name="多模型投票",
    x=numeric_df["指标"],
    y=numeric_df["多模型投票"],
    marker_color='#3498db',
))
fig_comp.add_trace(go.Bar(
    name="DiffDecision",
    x=numeric_df["指标"],
    y=numeric_df["DiffDecision"],
    marker_color='#e74c3c',
))
fig_comp.update_layout(
    barmode='group',
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis=dict(title="数值"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("""
**评委一眼看懂：**
- 🔴 **单模型**：给答案 → 覆盖 4 个关键词，发现 0 个差异
- 🔵 **多模型投票**：给共识 → 覆盖 6 个，但抹杀少数派洞察
- 🟢 **DiffDecision**：给差异和代价 → 覆盖 9 个，发现 5 个差异，量化 4 个影响

> **别人给答案，我们给差异和代价。**
""")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>DiffDecision © 多AI差异决策助手 | Demo Version</p>", unsafe_allow_html=True)
