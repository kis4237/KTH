"""
EcoShip-Analyzer: 선종별 연료 효율 및 탄소 배출 시뮬레이션 대시보드
실행 방법: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import rcParams
from datetime import datetime
import platform

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    rcParams["font.family"] = "Malgun Gothic"
else:
    try:
        import matplotlib.font_manager as fm
        nanum = [f for f in fm.findSystemFonts() if "Nanum" in f]
        if nanum:
            fm.fontManager.addfont(nanum[0])
            rcParams["font.family"] = fm.FontProperties(fname=nanum[0]).get_name()
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. 선종별 기본 제원 데이터베이스
# ══════════════════════════════════════════════════════════════════════════════
SHIP_DB = pd.DataFrame([
    {
        "ship_type":    "Container (10k TEU)",
        "displacement": 120_000,
        "design_speed": 22.0,
        "mcr_kw":       50_000,
        "sfoc_base":    165,
        "speed_min":    14.0,
        "speed_max":    25.0,
    },
    {
        "ship_type":    "Bulk Carrier (Capesize)",
        "displacement": 180_000,
        "design_speed": 14.5,
        "mcr_kw":       18_000,
        "sfoc_base":    170,
        "speed_min":    10.0,
        "speed_max":    18.0,
    },
    {
        "ship_type":    "VLCC (Tanker)",
        "displacement": 300_000,
        "design_speed": 15.0,
        "mcr_kw":       25_000,
        "sfoc_base":    168,
        "speed_min":    10.0,
        "speed_max":    18.0,
    },
    {
        "ship_type":    "LNGC",
        "displacement":  90_000,
        "design_speed": 19.5,
        "mcr_kw":       35_000,
        "sfoc_base":    160,
        "speed_min":    12.0,
        "speed_max":    22.0,
    },
])
SHIP_DB = SHIP_DB.set_index("ship_type")

# ── 연료별 탄소 배출계수 Cf ───────────────────────────────────────────────────
FUEL_DB = {
    "HFO (중유)":        {"cf": 3.114, "label": "HFO",      "color": "#c0392b"},
    "MDO (경유)":        {"cf": 3.206, "label": "MDO",      "color": "#e67e22"},
    "LNG":               {"cf": 2.750, "label": "LNG",      "color": "#2980b9"},
    "메탄올(Methanol)":  {"cf": 1.375, "label": "Methanol", "color": "#27ae60"},
}

# ── CII 등급 임계값 ───────────────────────────────────────────────────────────
CII_THRESHOLDS = {"A": 3.5, "B": 5.0, "C": 7.0, "D": 9.5, "E": float("inf")}
CII_COLORS     = {"A": "#27ae60", "B": "#2ecc71", "C": "#f39c12", "D": "#e67e22", "E": "#c0392b"}

CHART_BG   = "#0f1117"
AXIS_COLOR = "#aaaaaa"
GRID_COLOR = "#2a2a3a"
ACCENT     = "#00d4ff"
WARN_COLOR = "#f39c12"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 공학 연산 엔진
# ══════════════════════════════════════════════════════════════════════════════

def admiralty_coefficient(displacement, speed, bhp):
    return (displacement ** (2 / 3) * speed ** 3) / bhp

def bhp_from_admiralty(displacement, speed_array, c_adm):
    return (displacement ** (2 / 3) * speed_array ** 3) / c_adm

def fuel_consumption_per_hour(bhp_array, sfoc):
    return bhp_array * sfoc / 1_000_000

def co2_per_nautical_mile(fc_ton_hr, speed_array, cf):
    return fc_ton_hr / speed_array * cf

def cii_score(co2_per_nm, displacement, speed):
    cii_val = co2_per_nm * 1_000_000 / (displacement * speed)
    for grade, threshold in CII_THRESHOLDS.items():
        if cii_val <= threshold:
            return grade, cii_val
    return "E", cii_val

def run_calculations(ship_row, speed_selected, fuel_key):
    disp  = ship_row["displacement"]
    mcr   = ship_row["mcr_kw"]
    spd_d = ship_row["design_speed"]
    sfoc  = ship_row["sfoc_base"]
    cf    = FUEL_DB[fuel_key]["cf"]

    c_adm   = admiralty_coefficient(disp, spd_d, mcr)
    v_arr   = np.arange(ship_row["speed_min"], ship_row["speed_max"] + 0.1, 0.5)
    bhp_arr = bhp_from_admiralty(disp, v_arr, c_adm)
    fc_arr  = fuel_consumption_per_hour(bhp_arr, sfoc)
    co2_arr = co2_per_nautical_mile(fc_arr, v_arr, cf)

    v_sel   = np.clip(speed_selected, ship_row["speed_min"], ship_row["speed_max"])
    bhp_sel = bhp_from_admiralty(disp, np.array([v_sel]), c_adm)[0]
    fc_sel  = fuel_consumption_per_hour(np.array([bhp_sel]), sfoc)[0]
    co2_sel = co2_per_nautical_mile(np.array([fc_sel]), np.array([v_sel]), cf)[0]
    grade, cii_val = cii_score(co2_sel, disp, v_sel)

    fuel_compare = {}
    for fk, fv in FUEL_DB.items():
        co2_cmp = co2_per_nautical_mile(np.array([fc_sel]), np.array([v_sel]), fv["cf"])[0]
        fuel_compare[fv["label"]] = {"co2_ton_per_nm": co2_cmp, "color": fv["color"]}

    return {
        "v_arr": v_arr, "bhp_arr": bhp_arr, "fc_arr": fc_arr, "co2_arr": co2_arr,
        "v_sel": v_sel, "bhp_sel": bhp_sel, "fc_sel": fc_sel, "co2_sel": co2_sel,
        "grade": grade, "cii_val": cii_val, "c_adm": c_adm,
        "fuel_compare": fuel_compare,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Matplotlib 그래프 함수
# ══════════════════════════════════════════════════════════════════════════════

def make_speed_curve(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CHART_BG)
    fig.subplots_adjust(wspace=0.35)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(CHART_BG)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_xlabel(xlabel, color=AXIS_COLOR, fontsize=9)
        ax.set_ylabel(ylabel, color=AXIS_COLOR, fontsize=9)
        ax.tick_params(colors=AXIS_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)

    ax0 = axes[0]
    style_ax(ax0, "Speed – Fuel Consumption Curve", "Speed (knots)", "Fuel Consumption (ton/hr)")
    ax0.plot(res["v_arr"], res["fc_arr"], color=ACCENT, linewidth=2.0, label="FC Curve")
    ax0.scatter([res["v_sel"]], [res["fc_sel"]], color=WARN_COLOR, s=90, zorder=5,
                label=f"현재 속도 {res['v_sel']:.1f} kn")
    ax0.axvline(res["v_sel"], color=WARN_COLOR, linewidth=1.0, linestyle=":", alpha=0.7)
    ax0.legend(fontsize=8, facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor="white")

    ax1 = axes[1]
    style_ax(ax1, "Speed – CO₂ Emission Curve", "Speed (knots)", "CO₂ (ton/nm)")
    ax1.plot(res["v_arr"], res["co2_arr"], color="#ff6b6b", linewidth=2.0, label="CO₂ Curve")
    ax1.scatter([res["v_sel"]], [res["co2_sel"]], color=WARN_COLOR, s=90, zorder=5,
                label=f"현재 속도 {res['v_sel']:.1f} kn")
    ax1.axvline(res["v_sel"], color=WARN_COLOR, linewidth=1.0, linestyle=":", alpha=0.7)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor="white")

    return fig


def make_fuel_comparison(res):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    labels = list(res["fuel_compare"].keys())
    values = [v["co2_ton_per_nm"] for v in res["fuel_compare"].values()]
    colors = [v["color"] for v in res["fuel_compare"].values()]
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor=CHART_BG)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=9)
    ax.set_xlabel("CO₂ Emission (ton/nm)", color=AXIS_COLOR, fontsize=9)
    ax.set_title(f"연료별 탄소 배출량 비교 @ {res['v_sel']:.1f} kn",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors=AXIS_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlim(0, max(values) * 1.18)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plotly 차트 함수 (운항 기록 탭용)
# ══════════════════════════════════════════════════════════════════════════════

def make_plotly_bar(df):
    ship_co2 = df.groupby("선박명")["CO2배출량(ton)"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ship_co2["선박명"],
        y=ship_co2["CO2배출량(ton)"],
        marker_color="tomato",
        name="CO2 배출량",
    ))
    fig.update_layout(
        title="선박별 누적 CO2 배출량",
        xaxis_title="선박",
        yaxis_title="CO2 (ton)",
        height=380,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#aaaaaa",
    )
    return fig


def make_plotly_line(df):
    daily = df.groupby("날짜")["CO2배출량(ton)"].sum().reset_index().sort_values("날짜")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["날짜"],
        y=daily["CO2배출량(ton)"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=8),
        name="일일 배출량",
    ))
    fig.update_layout(
        title="날짜별 전체 CO2 배출량 추이",
        xaxis_title="날짜",
        yaxis_title="CO2 (ton)",
        height=380,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#aaaaaa",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Session State 초기화
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    if "ship_log" not in st.session_state:
        st.session_state.ship_log = pd.DataFrame(
            columns=["날짜", "선박명", "연료종류", "연료투입량(ton)", "CO2배출량(ton)"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="EcoShip-Analyzer",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; color:#00d4ff; font-family:monospace; letter-spacing:2px;'>
        🚢 EcoShip-Analyzer
    </h1>
    <p style='text-align:center; color:#888; font-size:14px; margin-top:-10px;'>
        선종별 연료 효율 및 탄소 배출 시뮬레이션 대시보드 &nbsp;|&nbsp; IMO CII 등급 평가
    </p>
    <hr style='border-color:#2a2a3a;'>
    """, unsafe_allow_html=True)

    # ── 사이드바 ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ 운항 조건 설정")
        st.divider()

        ship_names    = list(SHIP_DB.index)
        ship_selected = st.selectbox("🛳️ 선종 선택", ship_names, index=0)
        ship          = SHIP_DB.loc[ship_selected]

        st.caption(
            f"배수량: {ship['displacement']:,} ton  |  "
            f"설계속도: {ship['design_speed']} kn  |  "
            f"MCR: {ship['mcr_kw']:,} kW"
        )
        st.divider()

        speed = st.slider(
            "⚡ 운항 속도 (knots)",
            min_value=float(ship["speed_min"]),
            max_value=float(ship["speed_max"]),
            value=float(ship["design_speed"]),
            step=0.5,
            format="%.1f kn",
        )

        if speed > ship["design_speed"] * 0.95:
            st.warning("⚠️ 설계 속도 근처 – 연료 소모량이 급증합니다.")
        elif speed < ship["design_speed"] * 0.6:
            st.info("ℹ️ 저속 항해 – 연료 효율은 개선되나 일정에 영향을 줄 수 있습니다.")

        st.divider()

        fuel_key  = st.selectbox("⛽ 연료 종류", list(FUEL_DB.keys()), index=0)
        fuel_info = FUEL_DB[fuel_key]
        st.caption(f"탄소 배출계수 Cf = {fuel_info['cf']} ton CO₂/ton fuel")

        st.divider()
        st.markdown("##### 📌 연산 기준")
        st.caption("해군제수법 (Admiralty Coefficient)\nC = (Δ^⅔ × V³) / BHP")

    # ── 탭 구성 ───────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 시뮬레이션 대시보드", "📋 운항 기록 관리"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 시뮬레이션 대시보드 (기존 기능)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        res         = run_calculations(ship, speed, fuel_key)
        grade       = res["grade"]
        grade_color = CII_COLORS[grade]

        # KPI 메트릭
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🚀 운항 속도",  f"{res['v_sel']:.1f} kn")
        col2.metric("⚙️ 필요 마력",  f"{res['bhp_sel']:,.0f} kW")
        col3.metric("🛢️ 연료 소모",  f"{res['fc_sel']:.3f} ton/hr")
        col4.metric("💨 CO₂/해리",   f"{res['co2_sel']:.4f} t/nm")
        col5.markdown(
            f"""
            <div style='background:{grade_color}22; border:1px solid {grade_color};
                        border-radius:8px; padding:10px 14px; text-align:center;'>
                <div style='font-size:11px; color:#aaa;'>🏅 IMO CII 등급</div>
                <div style='font-size:36px; font-weight:bold; color:{grade_color}; line-height:1.1;'>{grade}</div>
                <div style='font-size:10px; color:#888;'>{res["cii_val"]:.3f} g/DWT·nm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # 그래프
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("#### 📈 Speed-Consumption / CO₂ 곡선")
            fig_speed = make_speed_curve(res)
            st.pyplot(fig_speed, use_container_width=True)
            plt.close(fig_speed)

        with col_right:
            st.markdown("#### 🔋 연료별 탄소 배출량 비교")
            fig_fuel = make_fuel_comparison(res)
            st.pyplot(fig_fuel, use_container_width=True)
            plt.close(fig_fuel)

        st.divider()

        # 상세 테이블
        with st.expander("📊 속도별 상세 연산 결과 테이블 (펼치기)"):
            cf = FUEL_DB[fuel_key]["cf"]
            df_detail = pd.DataFrame({
                "Speed (kn)":    res["v_arr"].round(1),
                "BHP (kW)":      res["bhp_arr"].round(0),
                "Fuel (ton/hr)": res["fc_arr"].round(4),
                "CO₂ (ton/nm)":  res["co2_arr"].round(5),
                "CO₂ (ton/day)": (res["fc_arr"] * cf * 24).round(2),
            })

            def highlight_selected(row):
                if abs(row["Speed (kn)"] - res["v_sel"]) < 0.01:
                    return ["background-color: #1a3a5c; color: #00d4ff"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_detail.style.apply(highlight_selected, axis=1).format(precision=4),
                use_container_width=True,
                height=280,
            )

        with st.expander("🔩 선박 기본 제원 DB 보기"):
            display_db = SHIP_DB.reset_index().rename(columns={
                "ship_type": "선종", "displacement": "배수량 (ton)",
                "design_speed": "설계속도 (kn)", "mcr_kw": "MCR (kW)",
                "sfoc_base": "SFOC (g/kWh)", "speed_min": "최저속도", "speed_max": "최고속도",
            })
            st.dataframe(display_db, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 운항 기록 관리 (친구 코드 기능 통합)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📝 연료 투입 기록 추가")

        with st.expander("새로운 연료 투입 기록 추가", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                date_input = st.date_input("투입 날짜", datetime.today())
            with c2:
                ship_log_names = list(SHIP_DB.index)
                ship_input = st.selectbox("선박 선택", ship_log_names, key="log_ship")
            with c3:
                fuel_input = st.selectbox("연료 종류", list(FUEL_DB.keys()), key="log_fuel")
            with c4:
                amount_input = st.number_input(
                    "연료 투입량 (ton)", min_value=0.0, value=100.0, step=10.0
                )

            if st.button("기록 추가", type="primary"):
                cf_log  = FUEL_DB[fuel_input]["cf"]
                co2_log = amount_input * cf_log
                new_row = pd.DataFrame([{
                    "날짜":           pd.to_datetime(date_input),
                    "선박명":         ship_input,
                    "연료종류":       fuel_input,
                    "연료투입량(ton)": amount_input,
                    "CO2배출량(ton)":  round(co2_log, 2),
                }])
                st.session_state.ship_log = pd.concat(
                    [st.session_state.ship_log, new_row], ignore_index=True
                )
                st.success(f"✅ 기록 추가 완료! CO2 배출량: **{co2_log:.2f} ton**")

        # 데이터 있을 때만 대시보드 표시
        df_log = st.session_state.ship_log

        if not df_log.empty:
            st.divider()

            # 요약 KPI
            total_co2  = df_log["CO2배출량(ton)"].sum()
            total_fuel = df_log["연료투입량(ton)"].sum()
            top_ship   = df_log.groupby("선박명")["CO2배출량(ton)"].sum().idxmax()
            record_cnt = len(df_log)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💨 총 누적 CO2 배출량", f"{total_co2:,.1f} ton")
            m2.metric("🛢️ 총 연료 투입량",     f"{total_fuel:,.1f} ton")
            m3.metric("🚢 최다 배출 선박",      top_ship)
            m4.metric("📋 총 기록 건수",         f"{record_cnt} 건")

            st.divider()

            # Plotly 차트
            st.markdown("#### 📊 배출량 분석 차트")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(make_plotly_bar(df_log), use_container_width=True)
            with ch2:
                st.plotly_chart(make_plotly_line(df_log), use_container_width=True)

            st.divider()

            # 전체 기록 테이블
            st.markdown("#### 📋 전체 투입 기록")
            col_tbl, col_btn = st.columns([5, 1])
            with col_btn:
                if st.button("🗑️ 전체 초기화", type="secondary"):
                    st.session_state.ship_log = pd.DataFrame(
                        columns=["날짜", "선박명", "연료종류", "연료투입량(ton)", "CO2배출량(ton)"]
                    )
                    st.rerun()
            st.dataframe(
                df_log.sort_values("날짜", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("👆 위에서 연료 투입 기록을 추가하면 차트와 통계가 표시됩니다.")

    # 푸터
    st.markdown("""
    <hr style='border-color:#2a2a3a; margin-top:40px;'>
    <p style='text-align:center; color:#444; font-size:11px;'>
        EcoShip-Analyzer · 해군제수법 기반 단순화 모델 · IMO CII 등급은 가상 기준 적용<br>
        실제 선박 저항은 파랑, 바람, 선체 노후도 등 추가 변수를 포함합니다.
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
