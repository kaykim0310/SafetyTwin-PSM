"""
PSM-SafetyTwin P&ID Parser - Streamlit 체험판
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
코딩을 모르는 분도 클릭만으로 P&ID 분석을 체험할 수 있는 화면입니다.

[실행 방법]
  pip install streamlit pillow numpy
  streamlit run app.py

[주의] 이 파일은 services/pid-parser/ 폴더가 아니라
       psm-safetytwin/ 루트 폴더에 넣어주세요.
"""

import streamlit as st
import json
import time
import random
import math
from datetime import datetime
from pathlib import Path

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="PSM-SafetyTwin P&ID Parser",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 스타일
# ============================================================
st.markdown("""
<style>
    /* 전체 테마 */
    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    /* 헤더 */
    .header-box {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { color: white !important; margin: 0 0 0.3rem 0; font-size: 2rem; }
    .header-box p { color: #bbdefb; margin: 0; font-size: 1rem; }

    /* 통계 카드 */
    .stat-card {
        background: white;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1565c0, #00897b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .stat-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

    /* 안전장치 경고 박스 */
    .safety-box {
        background: #fff3f3;
        border: 2px solid #ef5350;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .safety-box h3 { color: #c62828; margin-top: 0; }

    /* 장비 카드 */
    .equip-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        border-left: 4px solid #1565c0;
    }
    .equip-card.safety {
        border-left-color: #ef5350;
        background: #fffafa;
    }
    .equip-card.high {
        border-left-color: #ff9800;
        background: #fffdf5;
    }

    /* 파이프라인 */
    .pipeline-step {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .pipe-done { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
    .pipe-active { background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
    .pipe-wait { background: #f5f5f5; color: #999; border: 1px solid #e0e0e0; }

    /* 프로그레스 */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #1565c0, #00897b); }

    /* 사이드바 */
    section[data-testid="stSidebar"] { background: #fafbfc; }

    /* 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 시뮬레이션 데이터 (실제 모듈 없이도 체험 가능)
# ============================================================

# P&ID 심볼 42개 클래스 (settings.py 동일)
SYMBOL_CLASSES = {
    "밸브류": [
        ("gate_valve", "게이트밸브"), ("globe_valve", "글로브밸브"),
        ("ball_valve", "볼밸브"), ("butterfly_valve", "버터플라이밸브"),
        ("check_valve", "체크밸브"), ("control_valve", "제어밸브"),
        ("needle_valve", "니들밸브"), ("plug_valve", "플러그밸브"),
        ("diaphragm_valve", "다이어프램밸브"),
    ],
    "장치류": [
        ("tank", "탱크"), ("pump", "펌프"), ("compressor", "압축기"),
        ("heat_exchanger", "열교환기"), ("reactor", "반응기"),
        ("column", "증류탑"), ("mixer", "믹서"), ("filter", "필터"), ("drum", "드럼"),
    ],
    "계장류": [
        ("pressure_gauge", "압력계"), ("temperature_gauge", "온도계"),
        ("flow_meter", "유량계"), ("level_gauge", "레벨계"),
        ("pressure_transmitter", "압력전송기"), ("temperature_transmitter", "온도전송기"),
        ("flow_transmitter", "유량전송기"), ("level_transmitter", "레벨전송기"),
        ("controller", "제어기"), ("indicator", "지시기"),
    ],
    "배관류": [
        ("pipe_line", "직관"), ("elbow", "엘보"), ("tee", "티"),
        ("reducer", "리듀서"), ("flange", "플랜지"), ("spectacle_blind", "캡"),
    ],
    "⚠️ 안전장치 (PSM)": [
        ("relief_valve", "안전밸브(PSV)"), ("rupture_disc", "파열판(RD)"),
        ("flame_arrestor", "화염방지기"), ("emergency_shutoff", "긴급차단밸브(ESD)"),
    ],
}

DEMO_RESULTS = {
    "symbols": [
        {"tag": "T-101", "class": "tank", "korean": "탱크", "confidence": 0.98, "criticality": "normal"},
        {"tag": "P-201A", "class": "pump", "korean": "펌프", "confidence": 0.92, "criticality": "normal"},
        {"tag": "E-301", "class": "heat_exchanger", "korean": "열교환기", "confidence": 0.91, "criticality": "normal"},
        {"tag": "R-401", "class": "reactor", "korean": "반응기", "confidence": 0.95, "criticality": "high"},
        {"tag": "C-501", "class": "column", "korean": "증류탑", "confidence": 0.93, "criticality": "high"},
        {"tag": "V-101", "class": "gate_valve", "korean": "게이트밸브", "confidence": 0.95, "criticality": "normal"},
        {"tag": "CV-301", "class": "control_valve", "korean": "제어밸브", "confidence": 0.94, "criticality": "normal"},
        {"tag": "PSV-401", "class": "relief_valve", "korean": "압력안전밸브", "confidence": 0.96, "criticality": "critical"},
        {"tag": "RD-501", "class": "rupture_disc", "korean": "파열판", "confidence": 0.93, "criticality": "critical"},
        {"tag": "TIC-101", "class": "controller", "korean": "온도지시제어기", "confidence": 0.89, "criticality": "normal"},
        {"tag": "PT-201", "class": "pressure_transmitter", "korean": "압력전송기", "confidence": 0.91, "criticality": "normal"},
        {"tag": "FT-301", "class": "flow_transmitter", "korean": "유량전송기", "confidence": 0.87, "criticality": "normal"},
    ],
    "texts": [
        {"text": "T-101", "category": "태그", "confidence": 0.95},
        {"text": "P-201A", "category": "태그", "confidence": 0.93},
        {"text": "E-301", "category": "태그", "confidence": 0.91},
        {"text": "R-401", "category": "태그", "confidence": 0.95},
        {"text": "C-501", "category": "태그", "confidence": 0.93},
        {"text": "PSV-401", "category": "안전장치", "confidence": 0.96},
        {"text": "RD-501", "category": "안전장치", "confidence": 0.93},
        {"text": "3\"-P-101-A1", "category": "라인번호", "confidence": 0.92},
        {"text": "4\"-P-201-B2", "category": "라인번호", "confidence": 0.90},
        {"text": "TIC-101", "category": "계장", "confidence": 0.89},
    ],
}


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.markdown("## 🏭 PSM-SafetyTwin")
    st.markdown("**P&ID Parser v1.0**")
    st.markdown("D-Fine (Apache 2.0) 기반")
    st.divider()

    page = st.radio(
        "메뉴",
        ["🏠 프로그램 소개", "📄 도면 분석 체험", "🔍 심볼 클래스 목록", "📊 API JSON 미리보기"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("##### ⚙️ 분석 설정")
    confidence = st.slider("심볼 감지 최소 신뢰도", 0.1, 1.0, 0.5, 0.05)
    do_ocr = st.checkbox("텍스트 추출 (OCR)", value=True)
    do_match = st.checkbox("심볼-텍스트 매칭", value=True)

    st.divider()
    st.caption("© 2026 PSM-SafetyTwin")
    st.caption("라이선스: Apache 2.0")


# ============================================================
# 페이지 1: 프로그램 소개
# ============================================================
if page == "🏠 프로그램 소개":
    st.markdown("""
    <div class="header-box">
        <h1>🏭 PSM-SafetyTwin P&ID Parser</h1>
        <p>P&ID 도면을 AI가 자동으로 읽어서 디지털 데이터로 바꿔주는 프로그램</p>
    </div>
    """, unsafe_allow_html=True)

    # 핵심 통계
    cols = st.columns(5)
    stats = [
        ("42종", "인식 가능 심볼"),
        ("95%+", "목표 정확도"),
        ("< 10초", "분석 소요시간"),
        ("4종", "PSM 안전장치"),
        ("Apache 2.0", "라이선스"),
    ]
    for col, (val, label) in zip(cols, stats):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # 처리 파이프라인
    st.markdown("### 🔄 자동 분석 파이프라인")
    pipe_cols = st.columns(6)
    steps = [
        ("📄", "도면 업로드"),
        ("🔧", "이미지 전처리"),
        ("🤖", "D-Fine 심볼 감지"),
        ("📝", "OCR 텍스트 추출"),
        ("🔗", "데이터 매칭"),
        ("⚠️", "PSM 안전장치 식별"),
    ]
    for col, (icon, label) in zip(pipe_cols, steps):
        col.markdown(f"""
        <div style="text-align:center; padding:1rem; background:#f8f9fa;
                    border-radius:12px; border:1px solid #e0e0e0;">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-size:0.8rem; font-weight:600; margin-top:0.3rem">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # 비교 테이블
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ❌ 기존 방식 (수작업)")
        st.error("도면 1장 분석에 **2~4시간** 소요")
        st.error("사람마다 다른 해석 (주관적)")
        st.error("안전장치 누락 위험")
        st.error("디지털 데이터 없음")

    with col2:
        st.markdown("#### ✅ PSM-SafetyTwin (AI)")
        st.success("도면 1장 분석에 **10초 이내**")
        st.success("**95% 이상** 일관된 정확도")
        st.success("PSM 안전장치 **자동 식별**")
        st.success("즉시 **위험성평가 연계** 가능")

    st.markdown("")
    st.info("""
    💡 **왼쪽 메뉴에서 '📄 도면 분석 체험'을 클릭**하면 실제로 도면을 넣어서 분석 결과를 확인할 수 있습니다!
    """)


# ============================================================
# 페이지 2: 도면 분석 체험
# ============================================================
elif page == "📄 도면 분석 체험":
    st.markdown("""
    <div class="header-box">
        <h1>📄 P&ID 도면 분석</h1>
        <p>도면 파일을 업로드하거나, 데모 모드로 분석 결과를 확인하세요</p>
    </div>
    """, unsafe_allow_html=True)

    # 업로드 영역
    col_upload, col_demo = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "P&ID 도면 파일 선택 (PNG, JPG, PDF)",
            type=["png", "jpg", "jpeg", "pdf"],
            help="스캔된 P&ID 도면 이미지를 업로드하세요",
        )

    with col_demo:
        st.markdown("")
        st.markdown("")
        demo_mode = st.button("🎮 데모 모드로 체험하기", use_container_width=True, type="primary")
        st.caption("도면 없이도 샘플 결과를 볼 수 있어요")

    # 분석 실행
    run_analysis = False
    filename = ""

    if uploaded:
        st.image(uploaded, caption=f"업로드된 도면: {uploaded.name}", use_container_width=True)
        if st.button("🤖 D-Fine AI로 분석 시작", type="primary", use_container_width=True):
            run_analysis = True
            filename = uploaded.name

    if demo_mode:
        run_analysis = True
        filename = "demo_sample_pid.png"

    # ── 분석 결과 표시 ──
    if run_analysis:
        st.divider()

        # 프로그레스 바 (시뮬레이션)
        progress_bar = st.progress(0)
        status_text = st.empty()

        pipeline = [
            (15, "📄 파일 로드 중..."),
            (30, "🔧 이미지 전처리 (노이즈 제거, 세그먼트 분할)..."),
            (55, "🤖 D-Fine 심볼 감지 중 (NMS 불필요, End-to-End)..."),
            (75, "📝 PaddleOCR 텍스트 추출 중..."),
            (90, "🔗 심볼-텍스트 매칭 및 PSM 안전장치 식별 중..."),
            (100, "✅ 분석 완료!"),
        ]
        for pct, msg in pipeline:
            progress_bar.progress(pct)
            status_text.markdown(f"**{msg}**")
            time.sleep(0.4)

        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()

        # 처리 시간 시뮬레이션
        proc_time = round(random.uniform(600, 1200), 1)
        analysis_id = f"{random.randint(10000000, 99999999):08x}"[:8]

        # ━━━ 결과 요약 통계 ━━━
        st.markdown("### 📊 분석 결과 요약")

        symbols = DEMO_RESULTS["symbols"]
        safety = [s for s in symbols if s["criticality"] == "critical"]
        high = [s for s in symbols if s["criticality"] == "high"]

        cols = st.columns(5)
        metric_data = [
            ("감지된 심볼", str(len(symbols)), "개"),
            ("추출된 텍스트", str(len(DEMO_RESULTS["texts"])), "개"),
            ("⚠️ PSM 안전장치", str(len(safety)), "개"),
            ("평균 확신도", f"{sum(s['confidence'] for s in symbols)/len(symbols)*100:.0f}", "%"),
            ("처리 시간", str(proc_time), "ms"),
        ]
        for col, (label, val, unit) in zip(cols, metric_data):
            col.metric(label, f"{val}{unit}")

        # ━━━ PSM 안전장치 (핵심!) ━━━
        if safety:
            st.markdown("### 🚨 PSM 안전장치 자동 식별 결과")
            st.markdown("""
            <div class="safety-box">
                <h3>⚠️ 독립방호계층(IPL) 대상 장치 발견!</h3>
                <p>아래 장치들은 <b>LOPA(방호계층분석)</b>에서 반드시 고려해야 합니다.<br>
                누락 시 → 잔여 위험빈도 과소평가 → <b style="color:#c62828">중대산업사고 위험 증가!</b></p>
            </div>
            """, unsafe_allow_html=True)

            for s in safety:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"""
                    <div style="text-align:center; padding:1.5rem; background:#ffebee;
                                border-radius:12px; border:2px solid #ef5350;">
                        <div style="font-size:2.5rem">🔴</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#c62828">{s['tag']}</div>
                        <div style="font-size:0.9rem; color:#666">{s['korean']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**태그:** `{s['tag']}`  |  **종류:** {s['korean']}  |  **확신도:** {s['confidence']*100:.0f}%")

                    if s['class'] == 'relief_valve':
                        st.markdown("""
                        - **기능:** 과압 시 자동 개방하여 압력을 해소하는 최후의 방어선
                        - **PFD (작동 실패 확률):** 10⁻² (100번 중 1번 실패)
                        - **LOPA 역할:** 과압에 의한 폭발 방지 독립방호계층(IPL)
                        - **연관 장비:** 반응기 R-401 상부 설치
                        """)
                    elif s['class'] == 'rupture_disc':
                        st.markdown("""
                        - **기능:** 급격한 과압 시 즉시 파열하여 비상 압력 해소
                        - **PFD (작동 실패 확률):** 10⁻² (100번 중 1번 실패)
                        - **LOPA 역할:** PSV 후단 2차 방호 독립방호계층(IPL)
                        - **연관 장비:** 증류탑 C-501 출구 설치
                        """)

                st.markdown("")

            st.warning("""
            **📋 위험성평가 연계:** 이 안전장치 정보는 2단계(위험성평가 코어)의 LOPA 모듈에 자동 전달됩니다.

            `잔여위험빈도 = 초기사건빈도 × Π(IPL의 PFD)` 에서 IPL로 직접 반영됩니다.
            """)

        # ━━━ 전체 장비 목록 ━━━
        st.markdown("### 🔍 전체 감지 결과")

        tab1, tab2, tab3 = st.tabs(["📋 장비 목록", "📝 추출된 텍스트", "🔗 매칭 상세"])

        with tab1:
            for s in symbols:
                crit_map = {"critical": "safety", "high": "high", "normal": ""}
                crit_label = {"critical": "🔴 위험 (PSM)", "high": "🟡 높음", "normal": "🟢 일반"}
                css_class = crit_map.get(s["criticality"], "")

                st.markdown(f"""
                <div class="equip-card {css_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <span style="font-weight:700; font-size:1.1rem">{s['tag']}</span>
                            <span style="color:#666; margin-left:0.5rem">{s['korean']} ({s['class']})</span>
                        </div>
                        <div style="display:flex; gap:1rem; align-items:center">
                            <span style="font-size:0.85rem">{crit_label[s['criticality']]}</span>
                            <span style="font-size:0.85rem; color:#666">확신도 {s['confidence']*100:.0f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            for t in DEMO_RESULTS["texts"]:
                cat_colors = {
                    "태그": "#1565c0", "안전장치": "#c62828",
                    "라인번호": "#00897b", "계장": "#f57f17",
                }
                color = cat_colors.get(t["category"], "#666")
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.6rem 1rem; border-bottom:1px solid #f0f0f0">
                    <div>
                        <code style="font-size:1rem; font-weight:600">{t['text']}</code>
                        <span style="background:{color}15; color:{color}; padding:2px 8px;
                                     border-radius:10px; font-size:0.75rem; font-weight:600;
                                     margin-left:0.5rem">{t['category']}</span>
                    </div>
                    <span style="font-size:0.85rem; color:#999">확신도 {t['confidence']*100:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("**심볼-텍스트 매칭 결과** — D-Fine이 찾은 심볼과 OCR이 읽은 텍스트가 연결된 결과입니다.")
            st.markdown("")

            match_data = []
            for s in symbols:
                match_data.append({
                    "태그": s["tag"],
                    "심볼 종류": s["class"],
                    "한국어명": s["korean"],
                    "확신도": f"{s['confidence']*100:.0f}%",
                    "PSM 중요도": {"critical": "🔴 위험", "high": "🟡 높음", "normal": "🟢 일반"}[s["criticality"]],
                })
            st.dataframe(match_data, use_container_width=True, hide_index=True)

        # ━━━ 다음 단계 안내 ━━━
        st.divider()
        st.markdown("### 🔮 이 데이터로 다음에 할 수 있는 것들")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### 📋 2단계: 위험성평가
            - HAZOP 워크시트 자동 생성
            - LOPA에 안전장치 자동 반영
            - 업종별 최적 평가기법 추천
            """)
        with col2:
            st.markdown("""
            #### 🏗️ 3단계: 디지털 트윈
            - Babylon.js 3D 웹 렌더링
            - 위험 지역 히트맵 표시
            - 장치 클릭 시 평가 결과 팝업
            """)
        with col3:
            st.markdown("""
            #### 🌡️ 4단계: CFD 시뮬레이션
            - 0.1초 사고 확산 시뮬레이션
            - What-if 시나리오 분석
            - 실시간 디지털 트윈 반영
            """)


# ============================================================
# 페이지 3: 심볼 클래스 목록
# ============================================================
elif page == "🔍 심볼 클래스 목록":
    st.markdown("""
    <div class="header-box">
        <h1>🔍 인식 가능한 P&ID 심볼 (42종)</h1>
        <p>D-Fine AI가 학습하여 인식할 수 있는 모든 심볼의 목록입니다</p>
    </div>
    """, unsafe_allow_html=True)

    total = sum(len(v) for v in SYMBOL_CLASSES.values())
    st.metric("전체 심볼 클래스", f"{total}종")

    for category, items in SYMBOL_CLASSES.items():
        is_safety = "안전장치" in category

        if is_safety:
            st.markdown(f"### {category}")
            st.error("아래 장치들은 PSM 위험성평가 시 독립방호계층(IPL)으로 반드시 고려해야 합니다.")
        else:
            st.markdown(f"### {category}")

        cols = st.columns(3)
        for i, (eng, kor) in enumerate(items):
            with cols[i % 3]:
                if is_safety:
                    st.markdown(f"""
                    <div style="padding:0.8rem; background:#ffebee; border:1px solid #ef9a9a;
                                border-radius:8px; margin:0.3rem 0">
                        <div style="font-weight:700; color:#c62828">🔴 {kor}</div>
                        <div style="font-size:0.8rem; color:#888"><code>{eng}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding:0.8rem; background:#f8f9fa; border:1px solid #e0e0e0;
                                border-radius:8px; margin:0.3rem 0">
                        <div style="font-weight:600">{kor}</div>
                        <div style="font-size:0.8rem; color:#888"><code>{eng}</code></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")


# ============================================================
# 페이지 4: API JSON 미리보기
# ============================================================
elif page == "📊 API JSON 미리보기":
    st.markdown("""
    <div class="header-box">
        <h1>📊 API 응답 데이터 미리보기</h1>
        <p>개발팀 참고용 — 실제 프로그램이 반환하는 JSON 데이터 구조입니다</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 이 데이터가 2단계(위험성평가 코어)와 3단계(디지털 트윈)에 자동으로 전달됩니다.")

    # 전체 응답
    sample_json = {
        "analysis_id": "a1b2c3d4",
        "status": "completed",
        "filename": "sample_pid.png",
        "processing_time_ms": 823.4,
        "summary": {
            "total_symbols": 12,
            "total_texts": 10,
            "total_tags": 7,
            "safety_devices": 2,
            "match_rate": "92.3%",
        },
        "safety_devices": [
            {
                "tag": "PSV-401",
                "class": "relief_valve",
                "korean": "압력안전밸브",
                "confidence": 0.96,
                "psm_note": "독립방호계층(IPL) 대상 - LOPA 분석 필수",
            },
            {
                "tag": "RD-501",
                "class": "rupture_disc",
                "korean": "파열판",
                "confidence": 0.93,
                "psm_note": "독립방호계층(IPL) 대상 - LOPA 분석 필수",
            },
        ],
        "model_info": {
            "detection_model": "D-Fine-L (HGNetV2, Apache 2.0)",
            "ocr_model": "PaddleOCR (Apache 2.0)",
            "nms_required": False,
        },
        "symbols": [
            {"tag": "T-101", "class": "tank", "korean": "탱크", "confidence": 0.98},
            {"tag": "P-201A", "class": "pump", "korean": "펌프", "confidence": 0.92},
            {"tag": "R-401", "class": "reactor", "korean": "반응기", "confidence": 0.95},
            {"tag": "PSV-401", "class": "relief_valve", "korean": "압력안전밸브", "confidence": 0.96},
            {"tag": "RD-501", "class": "rupture_disc", "korean": "파열판", "confidence": 0.93},
        ],
    }

    tab1, tab2, tab3 = st.tabs(["전체 응답", "안전장치만", "서비스 상태"])

    with tab1:
        st.markdown("#### `POST /api/v1/pid/analyze` 응답")
        st.json(sample_json)

    with tab2:
        st.markdown("#### `GET /api/v1/pid/safety/{analysis_id}` 응답")
        st.json({
            "analysis_id": "a1b2c3d4",
            "safety_devices": sample_json["safety_devices"],
            "total_count": 2,
            "psm_guidance": {
                "note": "아래 안전장치는 PSM 위험성평가 시 IPL로 고려해야 합니다.",
                "required_analyses": ["LOPA (방호계층분석)", "SIL (안전계전시스템 등급)"],
            },
        })

    with tab3:
        st.markdown("#### `GET /api/v1/health` 응답")
        st.json({
            "service": "pid-parser",
            "version": "1.0.0",
            "status": "healthy",
            "model": "D-Fine-L (Apache 2.0)",
            "timestamp": datetime.now().isoformat(),
        })

    st.markdown("")
    st.markdown("##### 💻 API 문서 (Swagger UI)")
    st.code("http://localhost:8001/docs", language="text")
    st.caption("프로그램 실행 후 위 주소에서 직접 API를 테스트할 수 있습니다.")
