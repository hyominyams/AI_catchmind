# app.py
# -*- coding: utf-8 -*-

import os, io, csv, random, re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from difflib import SequenceMatcher

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI 스케치 퀴즈", page_icon="🎨", layout="wide")

# ---------------- Gemini ----------------
try:
    import google.generativeai as genai
except Exception:
    genai = None


def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if genai is None or not api_key:
        st.session_state["ai_status"] = "unavailable"
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        st.session_state["ai_status"] = "ok"
        return model
    except Exception as e:
        st.session_state["ai_status"] = "error"
        st.session_state["ai_error_msg"] = str(e)
        return None


# ---------------- Utils ----------------
SPACE_PTN = re.compile(r"\s+")
PUNCT_PTN = re.compile(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+")


def norm_ko(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    s = SPACE_PTN.sub("", s)
    s = PUNCT_PTN.sub("", s)
    return s


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_ko(a), norm_ko(b)).ratio()


def pil_from_canvas(image_data: Optional[np.ndarray]) -> Optional[Image.Image]:
    if image_data is None:
        return None
    arr = image_data.astype("uint8")  # RGBA
    img = Image.fromarray(arr, mode="RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(bg, img)
    return composed.convert("RGB")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def blank_png_bytes(w: int = 640, h: int = 360, color=(255, 255, 255)) -> bytes:
    return image_to_png_bytes(Image.new("RGB", (w, h), color))


# ---------------- 카테고리 ----------------
CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]

# ---------------- 프롬프트 ----------------
PROMPT_GUESS_FREE = """
너는 스케치 이미지를 보고 정답을 추측하는 역할을 맡은 "카테고리 정답 감별사"다.  
너의 핵심 임무는, 오직 지정된 **카테고리 안에 속한 단어만** 정답으로 판단하는 것이다.  
스케치가 모호하거나 여러 해석이 가능하더라도 **카테고리 안에 속한 단어 중에서 한 단어를 출력해야 한다.**

【입력】
- 카테고리: {{카테고리명}}
- 스케치 이미지: 1장 (형태, 윤곽, 상징 요소, 색상은 판단의 보조 힌트일 뿐이다)
- (선택) 참조 이미지: 라벨별 스케치 예시가 함께 주어질 수 있다

【최상위 판단 규칙 — 반드시 지켜야 함】
1. 반드시 주어진 "카테고리" 안에 속한 보통명사만 정답으로 판단한다.  
2. 카테고리에 속하지 않는 단어는 어떤 경우에도 절대 출력하지 않는다.  

【판단 절차】
1. 먼저, 해당 카테고리에 속하는 일반적인 보통명사(상위/하위 포함) 목록을 떠올린다.
2. 스케치 이미지와 (있다면) 참조 이미지를 보조 힌트로 활용해 후보를 좁힌다.
3. 단 하나의 정답만 출력한다.

【출력 규칙 — 매우 중요】
- 반드시 **한국어 단어 1개**만 출력한다.
- 설명, 문장, 영어, 숫자, 기호, 이모지, 접두사/접미사, 공백, 마침표, 따옴표는 절대 포함하지 않는다.
- 오직 ① 카테고리 내부 단어 1개만 허용된다.

예시:
- 카테고리: 동물 → 가능: 고양이, 강아지 / 불가: 사과, 자동차
- 카테고리: 과일 → 가능: 바나나, 사과 / 불가: 토끼, 자전거
- 카테고리: 교통수단 → 가능: 자전거, 버스 / 불가: 고양이, 포도

정답 판단을 시작하자.  
숨을 깊이 들이쉬고, 단계적으로 차분히 생각해보자.  
우리가 올바른 답을 찾기 위해 꼭 필요한 과정이다.
"""


# ---------------- 세션 ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("category", CATEGORIES[0])
    ss.setdefault("max_rounds", 5)

    ss.setdefault("game_started", False)
    ss.setdefault("score", 0)
    ss.setdefault("round", 0)                   # 1부터 진행
    ss.setdefault("canvas_key", "canvas_0")     # 라운드별 캔버스 키

    ss.setdefault("targets_pool", [])           # [{word:str, aliases:[...]}]
    ss.setdefault("pool_index", 0)
    ss.setdefault("target", None)               # {"word": ..., "aliases":[...]}

    ss.setdefault("submitted", False)
    ss.setdefault("last_guess", "")
    ss.setdefault("round_end_time", None)

    ss.setdefault("ai_status", "unknown")
    ss.setdefault("ai_error_msg", "")

    ss.setdefault("label_sets", [])

    # AI 대기 & 스냅샷
    ss.setdefault("ai_pending", False)
    ss.setdefault("last_canvas_png", None)

    # 결과 페이지용 히스토리
    ss.setdefault("history", [])

    # 팔레트 상태
    ss.setdefault("stroke_color", "#000000")  # 기본 검정


# ---------------- keyword.csv ----------------
@st.cache_data(show_spinner=False)
def load_keywords_from_csv(path: str = "keyword.csv") -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
    """
    CSV 형식(권장): category,keyword,aliases
    - aliases는 '|' 로 구분 (없어도 됨)
    반환: {카테고리: [ {"word": str, "aliases": [str,...]} , ... ]}
    """
    bank: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # 헤더 판단
            has_header = False
            idx_cat, idx_kw, idx_alias = 0, 1, None
            if header:
                h = [c.strip().lower() for c in header]
                if "category" in h and "keyword" in h:
                    has_header = True
                    idx_cat = h.index("category")
                    idx_kw = h.index("keyword")
                    idx_alias = h.index("aliases") if "aliases" in h else None
            if not has_header:
                f.seek(0)
                reader = csv.reader(f)
                idx_cat, idx_kw, idx_alias = 0, 1, None

            for row in reader:
                if len(row) <= idx_kw:
                    continue
                cat = (row[idx_cat] if idx_cat is not None else "").strip()
                word = (row[idx_kw] if idx_kw is not None else "").strip()
                if not cat or not word:
                    continue
                aliases: List[str] = []
                if idx_alias is not None and len(row) > idx_alias and row[idx_alias]:
                    aliases = [a.strip() for a in row[idx_alias].split("|") if a.strip()]
                bank.setdefault(cat, [])
                entry = {"word": word, "aliases": aliases}
                # 중복 방지
                if all(e["word"] != word for e in bank[cat]):
                    bank[cat].append(entry)

        return bank, None
    except FileNotFoundError:
        return {}, "keyword.csv 파일이 없습니다. 프로젝트 루트에 배치해 주세요."
    except Exception as e:
        return {}, f"keyword.csv 로딩 오류: {e}"


def build_targets_pool(category: str, bank: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    candidates = (bank or {}).get(category, []).copy()
    random.shuffle(candidates)
    return candidates


# ---------------- 라벨 참조 파트 ----------------
def build_reference_parts(label_sets: List[Dict[str, Any]]) -> List[Any]:
    parts: List[Any] = []
    for item in label_sets:
        name = item.get("name")
        imgs: List[bytes] = item.get("images") or []
        if not name or not imgs:
            continue
        parts.append(f"참조: {name}")
        for b in imgs[:10]:
            parts.append({"mime_type": "image/png", "data": b})
    return parts


# ---------------- Gemini 호출 ----------------
def guess_from_image(img: Optional[Image.Image], category: str) -> str:
    if img is None:
        return ""
    model = get_gemini_model()
    if model is None:
        return ""  # 상태 배너로 안내

    prompt = PROMPT_GUESS_FREE.format(category=category)
    try:
        parts = [prompt, {"mime_type": "image/png", "data": image_to_png_bytes(img)}]
        # (선택) 라벨링 참조 이미지 전달
        parts.extend(build_reference_parts(st.session_state.get("label_sets", [])))
        resp = model.generate_content(parts)
        text = (getattr(resp, "text", "") or "").strip()
        text = text.replace("\n", " ").replace("\r", " ").strip().strip('"\'')
        if not text:
            return "AI가 답을 찾지 못했습니다 😢"
        # 단어 1개만 강제
        if " " in text:
            text = text.split()[0]
        return text
    except Exception as e:
        st.session_state["ai_status"] = "error"
        st.session_state["ai_error_msg"] = str(e)
        return "AI가 답을 찾지 못했습니다 😢"


# ---------------- 정답 판정 ----------------
def is_correct(guess: str, target: Dict[str, Any], threshold: float = 0.8) -> bool:
    if not target:
        return False
    candidates = [target.get("word", "")]
    candidates += target.get("aliases", []) or []
    for c in candidates:
        if sim(guess, c) >= threshold:
            return True
    return False


# ---------------- 게임 플로우 ----------------
def start_game(keyword_bank: Dict[str, List[Dict[str, Any]]]):
    ss = st.session_state
    pool = build_targets_pool(ss["category"], keyword_bank)
    if len(pool) == 0:
        st.warning("해당 카테고리에서 제시어를 찾지 못했습니다. keyword.csv를 확인하세요.")
        return

    ss["history"] = []
    ss["targets_pool"] = pool
    ss["pool_index"] = 0
    ss["game_started"] = True
    ss["score"] = 0
    ss["round"] = 0
    ss["last_canvas_png"] = None
    ss["ai_pending"] = False
    next_round()  # round=1부터 시작
    ss["page"] = "Game"
    st.rerun()


def pick_next_target() -> Optional[Dict[str, Any]]:
    ss = st.session_state
    if ss["pool_index"] >= len(ss["targets_pool"]):
        return None
    t = ss["targets_pool"][ss["pool_index"]]
    ss["pool_index"] += 1
    return t


def next_round():
    ss = st.session_state
    ss["round"] += 1
    ss["submitted"] = False
    ss["last_guess"] = ""
    ss["ai_pending"] = False
    ss["last_canvas_png"] = None

    ss["target"] = pick_next_target()
    if ss["target"] is None:
        st.warning("더 이상 출제할 제시어가 없습니다. Home으로 돌아가 새로 시작하세요.")
        ss["game_started"] = False
        ss["page"] = "Home"
        return

    # 라운드별 캔버스 키 → 새 캔버스로 렌더
    ss["canvas_key"] = f"canvas_{ss['round']}"
    ss["round_end_time"] = datetime.utcnow() + timedelta(seconds=60)


def end_game_if_needed():
    ss = st.session_state
    if ss["round"] >= ss["max_rounds"] and ss["submitted"]:
        ss["game_started"] = False
        ss["page"] = "Results"


def submit_answer_with_image(img_pil: Optional[Image.Image]):
    """이미지를 받아 Gemini 호출 + 판정 + 히스토리 기록."""
    ss = st.session_state
    if ss["submitted"]:
        return
    ss["submitted"] = True

    guess = guess_from_image(img_pil, ss["category"]) if img_pil else "AI가 답을 찾지 못했습니다 😢"
    ss["last_guess"] = guess

    correct = is_correct(guess, ss.get("target"))
    if correct:
        ss["score"] += 1

    # 제출 당시 이미지
    if img_pil is not None:
        img_bytes = image_to_png_bytes(img_pil)
    else:
        img_bytes = ss.get("last_canvas_png") or blank_png_bytes()

    ss["history"].append({
        "round": ss["round"],
        "word": (ss.get("target") or {}).get("word", ""),
        "guess": guess,
        "correct": correct,
        "image": img_bytes,
    })


def pass_question():
    if not st.session_state.get("game_started"):
        return
    st.session_state["submitted"] = True
    if st.session_state["round"] >= st.session_state["max_rounds"]:
        st.session_state["game_started"] = False
        st.session_state["page"] = "Results"
    else:
        next_round()
    st.rerun()


def trigger_submit():
    """수동 제출 트리거(자동제출 없음)."""
    ss = st.session_state
    if ss.get("submitted") or ss.get("ai_pending"):
        return
    if ss.get("last_canvas_png") is None:
        ss["last_canvas_png"] = blank_png_bytes()
    ss["ai_pending"] = True
    st.rerun()


# ---------------- 라벨링(선택) ----------------
def add_label():
    st.session_state["label_sets"].append({"name": "", "images": []})


def remove_label(idx: int):
    labels = st.session_state.get("label_sets", [])
    if 0 <= idx < len(labels):
        labels.pop(idx)


def refresh_label_from_inputs(idx: int):
    key_name = f"label_name_{idx}"
    key_files = f"label_files_{idx}"
    name_val = st.session_state.get(key_name, "").strip()
    files_val = st.session_state.get(key_files) or []
    st.session_state["label_sets"][idx]["name"] = name_val

    imgs: List[bytes] = []
    for f in files_val[:10]:
        try:
            img = Image.open(f).convert("RGB")
            imgs.append(image_to_png_bytes(img))
        except Exception:
            pass
    st.session_state["label_sets"][idx]["images"] = imgs


# ---------------- UI ----------------
init_state()

st.title("🎨 AI 스케치 퀴즈")

# AI 상태 배너
ai_status = st.session_state.get("ai_status", "unknown")
if ai_status == "unavailable":
    st.warning("⚠️ Gemini API 키가 설정되지 않아 AI를 호출할 수 없습니다. (GEMINI_API_KEY 필요)")
elif ai_status == "error":
    st.error(f"❌ Gemini 호출 오류: {st.session_state.get('ai_error_msg', '')}")
elif ai_status == "ok":
    st.success("✅ Gemini 연결 정상")

# 키워드 CSV
KEYWORD_BANK, CSV_ERR = load_keywords_from_csv("keyword.csv")
if CSV_ERR:
    st.error(f"❌ {CSV_ERR}")

page = st.session_state.get("page", "Home")

# ========================= HOME =========================
if page == "Home":
    st.subheader("카테고리 선택")
    st.radio("카테고리", CATEGORIES, key="category", horizontal=True)
    st.number_input("문제 수", min_value=1, max_value=20, step=1, key="max_rounds")
    st.button("게임 시작", type="primary", on_click=start_game, args=(KEYWORD_BANK,))

    st.markdown("---")
    st.subheader("라벨링(선택) · 정확도 보조자료")
    st.caption("필수 아님: 라벨과 참조 이미지를 추가하면 판정 시 참고합니다.")
    for i, item in enumerate(st.session_state["label_sets"]):
        with st.container(border=True):
            cols = st.columns([6, 1])
            with cols[0]:
                st.text_input("라벨 이름", value=item.get("name", ""), key=f"label_name_{i}", placeholder="예: 사과")
            with cols[1]:
                st.button("🗑️ 삭제", key=f"delete_label_{i}", on_click=remove_label, args=(i,), use_container_width=True)

            st.file_uploader(
                "참조 이미지(최대 10장)",
                key=f"label_files_{i}",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )
            refresh_label_from_inputs(i)
    st.button("+ 라벨 추가", on_click=add_label)

# ========================= GAME =========================
elif page == "Game":

    # ---- 1) AI 처리(토글 없는 안내) ----
    if st.session_state.get("ai_pending") and not st.session_state.get("submitted"):
        st.info("🤖 AI가 생각중입니다… 잠시만요.")
        # 스냅샷에서 이미지 복원 후 즉시 판정
        img_bytes = st.session_state.get("last_canvas_png") or blank_png_bytes()
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img_pil = None
        submit_answer_with_image(img_pil)
        st.session_state["ai_pending"] = False
        st.rerun()

    # ---- 2) 남은 시간 & 만료 여부 ----
    expired = False
    remain = 0
    if st.session_state.get("round_end_time"):
        remain = max(0, int((st.session_state["round_end_time"] - datetime.utcnow()).total_seconds()))
        expired = remain <= 0

    # ---- 3) 상태 행(라운드/점수/JS 타이머) ----
    status_cols = st.columns([1, 1, 2])
    with status_cols[0]:
        st.metric("라운드", f"{st.session_state['round']}/{st.session_state['max_rounds']}")
    with status_cols[1]:
        st.metric("점수", f"{st.session_state['score']}")
    with status_cols[2]:
        # 제출 전일 때만 카운트다운 표시
        if st.session_state.get("game_started") and not st.session_state.get("submitted"):
            if not expired:
                end_dt = st.session_state["round_end_time"]
                timer_html = f"""
                <div style="display:flex;justify-content:flex-end;align-items:center;">
                  <div id="timer" style="font-size:48px;font-weight:700;line-height:64px;margin-top:-4px;padding-bottom:6px;">
                    {remain}
                  </div>
                </div>
                <script>
                  const endTs = {int(end_dt.timestamp()*1000)};
                  const el = document.getElementById('timer');
                  let reloaded = false;
                  function reloadOnce(){{
                    if (reloaded) return;
                    reloaded = true;
                    clearInterval(tId);
                    setTimeout(() => window.location.reload(), 50);
                  }}
                  function tick(){{
                    const left = Math.max(0, Math.floor((endTs - Date.now())/1000));
                    if (el) el.textContent = left;
                    if (left <= 0) reloadOnce();   // 0초가 되면 한 번만 리로드 → expired 전환
                  }}
                  const tId = setInterval(tick, 1000);
                  tick();
                </script>
                """
                st.components.v1.html(timer_html, height=88)
            else:
                # 만료 후에는 0을 고정 표기
                st.markdown(
                    f"<div style='text-align:right;font-size:48px;font-weight:700;line-height:64px;'>0</div>",
                    unsafe_allow_html=True,
                )

    # ---- 4) 메인 UI ----
    if st.session_state.get("game_started"):
        st.subheader(f"제시어: {st.session_state['target']['word']} (그려보세요!)")

        # 제출 전 상태
        if not st.session_state.get("submitted"):
            if not expired:
                # 팔레트(버튼형) – 사용 가능 상태에서만 노출
                palette = {
                    "⚫ 검정": "#555555",
                    "🔴 빨강": "#FF4C4C",
                    "🟠 오렌지": "#FFA500",
                    "🟡 노랑": "#FDFFB6",
                    "🟢 초록": "#32CD32",
                    "🔵 파랑": "#3399FF",
                    "🟣 보라": "#BDB2FF",
                    "🌸 분홍": "#FFB5E8",
                    "🌊 하늘": "#00CED1",
                    "🟤 갈색": "#8B4513",
                    "🍑 살구": "#FFDAB9",
                    "⚪ 회색": "#808080",
                }
                pcols = st.columns(len(palette))
                for i, (name, code) in enumerate(palette.items()):
                    if pcols[i].button(name, use_container_width=True):
                        st.session_state["stroke_color"] = code

                # 캔버스
                canvas_res = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=6,
                    stroke_color=st.session_state["stroke_color"],
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=360,
                    width=640,
                    drawing_mode="freedraw",
                    key=st.session_state["canvas_key"],
                )
                canvas_img = pil_from_canvas(canvas_res.image_data) if canvas_res is not None else None
                # 최신 스냅샷 유지
                if canvas_img is not None:
                    st.session_state["last_canvas_png"] = image_to_png_bytes(canvas_img)
                elif st.session_state.get("last_canvas_png") is None:
                    st.session_state["last_canvas_png"] = blank_png_bytes()

                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("제출", type="primary", use_container_width=True):
                        trigger_submit()
                with cols[1]:
                    if st.button("패스", use_container_width=True):
                        pass_question()
                with cols[2]:
                    # 제출 전에는 다음 문제 버튼 비활성화
                    st.button("다음 문제", use_container_width=True, disabled=True)

            else:
                # ⏰ 만료: 캔버스/팔레트 잠금 + 제출 유도 메시지
                img_preview = st.session_state.get("last_canvas_png") or blank_png_bytes()
                st.image(img_preview, caption="⏰ 시간이 끝났습니다. 그림은 잠겼어요.", width=640)
                st.warning("시간이 종료되었습니다. **제출하세요** 버튼을 눌러 결과를 확인해주세요.")

                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("제출", type="primary", use_container_width=True):
                        trigger_submit()
                with cols[1]:
                    if st.button("패스", use_container_width=True):
                        pass_question()
                with cols[2]:
                    st.button("다음 문제", use_container_width=True, disabled=True)

        # 제출 후 결과 패널
        else:
            st.markdown("---")
            st.subheader("결과")
            img_preview = st.session_state.get("last_canvas_png") or blank_png_bytes()
            st.image(img_preview, caption="제출한 그림", use_column_width=False, width=320)

            cols2 = st.columns(3)
            with cols2[0]:
                st.caption("AI 추측")
                guess_text = st.session_state["last_guess"] or "AI가 답을 찾지 못했습니다 😢"
                st.success(guess_text)
            with cols2[1]:
                st.caption("정답 제시어")
                st.info(st.session_state["target"]["word"] if st.session_state["target"] else "(없음)")
            with cols2[2]:
                verdict = (
                    "✅ 성공"
                    if is_correct(st.session_state["last_guess"], st.session_state.get("target"))
                    else "❌ 실패"
                )
                st.metric("판정", verdict)

            # 라운드 종료 시 이동 버튼
            cols_btns = st.columns([1, 1, 1])
            with cols_btns[0]:
                if st.button("다음 문제", type="primary", use_container_width=True):
                    end_game_if_needed()
                    if not st.session_state["game_started"]:
                        st.session_state["page"] = "Results"; st.rerun()
                    else:
                        next_round(); st.rerun()
            with cols_btns[1]:
                if st.button("결과 페이지", use_container_width=True):
                    st.session_state["page"] = "Results"
                    st.session_state["game_started"] = False
                    st.rerun()
            with cols_btns[2]:
                if st.button("홈으로", use_container_width=True):
                    st.session_state["page"] = "Home"
                    st.session_state["game_started"] = False
                    st.rerun()
    else:
        st.info("홈에서 카테고리를 고르고 '게임 시작'을 눌러주세요.")

# ========================= RESULTS =========================
elif page == "Results":
    st.header("📊 최종 결과")
    if not st.session_state.get("history"):
        st.info("표시할 결과가 없습니다. 홈에서 새 게임을 시작해 보세요.")
    else:
        total = len(st.session_state["history"])
        correct = sum(1 for h in st.session_state["history"] if h["correct"])
        st.metric("총 점수", f"{correct}/{total}")

        st.markdown("---")
        for h in st.session_state["history"]:
            with st.container(border=True):
                cols = st.columns([2, 2, 3])
                with cols[0]:
                    st.image(h["image"], caption=f"Round {h['round']}", use_column_width=True)
                with cols[1]:
                    st.write(f"**정답 제시어:** {h['word']}")
                    st.write(f"**AI 추측:** {h['guess']}")
                with cols[2]:
                    st.write("**판정:** " + ("✅ 성공" if h["correct"] else "❌ 실패"))

        st.markdown("---")
        c = st.columns(2)
        with c[0]:
            if st.button("다시 시작", type="primary", use_container_width=True):
                st.session_state["page"] = "Home"
                st.session_state["game_started"] = False
                st.rerun()
        with c[1]:
            if st.button("홈으로", use_container_width=True):
                st.session_state["page"] = "Home"
                st.session_state["game_started"] = False
                st.rerun()

st.markdown("---")
st.caption("제시어는 keyword.csv에서 무작위로 선정됩니다. AI에는 카테고리 외 단어 목록을 절대 제공하지 않습니다. (정답 인정: 문자열 유사도 0.8 이상, aliases 지원)")
