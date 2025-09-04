# app.py
# -*- coding: utf-8 -*-

import os
import io
import csv
import random
import re
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
[역할] 당신은 지금부터 학생들의 스케치를 보고 "단어"를 맞추는 탐정입니다. 주어진 스케치와 카테고리 정보를 결합하여 한국어 단어를 출력하세요. 
[지시]
- 카테고리: {category}
- 이미지를 보고 한국어 단어 1개만 출력하세요. (설명/문장/기호/영문 금지)
- 거친 선·부정확한 비율 허용. 윤곽/상징 요소(바퀴, 날개 등)를 중시하여 판단하세요.
- **주어진 카테고리 정보에 기반해서 단어를 출력하세요. 카테고리를 벗어나는 단어를 출력하지 않습니다.**
"""


# ---------------- 세션 ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("category", CATEGORIES[0])
    ss.setdefault("max_rounds", 5)

    ss.setdefault("game_started", False)
    ss.setdefault("score", 0)
    ss.setdefault("round", 0)
    ss.setdefault("canvas_key", "canvas_0")

    ss.setdefault("targets_pool", [])
    ss.setdefault("pool_index", 0)
    ss.setdefault("target", None)

    ss.setdefault("submitted", False)
    ss.setdefault("last_guess", "")
    ss.setdefault("round_end_time", None)

    ss.setdefault("ai_status", "unknown")
    ss.setdefault("ai_error_msg", "")

    ss.setdefault("label_sets", [])

    ss.setdefault("ai_pending", False)
    ss.setdefault("last_canvas_png", None)

    ss.setdefault("history", [])


# ---------------- keyword.csv ----------------
@st.cache_data(show_spinner=False)
def load_keywords_from_csv(path: str = "keyword.csv") -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
    bank: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
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


# ---------------- Gemini 호출 ----------------
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


def guess_from_image(img: Optional[Image.Image], category: str) -> str:
    if img is None:
        return ""
    model = get_gemini_model()
    if model is None:
        return ""
    prompt = PROMPT_GUESS_FREE.format(category=category)
    try:
        parts = [prompt, {"mime_type": "image/png", "data": image_to_png_bytes(img)}]
        parts.extend(build_reference_parts(st.session_state.get("label_sets", [])))
        resp = model.generate_content(parts)
        text = (getattr(resp, "text", "") or "").strip()
        text = text.replace("\n", " ").replace("\r", " ").strip().strip('"\'')
        if not text:
            return "AI가 답을 찾지 못했습니다 😢"
        if " " in text:
            text = text.split()[0]
        return text
    except Exception as e:
        st.session_state["ai_status"] = "error"
        st.session_state["ai_error_msg"] = str(e)
        return "AI가 답을 찾지 못했습니다 😢"


# ---------------- 판정 ----------------
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
    next_round()
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
    ss["canvas_key"] = f"canvas_{ss['round']}"
    ss["round_end_time"] = datetime.utcnow() + timedelta(seconds=60)


def end_game_if_needed():
    ss = st.session_state
    if ss["round"] >= ss["max_rounds"] and ss["submitted"]:
        ss["game_started"] = False
        ss["page"] = "Results"


def submit_answer_with_image(img_pil: Optional[Image.Image]):
    ss = st.session_state
    if ss["submitted"]:
        return
    ss["submitted"] = True
    guess = guess_from_image(img_pil, ss["category"]) if img_pil else "AI가 답을 찾지 못했습니다 😢"
    ss["last_guess"] = guess
    correct = is_correct(guess, ss.get("target"))
    if correct:
        ss["score"] += 1
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

# AI 상태
ai_status = st.session_state.get("ai_status", "unknown")
if ai_status == "unavailable":
    st.warning("⚠️ Gemini API 키가 설정되지 않아 AI를 호출할 수 없습니다.")
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
    st.number_input("문제 수", 1, 20, key="max_rounds")
    st.button("게임 시작", type="primary", on_click=start_game, args=(KEYWORD_BANK,))
    st.markdown("---")
    st.subheader("라벨링(선택) · 정확도 보조자료")
    for i, item in enumerate(st.session_state["label_sets"]):
        with st.container(border=True):
            cols = st.columns([6, 1])
            with cols[0]:
                st.text_input("라벨 이름", value=item.get("name", ""), key=f"label_name_{i}")
            with cols[1]:
                st.button("🗑️ 삭제", key=f"delete_label_{i}", on_click=remove_label, args=(i,))
            st.file_uploader("참조 이미지", key=f"label_files_{i}", type=["png","jpg","jpeg"], accept_multiple_files=True)
            refresh_label_from_inputs(i)
    st.button("+ 라벨 추가", on_click=add_label)

# ========================= GAME =========================
elif page == "Game":
    if st.session_state.get("ai_pending") and not st.session_state.get("submitted"):
        st.info("🤖 AI가 생각중입니다… 잠시만요.")
        img_bytes = st.session_state.get("last_canvas_png") or blank_png_bytes()
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img_pil = None
        submit_answer_with_image(img_pil)
        st.session_state["ai_pending"] = False
        st.rerun()

    expired, remain = False, 0
    if st.session_state.get("round_end_time"):
        remain = max(0, int((st.session_state["round_end_time"] - datetime.utcnow()).total_seconds()))
        expired = remain <= 0

    status_cols = st.columns([1, 1, 2])
    with status_cols[0]:
        st.metric("라운드", f"{st.session_state['round']}/{st.session_state['max_rounds']}")
    with status_cols[1]:
        st.metric("점수", f"{st.session_state['score']}")
    with status_cols[2]:
        if st.session_state.get("game_started") and not st.session_state.get("submitted"):
            if not expired:
                end_dt = st.session_state["round_end_time"]
                timer_html = f"""
                <div style="text-align:right;font-size:48px;font-weight:700;">{remain}</div>
                <script>
                  const endTs = {int(end_dt.timestamp()*1000)};
                  const el=document.querySelector('div[style*="font-size:48px"]');
                  function tick(){{
                    const left=Math.max(0,Math.floor((endTs-Date.now())/1000));
                    if(el) el.textContent=left;
                    if(left<=0) window.location.reload();
                  }}
                  setInterval(tick,1000);tick();
                </script>
                """
                st.components.v1.html(timer_html, height=64)

    if st.session_state.get("game_started"):
        st.subheader(f"제시어: {st.session_state['target']['word']} (그려보세요!)")
        if not st.session_state.get("submitted"):
            if not expired:
                # 🎨 컬러 & 굵기 선택기
                draw_color = st.color_picker("펜 색상 선택", "#000000", key=f"color_{st.session_state['round']}")
                stroke_w = st.slider("펜 굵기", 2, 20, 6, key=f"stroke_{st.session_state['round']}")
                canvas_res = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=stroke_w,
                    stroke_color=draw_color,
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=360,
                    width=640,
                    drawing_mode="freedraw",
                    key=st.session_state["canvas_key"],
                )
                canvas_img = pil_from_canvas(canvas_res.image_data) if canvas_res is not None else None
                if canvas_img is not None:
                    st.session_state["last_canvas_png"] = image_to_png_bytes(canvas_img)
                elif st.session_state.get("last_canvas_png") is None:
                    st.session_state["last_canvas_png"] = blank_png_bytes()
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("제출", type="primary"):
                        trigger_submit()
                with cols[1]:
                    if st.button("패스"):
                        pass_question()
                with cols[2]:
                    st.button("다음 문제", disabled=True)
            else:
                img_preview = st.session_state.get("last_canvas_png") or blank_png_bytes()
                st.image(img_preview, caption="⏰ 시간이 끝났습니다.", width=640)
                st.warning("시간이 종료되었습니다. **제출하세요** 버튼을 눌러 결과를 확인해주세요.")
                cols = st.columns([1,1,1])
                with cols[0]:
                    if st.button("제출", type="primary"):
                        trigger_submit()
                with cols[1]:
                    if st.button("패스"):
                        pass_question()
                with cols[2]:
                    st.button("다음 문제", disabled=True)
        else:
            st.markdown("---")
            st.subheader("결과")
            img_preview = st.session_state.get("last_canvas_png") or blank_png_bytes()
            st.image(img_preview, caption="제출한 그림", width=320)
            cols2 = st.columns(3)
            with cols2[0]:
                st.caption("AI 추측")
                st.success(st.session_state["last_guess"])
            with cols2[1]:
                st.caption("정답 제시어")
                st.info(st.session_state["target"]["word"])
            with cols2[2]:
                verdict = "✅ 성공" if is_correct(st.session_state["last_guess"], st.session_state.get("target")) else "❌ 실패"
                st.metric("판정", verdict)
            cols_btns = st.columns([1,1,1])
            with cols_btns[0]:
                if st.button("다음 문제"):
                    end_game_if_needed()
                    if not st.session_state["game_started"]:
                        st.session_state["page"] = "Results"; st.rerun()
                    else:
                        next_round(); st.rerun()
            with cols_btns[1]:
                if st.button("결과 페이지"):
                    st.session_state["page"] = "Results"; st.session_state["game_started"]=False; st.rerun()
            with cols_btns[2]:
                if st.button("홈으로"):
                    st.session_state["page"]="Home"; st.session_state["game_started"]=False; st.rerun()

# ========================= RESULTS =========================
elif page == "Results":
    st.header("📊 최종 결과")
    if not st.session_state.get("history"):
        st.info("표시할 결과가 없습니다.")
    else:
        total=len(st.session_state["history"])
        correct=sum(1 for h in st.session_state["history"] if h["correct"])
        st.metric("총 점수", f"{correct}/{total}")
        for h in st.session_state["history"]:
            with st.container(border=True):
                cols=st.columns([2,2,3])
                with cols[0]:
                    st.image(h["image"], caption=f"Round {h['round']}")
                with cols[1]:
                    st.write(f"**정답:** {h['word']}")
                    st.write(f"**AI 추측:** {h['guess']}")
                with cols[2]:
                    st.write("**판정:** " + ("✅ 성공" if h["correct"] else "❌ 실패"))
        c=st.columns(2)
        with c[0]:
            if st.button("다시 시작", type="primary"):
                st.session_state["page"]="Home"; st.session_state["game_started"]=False; st.rerun()
        with c[1]:
            if st.button("홈으로"):
                st.session_state["page"]="Home"; st.session_state["game_started"]=False; st.rerun()

st.markdown("---")
st.caption("제시어는 keyword.csv에서 무작위로 선정됩니다. AI에는 카테고리 외 단어 목록을 절대 제공하지 않습니다.")
