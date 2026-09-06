import os
import re
import base64
import unicodedata
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(
    page_title="Apex Motors | Smart Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تحميل صورة الخلفية وتحويلها إلى Base64
@st.cache_data(show_spinner=False)
def get_image_data(image_path="background_car.png", mime="image/png"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    return ""

BG_IMAGE = get_image_data("background_car.png", "image/png")

# ==============================================================================
# CSS: شريط بحث مدمج بزر الكاميرا بدقة مع خلفية الصفحة
# ==============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
    --red: #ff3434;
    --white: #f7f7f7;
    --muted: #a9adb5;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: #050607 !important;
}}

.stApp {{
    min-height: 100vh;
    background: transparent !important;
    color: var(--white);
    font-family: 'Inter', sans-serif;
}}

/* خلفية الصفحة كاملة */
.background-car {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image: url("{BG_IMAGE}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: .82;
}}

.background-car:before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(0,0,0,.72) 0%, rgba(0,0,0,.34) 48%, rgba(0,0,0,.58) 100%),
        linear-gradient(180deg, rgba(0,0,0,.30) 0%, rgba(0,0,0,.08) 46%, rgba(0,0,0,.72) 100%);
}}

.background-car:after {{
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 38%, rgba(255,52,52,.06), transparent 34%);
}}

.main .block-container {{
    position: relative;
    z-index: 2;
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

/* Hero Section */
.hero-box {{
    position: relative;
    min-height: 230px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 0 auto;
    background: transparent;
    border: 0;
}}

.hero-box:before {{
    content: "";
    position: absolute;
    width: 520px;
    height: 520px;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -48%);
    background: radial-gradient(circle, rgba(0,0,0,.42) 0%, rgba(0,0,0,.20) 42%, transparent 72%);
    z-index: 0;
}}

.hero-content {{
    position: relative;
    z-index: 2;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.hero-kicker {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px;
    background: rgba(0,0,0,.4);
    color: #e9e9eb;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: 2px;
    backdrop-filter: blur(9px);
}}

.hero-title {{
    margin: 0;
    color: #fff;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    line-height: 1;
    font-weight: 800;
    letter-spacing: -2px;
    text-shadow: 0 10px 40px rgba(0,0,0,.9);
}}

.hero-title span {{
    color: var(--red);
}}

.hero-subtitle {{
    color: #e1e3e7;
    font-size: .95rem;
    max-width: 620px;
    margin: 12px auto 0;
    line-height: 1.55;
    text-shadow: 0 3px 18px #000;
}}

.hero-line {{
    width: 56px;
    height: 3px;
    background: var(--red);
    border-radius: 99px;
    margin: 12px auto 0;
    box-shadow: 0 0 22px rgba(255,52,52,.45);
}}

/* Features */
.feature-row {{
    position: relative;
    z-index: 3;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    max-width: 760px;
    margin: 16px auto 20px;
    gap: 0;
}}

.feature-item {{
    text-align: center;
    padding: 5px 18px;
    border-right: 1px solid rgba(255,255,255,.16);
}}

.feature-item:last-child {{
    border-right: 0;
}}

.feature-icon {{
    color: var(--red);
    font-size: 1rem;
    margin-bottom: 3px;
}}

.feature-title {{
    color: #fff;
    font-size: .78rem;
    font-weight: 700;
}}

.feature-desc {{
    color: #aeb2ba;
    font-size: .65rem;
    margin-top: 2px;
}}

/* Search bar wrapper */
.search-container-box {{
    position: relative;
    width: 100%;
    margin: 0 auto;
}}

/* ضبط حقل الإدخال وإزالة المستطيلات الزائدة */
[data-testid="stTextInput"] {{
    position: relative !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}}

[data-testid="stTextInput"] > div {{
    border: 1px solid rgba(255,255,255,.24) !important;
    border-radius: 999px !important;
    background: rgba(10, 12, 14, 0.78) !important;
    box-shadow: 0 18px 55px rgba(0,0,0,.60) !important;
    backdrop-filter: blur(16px) !important;
    height: 52px !important;
    overflow: hidden !important;
}}

[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextInput"] [data-baseweb="input"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    height: 100% !important;
}}

[data-testid="stTextInput"] input {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #fff !important;
    height: 52px !important;
    line-height: 52px !important;
    padding-left: 20px !important;
    padding-right: 58px !important;
    font-size: .95rem !important;
    direction: ltr;
    text-align: left;
}}

[data-testid="stTextInput"] input:focus {{
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="stTextInput"] input::placeholder {{
    color: #b9bcc2 !important;
}}

/* تثبيت زر الكاميرا داخل حقل البحث على اليمين */
[data-testid="stFileUploader"] {{
    position: absolute !important;
    top: 0 !important;
    right: 8px !important;
    height: 52px !important;
    width: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 20 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}}

[data-testid="stFileUploader"] section {{
    padding: 0 !important;
    min-height: unset !important;
    border: none !important;
    background: transparent !important;
    pointer-events: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzone"] > div:not(:has(button)) {{
    display: none !important;
}}

[data-testid="stFileUploader"] button {{
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    background: rgba(18, 18, 20, 0.9) !important;
    border: 1px solid rgba(255, 52, 52, 0.6) !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #fff !important;
    cursor: pointer !important;
    transition: all .2s ease !important;
}}

[data-testid="stFileUploader"] button:hover {{
    transform: scale(1.08) !important;
    background: rgba(255, 52, 52, 0.22) !important;
    border-color: var(--red) !important;
}}

[data-testid="stFileUploader"] button:before {{
    content: "📷";
    font-size: .92rem;
    line-height: 1;
}}

[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {{
    display: none !important;
}}

[data-testid="stFileUploaderFile"] {{
    margin-top: 14px !important;
    background: rgba(5,5,6,.90) !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    border-radius: 12px !important;
}}

/* Cards & Badges */
.car-card {{
    background: rgba(5,6,8,.78);
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 16px 40px rgba(0,0,0,.52);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}}

.car-card:hover {{
    border-color: rgba(255,255,255,.30);
    transform: translateY(-2px);
}}

.deal-badge-great {{
    background: rgba(34,197,94,.13);
    border: 1px solid rgba(34,197,94,.65);
    color: #86efac;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}

.deal-badge-overpriced {{
    background: rgba(239,68,68,.13);
    border: 1px solid rgba(239,68,68,.65);
    color: #fca5a5;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}

.deal-badge-fair {{
    background: rgba(234,179,8,.13);
    border: 1px solid rgba(234,179,8,.65);
    color: #fde047;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}

@media(max-width:700px) {{
    .hero-title {{ font-size: 2.5rem; }}
    .hero-subtitle {{ font-size: .85rem; }}
    .feature-row {{ grid-template-columns: repeat(2, 1fr); }}
    .feature-item {{ border-right: 0; border-bottom: 1px solid rgba(255,255,255,.12); padding: 7px 10px; }}
    .feature-item:nth-child(odd) {{ border-right: 1px solid rgba(255,255,255,.12); }}
    .feature-item:nth-child(3), .feature-item:nth-child(4) {{ border-bottom: 0; }}
}}
</style>
""", unsafe_allow_html=True)

# طبقة صورة الخلفية
st.markdown('<div class="background-car"></div>', unsafe_allow_html=True)

# ==============================================================================
# 1. تحميل النماذج والبيانات
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_data = pd.read_csv("hatla2ee_cleaned_data.csv")
    for col in ["brand", "model", "location", "transmission", "car_condition", "search_text"]:
        if col in df_data.columns:
            df_data[col] = df_data[col].fillna("").astype(str)

    pricing_model = joblib.load("car_price_model.pkl")
    tfidf_vectorizer = joblib.load("car_tfidf.pkl")
    matrix = tfidf_vectorizer.transform(df_data["search_text"])

    nlp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = nlp_model.encode(
        df_data["search_text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    vision_name = "dima806/car_models_image_detection"
    processor = AutoImageProcessor.from_pretrained(vision_name)
    v_model = AutoModelForImageClassification.from_pretrained(vision_name).to(device)
    v_model.eval()

    return device, df_data, pricing_model, tfidf_vectorizer, matrix, nlp_model, embeddings, processor, v_model

try:
    (device, df_recommend, full_pricing_pipeline, tfidf, tfidf_matrix,
     st_model, car_text_embeddings, img_processor, car_vision_model) = load_all()
except Exception as e:
    st.error(f"⚠️ Project loading error: {e}")
    st.stop()

# ==============================================================================
# 2. دوال المعالجة والـ NLP
# ==============================================================================
CITY_MAP = {
    "القاهرة": "cairo", "cairo": "cairo", "اسكندرية": "alexandria", "إسكندرية": "alexandria", "alex": "alexandria",
    "التجمع": "tagamo3", "تجمع": "tagamo3", "new cairo": "tagamo3", "مدينة نصر": "nasr city", "nasr city": "nasr city",
    "الجيزة": "giza", "giza": "giza", "المعادي": "maadi", "maadi": "maadi", "العبور": "obour", "obour": "obour",
    "زايد": "zayed", "الشيخ زايد": "zayed", "اكتوبر": "october", "أكتوبر": "october",
    "كفر الدوار": "kafr el-dawwar", "المنصورة": "mansoura", "طنطا": "tanta"
}

TRANS_MAP = {
    "اوتوماتيك": "Automatic", "أوتوماتيك": "Automatic", "automatic": "Automatic", "auto": "Automatic",
    "مانيوال": "Manual", "manual": "Manual"
}

CONDITION_MAP = {
    "جديدة": "New", "جديد": "New", "زيرو": "New", "new": "New",
    "مستعملة": "Used", "مستعمل": "Used", "used": "Used"
}

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text.lower())
    return text.replace("ونص", ".5").replace("وربع", ".25")

def extract_budget(text):
    def scale(v, unit):
        if unit in ("m", "مليون"): return v * 1_000_000
        if unit in ("k", "الف", "ألف"): return v * 1_000
        return v
    m = re.search(r'(?:من\s*)?(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?\s*(?:-|to|حتى|لحد|الى|إلى|لـ)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?', text)
    if m:
        return scale(float(m.group(1)), m.group(2)), scale(float(m.group(3)), m.group(4))
    m = re.search(r'(?:under|below|less than|حتى|لحد|اقل من|أقل من|تحت)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?', text)
    if m:
        return None, scale(float(m.group(1)), m.group(2))
    return None, None

def parse_query_to_filters(query, catalog_df):
    text = normalize_text(query)
    filters = {}
    min_p, max_p = extract_budget(text)
    if min_p is not None: filters["min_price"] = min_p
    if max_p is not None: filters["max_price"] = max_p

    for k, v in TRANS_MAP.items():
        if k in text:
            filters["transmission"] = v
            break

    for k, v in CONDITION_MAP.items():
        if k in text:
            filters["car_condition"] = v
            break

    if "brand" in catalog_df.columns:
        brands = [b for b in catalog_df["brand"].dropna().unique() if str(b).strip()]
        for b in sorted(brands, key=len, reverse=True):
            if str(b).lower() in text:
                filters["brand"] = b
                break

    if "location" in catalog_df.columns:
        for ar_key, mapped_val in CITY_MAP.items():
            if ar_key in text:
                for loc in catalog_df["location"].dropna().unique():
                    if mapped_val in str(loc).lower():
                        filters["location"] = loc
                        break
                break
    return filters

def predict_vision_top5(image_pil, top_k=5):
    inputs = img_processor(images=image_pil.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = car_vision_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    k = min(top_k, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, k)
    return [{"label": car_vision_model.config.id2label[idx.item()], "confidence": float(p.item())}
            for p, idx in zip(top_probs, top_indices)]

# ==============================================================================
# 3. واجهة البحث
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-content">
        <div class="hero-kicker">✦ SMART CAR MARKET</div>
        <div class="hero-title">Apex <span>Motors</span></div>
        <div class="hero-line"></div>
        <div class="hero-subtitle">Find the right car, get expert insights, and make smarter decisions with the power of AI.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-row">
    <div class="feature-item"><div class="feature-icon">⌁</div><div class="feature-title">Analysis</div><div class="feature-desc">Understand your needs</div></div>
    <div class="feature-item"><div class="feature-icon">▧</div><div class="feature-title">Image Detection</div><div class="feature-desc">Identify car details</div></div>
    <div class="feature-item"><div class="feature-icon">◇</div><div class="feature-title">Price Prediction</div><div class="feature-desc">Get fair market value</div></div>
    <div class="feature-item"><div class="feature-icon">▥</div><div class="feature-title">Smart Results</div><div class="feature-desc">Best matches for you</div></div>
</div>
""", unsafe_allow_html=True)

_, col_search, _ = st.columns([1, 2.6, 1])

with col_search:
    st.markdown('<div class="search-container-box">', unsafe_allow_html=True)
    user_query = st.text_input(
        "Search",
        placeholder="Type your car requirements and press Enter...",
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader(
        "Upload car image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. معالجة النتائج
# ==============================================================================
has_query = bool(user_query.strip())
has_image = uploaded_file is not None

if has_query or has_image:
    sub_df = df_recommend.copy()
    filters = {}

    if has_query:
        filters = parse_query_to_filters(user_query, df_recommend)
        if "price" in sub_df.columns:
            if "min_price" in filters: sub_df = sub_df[sub_df["price"] >= filters["min_price"]]
            if "max_price" in filters: sub_df = sub_df[sub_df["price"] <= filters["max_price"]]
        for col in ["brand", "location", "transmission", "car_condition"]:
            if col in filters and col in sub_df.columns:
                sub_df = sub_df[sub_df[col].astype(str).str.lower() == str(filters[col]).lower()]

    if has_image:
        try:
            pil_img = Image.open(uploaded_file)
            preds = predict_vision_top5(pil_img)
            
            top_car_label = preds[0]['label']
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <span style="background: rgba(255, 52, 52, 0.15); border: 1px solid var(--red); color: #fff; padding: 6px 18px; border-radius: 20px; font-size: 0.95rem;">
                    📷 Detected car: <strong>{top_car_label}</strong> ({preds[0]['confidence']*100:.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            if "brand" in sub_df.columns and "model" in sub_df.columns:
                matched_rows = pd.DataFrame()
                for p in preds:
                    lbl = p["label"].lower()
                    mask = (
                        sub_df["brand"].fillna("").astype(str).str.lower().apply(lambda b: bool(b) and b in lbl)
                        | sub_df["model"].fillna("").astype(str).str.lower().apply(lambda m: bool(m) and m in lbl)
                    )
                    subset = sub_df[mask].copy()
                    if not subset.empty:
                        subset["vision_confidence"] = p["confidence"]
                        matched_rows = pd.concat([matched_rows, subset])
                
                dedup_cols = [c for c in ["brand", "model", "year"] if c in matched_rows.columns]
                if not matched_rows.empty and dedup_cols:
                    sub_df = matched_rows.drop_duplicates(subset=dedup_cols).copy()
                elif not matched_rows.empty:
                    sub_df = matched_rows.copy()
        except Exception:
            pass

    if sub_df.empty:
        sub_df = df_recommend.copy()

    if has_query:
        sub_pos = [df_recommend.index.get_loc(i) for i in sub_df.index]
        q_emb = st_model.encode([user_query], normalize_embeddings=True)[0]
        sem_scores = np.dot(car_text_embeddings[sub_pos], q_emb)
        q_tfidf = tfidf.transform([user_query])
        tfidf_scores = cosine_similarity(q_tfidf, tfidf_matrix[sub_pos])[0]
        nlp_score = (0.75 * sem_scores + 0.25 * tfidf_scores).clip(0, 1) * 100
    else:
        nlp_score = 100.0

    sub_df = sub_df.copy()
    if "vision_confidence" in sub_df.columns:
        sub_df["match_score"] = (0.60 * nlp_score + 0.40 * (sub_df["vision_confidence"] * 100)).round(2)
    else:
        sub_df["match_score"] = np.round(nlp_score, 2)

    top_results = sub_df.sort_values("match_score", ascending=False).head(6).copy()

    # التنبؤ بالسعر العادل مع تفادي الـ KeyError
    try:
        try:
            pred_features = list(full_pricing_pipeline.feature_names_in_)
        except AttributeError:
            pred_features = list(full_pricing_pipeline.named_steps["preprocessor"].feature_names_in_)

        eval_df = pd.DataFrame(index=top_results.index)
        for f in pred_features:
            if f in top_results.columns:
                eval_df[f] = top_results[f]
            else:
                eval_df[f] = np.nan

        for col in eval_df.columns:
            if col in ["year", "mileage", "engine_capacity", "horsepower", "car_age", "km_per_year"]:
                eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

        pred_log = full_pricing_pipeline.predict(eval_df)
        top_results["predicted_fair_price"] = np.expm1(pred_log).round(0)
    except Exception:
        top_results["predicted_fair_price"] = top_results["price"]

    top_results["price_difference"] = (top_results["price"] - top_results["predicted_fair_price"]).round(0)
    safe_fair_price = top_results["predicted_fair_price"].replace(0, np.nan)
    top_results["pct_diff"] = ((top_results["price_difference"] / safe_fair_price) * 100).fillna(0)

    top_results["deal_label"] = np.select(
        [top_results["pct_diff"] <= -8.0, top_results["pct_diff"] >= 8.0],
        ["Great Deal", "Overpriced"],
        default="Fair Price"
    )

    _, col_results, _ = st.columns([1, 2.6, 1])
    with col_results:
        st.markdown('<div class="results-title">Best Matches</div>', unsafe_allow_html=True)

        for _, row in top_results.iterrows():
            b_name = row.get("brand", "")
            m_name = row.get("model", "")
            y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
            price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
            fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
            deal_tag = row.get("deal_label", "Fair Price")
            loc_val = row.get("location", "Egypt")
            trans_val = row.get("transmission", "-")
            mileage_val = f"{int(row.get('mileage', 0)):,} km" if pd.notna(row.get("mileage")) and row.get("mileage") > 0 else "Brand New"
            item_link = row.get("item_url", "#")

            if deal_tag == "Great Deal":
                badge_html = '<span class="deal-badge-great">🟢 Great Deal</span>'
            elif deal_tag == "Overpriced":
                badge_html = '<span class="deal-badge-overpriced">🔴 Overpriced</span>'
            else:
                badge_html = '<span class="deal-badge-fair">🟡 Fair Price</span>'

            st.markdown(f"""
            <div class="car-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">
                        {b_name} {m_name} <span style="color: var(--red);">{y_val}</span>
                    </span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; gap: 15px; margin-top: 8px; color: #94a3b8; font-size: 0.88rem;">
                    <span>🕹️ {trans_val}</span>
                    <span>🛣️ {mileage_val}</span>
                    <span>📍 {loc_val}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 14px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Listed Price:</span><br>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Estimated Fair Price:</span><br>
                        <strong style="color: var(--red); font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <a href="{item_link}" target="_blank" style="display: inline-block; background: rgba(255, 52, 52, 0.15); border: 1px solid var(--red); color: #fff; padding: 8px 18px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 0.92rem;">
                            View Listing ↗
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <p style="font-size: 0.95rem;">Type your car requirements and press <strong>Enter</strong>, or click the camera 📷 to upload a car image</p>
    </div>
    """, unsafe_allow_html=True)
