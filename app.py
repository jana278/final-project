import os
import re
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
    page_title="Apex Motors | AI Valuation & Search",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# FUTURISTIC NEON BLUE CYBER THEME (CUSTOM CSS)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap');

    /* Global Dark Midnight Theme */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d1b2a 0%, #070d14 60%, #03070b 100%);
        color: #e0f2fe;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Titles & Headings */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 25px rgba(0, 242, 254, 0.4);
        margin-bottom: 0px;
    }

    .sub-title {
        text-align: center;
        color: #7dd3fc;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Glassmorphic Neon Cards */
    .cyber-card {
        background: rgba(13, 27, 42, 0.65);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 242, 254, 0.05);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .cyber-card:hover {
        border-color: rgba(0, 242, 254, 0.6);
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
    }

    /* Inputs Styling */
    .stTextInput input {
        background: rgba(7, 13, 20, 0.8) !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        color: #f0f9ff !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        box-shadow: inset 0 0 10px rgba(0, 242, 254, 0.1) !important;
    }
    .stTextInput input:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.5) !important;
    }

    /* Neon Cyber Button */
    .stButton button {
        background: linear-gradient(135deg, #0052d4 0%, #4364f7 50%, #6fb1fc 100%) !important;
        border: 1px solid #00f2fe !important;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        padding: 12px 24px !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4364f7 0%, #00f2fe 100%) !important;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.8) !important;
        transform: translateY(-2px);
    }

    /* Metrics & Badges */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #38bdf8 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
    }

    /* Streamlit DataFrame Customization */
    .stDataFrame {
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LOAD MODELS & ARTIFACTS
# ==============================================================================
@st.cache_resource
def load_models_and_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    required_files = ["hatla2ee_cleaned_data.csv", "car_price_model.pkl", "car_tfidf.pkl"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"الملفات دي ناقصة من الريبو: {', '.join(missing)}. "
            "تأكدي إنها مرفوعة على GitHub في نفس فولدر app.py."
        )

    df_data = pd.read_csv("hatla2ee_cleaned_data.csv")

    for col in ["brand", "model", "location", "search_text"]:
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
    vision_model = AutoModelForImageClassification.from_pretrained(vision_name).to(device)
    vision_model.eval()

    return device, df_data, pricing_model, tfidf_vectorizer, matrix, nlp_model, embeddings, processor, vision_model


try:
    with st.spinner("Initializing Cyber Neural Engines..."):
        (device, df_recommend, full_pricing_pipeline, tfidf, tfidf_matrix,
         st_model, car_text_embeddings, img_processor, car_vision_model) = load_models_and_data()
except Exception as e:
    st.error(f"⚠️ فشل تحميل الموديلات والبيانات: {e}")
    st.stop()

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
CITY_MAP = {
    "القاهرة": "cairo", "cairo": "cairo", "اسكندرية": "alexandria", "إسكندرية": "alexandria",
    "التجمع": "tagamo3", "تجمع": "tagamo3", "مدينة نصر": "nasr city", "الجيزة": "giza",
    "زايد": "zayed", "اكتوبر": "october", "كفر الدوار": "kafr el-dawwar", "المنصورة": "mansoura"
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
# 3. INTERFACE
# ==============================================================================
st.markdown('<div class="main-title">APEX // INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Neural Vehicle Valuation & Multimodal Search Matrix</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1.2], gap="large")

with col1:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("### 💬 NATURAL QUERY PARSER")
    user_query = st.text_input(
        "Search Protocol",
        placeholder="e.g. عايز عربية اوتوماتيك في القاهرة تحت 2 مليون",
        label_visibility="collapsed"
    )
    st.caption("Supports Egyptian Arabic, Franco-Arab, City constraints, and budget limits.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("### 📷 COMPUTER VISION SCANNER")
    uploaded_file = st.file_uploader(
        "Image Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.caption("Upload any exterior car photo for high-accuracy neural identification.")
    st.markdown('</div>', unsafe_allow_html=True)

col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    top_n = st.slider("MAX CANDIDATE EXTRACTION", min_value=1, max_value=15, value=5)
with col_ctrl2:
    st.write("")
    search_btn = st.button("RUN PIPELINE ⚡", use_container_width=True)

if search_btn:
    sub_df = df_recommend.copy()
    filters = {}
    
    # Text Filter
    if user_query.strip():
        filters = parse_query_to_filters(user_query, df_recommend)
        if "price" in sub_df.columns:
            if "min_price" in filters: sub_df = sub_df[sub_df["price"] >= filters["min_price"]]
            if "max_price" in filters: sub_df = sub_df[sub_df["price"] <= filters["max_price"]]
        for col in ["brand", "location"]:
            if col in filters and col in sub_df.columns:
                sub_df = sub_df[sub_df[col].astype(str).str.lower() == str(filters[col]).lower()]

    # Vision Filter
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
        except Exception:
            st.error("⚠️ مقدرتش أفتح الصورة المرفوعة. جربي صورة JPG/PNG سليمة.")
            pil_img = None

        if pil_img is not None:
            preds = predict_vision_top5(pil_img)

            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 🛰️ VISION DECODER TELEMETRY")
            v_cols = st.columns(len(preds))
            for idx, p in enumerate(preds):
                with v_cols[idx]:
                    st.metric(label=p['label'], value=f"{p['confidence']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

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

    if sub_df.empty:
        st.info("⚠️ System Notice: Broadening search scope across the entire catalog.")
        sub_df = df_recommend.copy()

    # NLP Ranking
    if user_query.strip():
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

    top_results = sub_df.sort_values("match_score", ascending=False).head(int(top_n)).copy()

    if top_results.empty:
        st.warning("⚠️ مفيش نتائج مطابقة. جربي تغيّري شروط البحث.")
        st.stop()

    # Valuation (Safe & Robust)
    try:
        pred_features = list(full_pricing_pipeline.feature_names_in_)
    except AttributeError:
        pred_features = list(full_pricing_pipeline.named_steps["preprocessor"].feature_names_in_)

    missing_features = [f for f in pred_features if f not in top_results.columns]

    if missing_features:
        st.warning(
            "⚠️ مقدرتش أحسب السعر العادل: أعمدة ناقصة في البيانات "
            f"({', '.join(missing_features)}). هيتم عرض النتائج من غير تقييم سعر."
        )
        top_results["predicted_fair_price"] = np.nan
        top_results["deal_label"] = "—"
    else:
        eval_df = top_results[pred_features].copy()
        for col in eval_df.columns:
            if col in ["year", "mileage", "engine_capacity", "horsepower", "car_age", "km_per_year"]:
                eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

        pred_log = full_pricing_pipeline.predict(eval_df)
        top_results["predicted_fair_price"] = np.expm1(pred_log).round(0)
        top_results["price_difference"] = (top_results["price"] - top_results["predicted_fair_price"]).round(0)

        safe_fair_price = top_results["predicted_fair_price"].replace(0, np.nan)
        pct_diff = (top_results["price_difference"] / safe_fair_price).fillna(0)
        top_results["deal_label"] = np.select(
            [pct_diff <= -0.08, pct_diff >= 0.08],
            ["⚡ GREAT DEAL", "🚨 OVERPRICED"],
            default="💠 FAIR PRICE"
        )

    st.markdown("### 🏁 MATCHED MARKET INVENTORY")
    display_cols = ["brand", "model", "year", "price", "predicted_fair_price", "deal_label", "location", "match_score", "item_url"]
    show_cols = [c for c in display_cols if c in top_results.columns]

    st.dataframe(
        top_results[show_cols],
        column_config={
            "price": st.column_config.NumberColumn("Listed Price", format="%d EGP"),
            "predicted_fair_price": st.column_config.NumberColumn("Fair Valuation", format="%d EGP"),
            "match_score": st.column_config.ProgressColumn("Confidence Match", format="%.1f%%", min_value=0, max_value=100),
            "item_url": st.column_config.LinkColumn("Listing URL")
        },
        use_container_width=True,
        hide_index=True
    )
