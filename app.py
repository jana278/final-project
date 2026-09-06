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
    page_title="Apex Motors | دور على عربيتك",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# MINIMALIST MODERN GLASS UI (CLEAN & USER-FRIENDLY)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(130% 100% at 50% -10%, #0d1b2a 0%, #060d17 60%, #020509 100%);
        color: #f8fafc;
        font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
    }

    /* Header */
    .hero-box {
        text-align: center;
        padding: 40px 10px 20px 10px;
        max-width: 800px;
        margin: 0 auto;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    /* Clean Search Bar Wrapper */
    .stTextInput input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.3) !important;
        color: #ffffff !important;
        border-radius: 30px !important;
        padding: 14px 25px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.4) !important;
    }

    /* Clean Uploader Button styling */
    [data-testid="stFileUploader"] {
        padding-top: 5px;
    }

    /* Car Result Cards */
    .car-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .car-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
    }

    /* Badges */
    .deal-badge-great {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .deal-badge-overpriced {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .deal-badge-fair {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #facc15;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LOAD MODELS & ARTIFACTS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    v_model = AutoModelForImageClassification.from_pretrained(vision_name).to(device)
    v_model.eval()

    return device, df_data, pricing_model, tfidf_vectorizer, matrix, nlp_model, embeddings, processor, v_model

try:
    (device, df_recommend, full_pricing_pipeline, tfidf, tfidf_matrix,
     st_model, car_text_embeddings, img_processor, car_vision_model) = load_all()
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل ملفات المشروع: {e}")
    st.stop()

# ==============================================================================
# 2. NLP & SEARCH HELPERS
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
# 3. UNIFIED USER-FRIENDLY SEARCH BAR
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Apex Motors</div>
    <div class="hero-subtitle">محرك البحث الذكي وتقييم أسعار السيارات في السوق المصري</div>
</div>
""", unsafe_allow_html=True)

# شريط البحث المتكامل في المنتصف
search_container = st.container()
with search_container:
    col_input, col_file = st.columns([4, 1.2], gap="small")
    
    with col_input:
        user_query = st.text_input(
            "ابحث هنا",
            placeholder="اكتب مواصفات العربية اللي بتدور عليها واضغط Enter...",
            label_visibility="collapsed"
        )
    with col_file:
        uploaded_file = st.file_uploader(
            "📷 بالصورة",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="ارفع صورة عربية للبحث المباشر"
        )

# ==============================================================================
# 4. INSTANT RESULTS ON ENTER OR IMAGE UPLOAD
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
        for col in ["brand", "location"]:
            if col in filters and col in sub_df.columns:
                sub_df = sub_df[sub_df[col].astype(str).str.lower() == str(filters[col]).lower()]

    if has_image:
        try:
            pil_img = Image.open(uploaded_file)
            preds = predict_vision_top5(pil_img)
            
            top_car_label = preds[0]['label']
            st.success(f"🔍 تم التعرف على السيارة: **{top_car_label}** بنسبة ثقة {preds[0]['confidence']*100:.1f}%")

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

    # التثمين والسعر العادل
    try:
        pred_features = list(full_pricing_pipeline.feature_names_in_)
    except AttributeError:
        pred_features = list(full_pricing_pipeline.named_steps["preprocessor"].feature_names_in_)

    eval_df = top_results[[f for f in pred_features if f in top_results.columns]].copy()
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
        ["Great Deal", "Overpriced"],
        default="Fair Price"
    )

    st.write("")
    st.subheader("أفضل السيارات المتاحة المطابقة لطلبك:")

    # كروت النتائج النظيفة
    for _, row in top_results.iterrows():
        b_name = row.get("brand", "")
        m_name = row.get("model", "")
        y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
        price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
        fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
        deal_tag = row.get("deal_label", "Fair Price")
        loc_val = row.get("location", "مصر")
        item_link = row.get("item_url", "#")

        if deal_tag == "Great Deal":
            badge_html = '<span class="deal-badge-great">🟢 صفقة لقطة</span>'
        elif deal_tag == "Overpriced":
            badge_html = '<span class="deal-badge-overpriced">🔴 سعر أعلى من السوق</span>'
        else:
            badge_html = '<span class="deal-badge-fair">🟡 سعر عادل</span>'

        st.markdown(f"""
        <div class="car-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                    {b_name} {m_name} <span style="color: #38bdf8;">{y_val}</span>
                </span>
                <div>{badge_html}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">السعر المطلوب:</span> 
                    <strong style="color: #ffffff; font-size: 1.15rem; margin-right: 5px;">{price_val:,.0f} EGP</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">السعر العادل بالذكاء الاصطناعي:</span> 
                    <strong style="color: #38bdf8; font-size: 1.15rem; margin-right: 5px;">{fair_val:,.0f} EGP</strong>
                </div>
                <div>
                    <span style="color: #cbd5e1; font-size: 0.95rem;">📍 {loc_val}</span>
                </div>
                <div>
                    <a href="{item_link}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 700;">
                        تفاصيل الإعلان ↗
                    </a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # شاشة ترحيبية نظيفة قبل البحث
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 60px;">
        <p style="font-size: 1.1rem;">جرب تكتب مثلاً: <strong>"هيونداي النترا مستعملة في إسكندرية بحدود 600 ألف"</strong></p>
        <p style="font-size: 0.9rem;">أو ارفع صورة أي عربية من زرار الكاميرا فوق 📷</p>
    </div>
    """, unsafe_allow_html=True)

