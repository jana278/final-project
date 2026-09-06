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
    page_title="APEX MOTORS | Next-Gen AI Valuation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# LUXURY SHOWROOM UI & VISUAL ASSETS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Cairo:wght@600;700;900&display=swap');

    .stApp {
        background: radial-gradient(120% 90% at 50% -10%, #0d1e3a 0%, #050b14 60%, #020408 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
    }

    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 40px 15px 15px 15px;
        max-width: 900px;
        margin: 0 auto;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 14px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #bae6fd 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 10px;
        font-weight: 400;
    }

    /* Unified Glass Search Bar */
    .unified-search-box {
        position: relative;
        max-width: 780px;
        margin: 25px auto 35px auto;
    }
    .unified-search-box [data-testid="stTextInput"] input {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1.8px solid rgba(56, 189, 248, 0.35) !important;
        color: #ffffff !important;
        border-radius: 45px !important;
        height: 64px !important;
        padding-left: 28px !important;
        padding-right: 70px !important;
        font-size: 1.15rem !important;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.7), inset 0 0 15px rgba(56, 189, 248, 0.05) !important;
        transition: all 0.3s ease !important;
        direction: rtl;
        text-align: right;
    }
    .unified-search-box [data-testid="stTextInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.5) !important;
    }

    /* Embedded Camera Icon */
    .unified-search-box [data-testid="stFileUploader"] {
        margin-top: -64px !important;
        height: 64px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-right: 20px !important;
        pointer-events: none !important;
        border: none !important;
        background: transparent !important;
    }
    .unified-search-box [data-testid="stFileUploader"] section {
        padding: 0 !important;
        min-height: unset !important;
        border: none !important;
        background: transparent !important;
        pointer-events: auto !important;
    }
    .unified-search-box [data-testid="stFileUploaderDropzone"] {
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    .unified-search-box [data-testid="stFileUploaderDropzoneInstructions"],
    .unified-search-box [data-testid="stFileUploaderDropzone"] > div:not(:has(button)) {
        display: none !important;
    }
    .unified-search-box [data-testid="stFileUploader"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px !important;
        color: #38bdf8 !important;
        transition: transform 0.2s ease !important;
    }
    .unified-search-box [data-testid="stFileUploader"] button:hover {
        transform: scale(1.25);
    }
    .unified-search-box [data-testid="stFileUploader"] button::before {
        content: "📷";
        font-size: 1.5rem;
    }
    .unified-search-box [data-testid="stFileUploader"] button span,
    .unified-search-box [data-testid="stFileUploader"] button p {
        display: none !important;
    }

    /* Showcase Vehicle Cards */
    .showroom-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
        gap: 24px;
        max-width: 1200px;
        margin: 0 auto;
        padding-bottom: 50px;
    }
    .car-card-modern {
        background: linear-gradient(180deg, rgba(17, 27, 46, 0.85) 0%, rgba(10, 16, 28, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.65);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
    }
    .car-card-modern:hover {
        border-color: rgba(56, 189, 248, 0.65);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px -10px rgba(14, 165, 233, 0.25);
    }
    .car-image-container {
        width: 100%;
        height: 200px;
        background: #09121d;
        overflow: hidden;
        position: relative;
    }
    .car-image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .car-card-modern:hover .car-image-container img {
        transform: scale(1.06);
    }
    .car-details {
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        flex-grow: 1;
    }

    /* Badges */
    .deal-badge-great {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
    .deal-badge-overpriced {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
    .deal-badge-fair {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #facc15;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LOAD MODELS & ASSETS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_core_systems():
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
     st_model, car_text_embeddings, img_processor, car_vision_model) = load_core_systems()
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل ملفات المشروع: {e}")
    st.stop()

# ==============================================================================
# 2. HELPER FUNCTIONS & IMAGE MAPPING
# ==============================================================================
CITY_MAP = {
    "القاهرة": "cairo", "cairo": "cairo", "اسكندرية": "alexandria", "إسكندرية": "alexandria",
    "التجمع": "tagamo3", "تجمع": "tagamo3", "مدينة نصر": "nasr city", "الجيزة": "giza",
    "زايد": "zayed", "اكتوبر": "october", "كفر الدوار": "kafr el-dawwar", "المنصورة": "mansoura"
}

def get_car_image_url(brand, model):
    query = f"{brand} {model}".strip()
    return f"https://source.unsplash.com/featured/800x500/?car,{query}"

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text.lower())
    return text.replace("ونص", ".5").replace("وربع", ".25")

def extract_budget(text):
    def scale(v, unit):
        if unit in ("m", "مليون"): return v * 1_000_000
        if unit in ("k", "الف", "ألف"): return v * 1_000
        return v
    m = re.search(r'(?:من\s*)?(\\d+(?:\\.\\d+)?)\\s*(m|مليون|k|الف|ألف)?\\s*(?:-|to|حتى|لحد|الى|إلى|لـ)\\s*(\\d+(?:\\.\\d+)?)\\s*(m|مليون|k|الف|ألف)?', text)
    if m:
        return scale(float(m.group(1)), m.group(2)), scale(float(m.group(3)), m.group(4))
    m = re.search(r'(?:under|below|less than|حتى|لحد|اقل من|أقل من|تحت)\\s*(\\d+(?:\\.\\d+)?)\\s*(m|مليون|k|الف|ألف)?', text)
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
# 3. INTERACTIVE HERO & INTEGRATED SEARCH
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ End-to-End Multimodal AI</div>
    <h1 class="hero-title">APEX MOTORS</h1>
    <p class="hero-subtitle">محرك البحث الذكي وتقييم الأسعار العادلة للسيارات في السوق المصري</p>
</div>
""", unsafe_allow_html=True)

# شريط البحث المتكامل
st.markdown('<div class="unified-search-box">', unsafe_allow_html=True)
user_query = st.text_input(
    "Search",
    placeholder="اكتب مواصفات العربية (مثال: كيا سبورتاج في القاهرة أقل من مليون ونصف)...",
    label_visibility="collapsed"
)
uploaded_file = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. INSTANT AUTO EXECUTION & VALUATION
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
            top_pred = preds[0]
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 25px;">
                <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #bae6fd; padding: 8px 22px; border-radius: 30px; font-weight: 700; font-size: 1rem;">
                    📷 طراز السيارة المكتشف: {top_pred['label']} ({top_pred['confidence']*100:.1f}%)
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

    # التثمين
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

    # عرض كروت السيارات بصور حقيقية
    st.markdown("""
    <div style="max-width: 1200px; margin: 0 auto 20px auto; display: flex; justify-content: space-between; align-items: center;">
        <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700;">🏁 أفضل السيارات المتاحة المطابقة:</h3>
        <span style="color: #94a3b8; font-size: 0.95rem;">مرتبة بدقة التوافق الدلالي والتسعير العادل</span>
    </div>
    """, unsafe_allow_html=True)

    cards_html = '<div class="showroom-grid">'
    for _, row in top_results.iterrows():
        b_name = row.get("brand", "")
        m_name = row.get("model", "")
        y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
        price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
        fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
        deal_tag = row.get("deal_label", "Fair Price")
        loc_val = row.get("location", "مصر")
        trans_val = row.get("transmission", "Automatic")
        item_link = row.get("item_url", "#")
        img_url = get_car_image_url(b_name, m_name)

        if deal_tag == "Great Deal":
            badge_html = '<span class="deal-badge-great">🟢 صفقة لقطة</span>'
        elif deal_tag == "Overpriced":
            badge_html = '<span class="deal-badge-overpriced">🔴 سعر أعلى من السوق</span>'
        else:
            badge_html = '<span class="deal-badge-fair">🟡 سعر عادل</span>'

        cards_html += f"""
        <div class="car-card-modern">
            <div class="car-image-container">
                <img src="{img_url}" alt="{b_name} {m_name}" loading="lazy" onerror="this.src='https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800';"/>
            </div>
            <div class="car-details">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h4 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #ffffff;">{b_name} {m_name}</h4>
                        <span style="color: #38bdf8; font-weight: 600; font-size: 0.95rem;">موديل {y_val}</span>
                    </div>
                    {badge_html}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; background: rgba(3, 7, 18, 0.5); padding: 12px 14px; border-radius: 12px; margin-top: 5px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.8rem; display: block;">السعر المعروض</span>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 0.8rem; display: block;">السعر العادل المقدر</span>
                        <strong style="color: #38bdf8; font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                    <span>📍 {loc_val}</span>
                    <span>⚙️ {trans_val}</span>
                </div>
                <a href="{item_link}" target="_blank" style="display: block; text-align: center; background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff; padding: 10px; border-radius: 10px; font-weight: 700; text-decoration: none; margin-top: 6px; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3);">
                    عرض تفاصيل الإعلان ↗
                </a>
            </div>
        </div>
        """
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 40px;">
        <p style="font-size: 1.15rem; margin-bottom: 6px;">💡 اكتب مواصفات العربية واضغط <strong>Enter</strong> مباشرة</p>
        <p style="font-size: 0.95rem;">أو اضغط على أيقونة الكاميرا 📷 داخل شريط البحث للبحث بصورة عربية</p>
    </div>
    """, unsafe_allow_html=True)
