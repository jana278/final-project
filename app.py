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
    page_title="Apex Motors | دور على عربيتك",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# صورة العربية المستخدمة كخلفية كاملة للواجهة
@st.cache_data(show_spinner=False)
def get_background_image():
    image_path = "hero_car.jpg"
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

BG_IMAGE = get_background_image()

# ==============================================================================
# CSS: شريط بحث مدمج في المنتصف مع أيقونة الكاميرا بالداخل
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@500;700&display=swap');

    /* ============================================================
       الخلفية: صورة العربية تغطي الصفحة كلها مع طبقة داكنة
       ============================================================ */
    .stApp {{
        background:
            linear-gradient(180deg, rgba(2, 5, 9, 0.56) 0%, rgba(2, 5, 9, 0.76) 48%, rgba(2, 5, 9, 0.94) 100%),
            url("{BG_IMAGE}") center center / cover fixed no-repeat !important;
        color: #f8fafc;
        font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    /* تأثير زجاجي خفيف فوق الخلفية */
    .main .block-container::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 50% 18%, rgba(56, 189, 248, 0.08), transparent 34%);
        z-index: -1;
    }}

    /* ============================================================
       Hero جديد - أكثر فخامة ووضوحاً
       ============================================================ */
    .hero-box {{
        text-align: center;
        padding: 48px 15px 30px 15px;
        position: relative;
    }}

    .hero-kicker {{
        display: inline-block;
        padding: 7px 16px;
        margin-bottom: 13px;
        border: 1px solid rgba(125, 211, 252, 0.35);
        border-radius: 999px;
        background: rgba(8, 15, 27, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #bae6fd;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
    }}

    .hero-title {{
        font-size: clamp(2.4rem, 5vw, 4.1rem);
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff 0%, #7dd3fc 42%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }}

    .hero-subtitle {{
        color: #dbeafe;
        font-size: 1.02rem;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.9;
        text-shadow: 0 2px 12px rgba(0,0,0,0.8);
    }}

    /* ============================================================
       بوكس البحث - شكل مستوحى من جسم العربية / الكابينة
       ============================================================ */
    [data-testid="stTextInput"] {{
        position: relative;
        z-index: 5;
        padding: 18px 20px 20px 20px !important;
        border: 1px solid rgba(125, 211, 252, 0.38) !important;
        border-radius: 48px 48px 28px 28px !important;
        background:
            linear-gradient(180deg, rgba(17, 28, 43, 0.88), rgba(5, 11, 19, 0.92)) !important;
        box-shadow:
            0 24px 60px rgba(0,0,0,0.48),
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 0 0 1px rgba(56, 189, 248, 0.05) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }}

    /* سقف العربية / الزجاج الأمامي */
    [data-testid="stTextInput"]::before {{
        content: "";
        position: absolute;
        top: -17px;
        left: 31%;
        width: 38%;
        height: 31px;
        border: 1px solid rgba(125, 211, 252, 0.28);
        border-bottom: 0;
        border-radius: 48px 48px 0 0;
        background: linear-gradient(180deg, rgba(10,18,29,0.55), rgba(10,18,29,0));
        transform: perspective(90px) rotateX(-5deg);
        pointer-events: none;
    }}

    [data-testid="stTextInput"] input {{
        background: rgba(2, 7, 14, 0.74) !important;
        border: 1px solid rgba(148, 163, 184, 0.23) !important;
        color: #ffffff !important;
        border-radius: 30px !important;
        height: 62px !important;
        padding-left: 25px !important;
        padding-right: 72px !important;
        font-size: 1.08rem !important;
        box-shadow:
            inset 0 2px 12px rgba(0,0,0,0.35),
            0 8px 25px rgba(0,0,0,0.22) !important;
        transition: all 0.3s ease !important;
        direction: rtl;
        text-align: right;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #94a3b8 !important;
    }}

    [data-testid="stTextInput"] input:focus {{
        border-color: #38bdf8 !important;
        box-shadow:
            0 0 0 3px rgba(56, 189, 248, 0.11),
            0 0 28px rgba(56, 189, 248, 0.24),
            inset 0 2px 12px rgba(0,0,0,0.35) !important;
    }}

    /* زر رفع الصورة يظل داخل منطقة البحث كزر كاميرا */
    [data-testid="stFileUploader"] {{
        margin-top: -62px !important;
        height: 62px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-right: 28px !important;
        pointer-events: none !important;
        border: none !important;
        background: transparent !important;
        position: relative;
        z-index: 10;
    }}

    [data-testid="stFileUploader"] section {{
        padding: 0 !important;
        min-height: unset !important;
        border: none !important;
        background: transparent !important;
        pointer-events: auto !important;
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
        background: rgba(56, 189, 248, 0.10) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        padding: 4px !important;
        cursor: pointer !important;
        color: #7dd3fc !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stFileUploader"] button:hover {{
        transform: scale(1.08) !important;
        background: rgba(56, 189, 248, 0.18) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.25) !important;
    }}

    [data-testid="stFileUploader"] button::before {{
        content: "📷";
        font-size: 1.25rem;
    }}

    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {{
        display: none !important;
    }}

    /* اسم الصورة بعد الرفع */
    [data-testid="stFileUploaderFile"] {{
        margin-top: 16px !important;
        background: rgba(5, 12, 22, 0.86) !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
        border-radius: 14px !important;
        pointer-events: auto !important;
        backdrop-filter: blur(10px);
    }}

    /* ============================================================
       كروت النتائج - باقي الكونسيبت كما هو مع لمسة glass بسيطة
       ============================================================ */
    .car-card {{
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
    }}

    .car-card:hover {{
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
    }}

    .deal-badge-great {{
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-overpriced {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-fair {{
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #facc15;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    /* تحسين العرض على الموبايل */
    @media (max-width: 700px) {{
        .hero-box {{ padding-top: 30px; }}
        .hero-subtitle {{ font-size: 0.92rem; }}
        [data-testid="stTextInput"] {{ border-radius: 34px 34px 22px 22px !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. تحميل النماذج والبيانات
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
# 2. دوال المعالجة والـ NLP
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
# 3. واجهة البحث في المنتصف تماماً
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-kicker">🚘 SMART CAR MARKET</div>
    <div class="hero-title">Apex Motors</div>
    <div class="hero-subtitle">محرك البحث الذكي وتقييم أسعار السيارات في السوق المصري</div>
</div>
""", unsafe_allow_html=True)

# ضبط الشريط في منتصف الشاشة بدقة
_, col_search, _ = st.columns([1, 2.6, 1])

with col_search:
    user_query = st.text_input(
        "Search",
        placeholder="اكتب مواصفات العربية واضغط Enter...",
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ==============================================================================
# 4. استخراج النتائج عند الضغط على Enter أو رفع صورة
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
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #bae6fd; padding: 6px 18px; border-radius: 20px; font-size: 0.95rem;">
                    📷 تم التعرف على السيارة: <strong>{top_car_label}</strong> ({preds[0]['confidence']*100:.1f}%)
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

    # التثمين الآمن
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

    _, col_results, _ = st.columns([1, 2.6, 1])
    with col_results:
        st.write("### أفضل النتائج المطابقة:")

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
                badge_html = '<span class="deal-badge-great">🟢 لقطة (سعر ممتاز)</span>'
            elif deal_tag == "Overpriced":
                badge_html = '<span class="deal-badge-overpriced">🔴 سعر مبالغ فيه</span>'
            else:
                badge_html = '<span class="deal-badge-fair">🟡 سعر عادل</span>'

            st.markdown(f"""
            <div class="car-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">
                        {b_name} {m_name} <span style="color: #38bdf8;">{y_val}</span>
                    </span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر المعروض:</span><br>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر العادل المقدر:</span><br>
                        <strong style="color: #38bdf8; font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">المحافظة:</span><br>
                        <span style="color: #cbd5e1; font-size: 1.05rem;">📍 {loc_val}</span>
                    </div>
                    <div>
                        <a href="{item_link}" target="_blank" style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 8px 18px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 0.95rem;">
                            معاينة الإعلان ↗
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 30px;">
        <p style="font-size: 1.05rem;">اكتب مواصفات العربية واضغط <strong>Enter</strong> مباشرة، أو اضغط على أيقونة الكاميرا 📷 لرفع صورة عربية</p>
    </div>
    """, unsafe_allow_html=True)
