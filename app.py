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
    image_path = "hero_car.jpg.jpeg"
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

BG_IMAGE = get_background_image()

# ==============================================================================
# CSS: ضبط الألوان وتثبيت زر الكاميرا داخل شريط البحث
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

    .stApp {{
        background:
            linear-gradient(180deg, rgba(2, 5, 9, 0.65) 0%, rgba(2, 5, 9, 0.82) 48%, rgba(2, 5, 9, 0.96) 100%),
            url("{BG_IMAGE}") center center / cover fixed no-repeat !important;
        color: #f8fafc;
        font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }}

    .hero-box {{
        text-align: center;
        padding: 40px 15px 25px 15px;
        position: relative;
    }}

    .hero-kicker {{
        display: inline-block;
        padding: 6px 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 999px;
        background: rgba(8, 15, 27, 0.65);
        backdrop-filter: blur(12px);
        color: #7dd3fc;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }}

    /* اسم المشروع: درجة أغمق متناسقة مع لون الإضاءة في الخلفية */
    .hero-title {{
        font-size: clamp(2.3rem, 5vw, 3.8rem);
        line-height: 1.1;
        font-weight: 800;
        background: linear-gradient(180deg, #93c5fd 0%, #38bdf8 60%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        text-shadow: 0 4px 20px rgba(2, 132, 199, 0.25);
    }}

    .hero-subtitle {{
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.8;
    }}

    /* حاوية البحث الخارجية */
    [data-testid="stTextInput"] {{
        position: relative !important;
        z-index: 5 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 40px !important;
        background: rgba(15, 23, 42, 0.88) !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5) !important;
        backdrop-filter: blur(16px);
        padding: 0 !important;
        height: 60px !important;
    }}

    [data-testid="stTextInput"] > div {{
        height: 100% !important;
        border: none !important;
        background: transparent !important;
    }}

    [data-testid="stTextInput"] input {{
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
        height: 60px !important;
        line-height: 60px !important;
        padding-left: 20px !important;
        padding-right: 68px !important;
        font-size: 1.05rem !important;
        direction: rtl;
        text-align: right;
    }}

    [data-testid="stTextInput"] input:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    /* تثبيت زرار الكاميرا في السنتر الرأسي تماماً جوه البار */
    [data-testid="stFileUploader"] {{
        margin-top: -60px !important;
        height: 60px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-right: 12px !important;
        pointer-events: none !important;
        border: none !important;
        background: transparent !important;
        position: relative !important;
        z-index: 10 !important;
    }}

    [data-testid="stFileUploader"] section {{
        padding: 0 !important;
        min-height: unset !important;
        border: none !important;
        background: transparent !important;
        pointer-events: auto !important;
        display: flex !important;
        align-items: center !important;
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
        background: rgba(56, 189, 248, 0.15) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stFileUploader"] button:hover {{
        transform: scale(1.08) !important;
        background: rgba(56, 189, 248, 0.3) !important;
        border-color: #38bdf8 !important;
    }}

    [data-testid="stFileUploader"] button::before {{
        content: "📷";
        font-size: 1.15rem;
        line-height: 1;
    }}

    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {{
        display: none !important;
    }}

    /* كروت تفاصيل السيارات */
    .car-card {{
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        transition: all 0.25s ease;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(10px);
    }}

    .car-card:hover {{
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-3px);
    }}

    .deal-badge-great {{
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-overpriced {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-fair {{
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #facc15;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
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
    st.error(f"⚠️ خطأ في تحميل ملفات المشروع: {e}")
    st.stop()

# ==============================================================================
# 2. قواميس المعالجة الدلالية للبحث
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
    "مانيوال": "Manual", "manual": "Manual", "عادي": "Manual"
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
    
    # 1. الميزانية
    min_p, max_p = extract_budget(text)
    if min_p is not None: filters["min_price"] = min_p
    if max_p is not None: filters["max_price"] = max_p

    # 2. ناقل الحركة
    for k, v in TRANS_MAP.items():
        if k in text:
            filters["transmission"] = v
            break

    # 3. الحالة
    for k, v in CONDITION_MAP.items():
        if k in text:
            filters["car_condition"] = v
            break

    # 4. الماركة
    if "brand" in catalog_df.columns:
        brands = [b for b in catalog_df["brand"].dropna().unique() if str(b).strip()]
        for b in sorted(brands, key=len, reverse=True):
            if str(b).lower() in text:
                filters["brand"] = b
                break

    # 5. المكان
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
    <div class="hero-kicker">🚘 SMART CAR MARKET</div>
    <div class="hero-title">Apex Motors</div>
    <div class="hero-subtitle">محرك البحث الذكي وتقييم أسعار السيارات في السوق المصري</div>
</div>
""", unsafe_allow_html=True)

_, col_search, _ = st.columns([1, 2.6, 1])

with col_search:
    user_query = st.text_input(
        "Search",
        placeholder="مثال: كيا سبورتاج اوتوماتيك في القاهرة تحت 2 مليون...",
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ==============================================================================
# 4. استخراج ومعالجة النتائج
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

    # التثمين والتنبؤ بالسعر العادل
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
        st.write("### أفضل النتائج المطابقة:")

        for _, row in top_results.iterrows():
            b_name = row.get("brand", "")
            m_name = row.get("model", "")
            y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
            price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
            fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
            deal_tag = row.get("deal_label", "Fair Price")
            diff_pct = abs(row.get("pct_diff", 0))
            loc_val = row.get("location", "مصر")
            trans_val = row.get("transmission", "-")
            mileage_val = f"{int(row.get('mileage', 0)):,} كم" if pd.notna(row.get("mileage")) and row.get("mileage") > 0 else "زيرو"
            item_link = row.get("item_url", "#")

            if deal_tag == "Great Deal":
                badge_html = f'<span class="deal-badge-great">🟢 لقطة (توفير {diff_pct:.0f}%)</span>'
            elif deal_tag == "Overpriced":
                badge_html = f'<span class="deal-badge-overpriced">🔴 زائد بنسبة {diff_pct:.0f}%</span>'
            else:
                badge_html = '<span class="deal-badge-fair">🟡 سعر عادل ومناسب</span>'

            st.markdown(f"""
            <div class="car-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                        {b_name} {m_name} <span style="color: #38bdf8;">{y_val}</span>
                    </span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; gap: 15px; margin-top: 6px; color: #94a3b8; font-size: 0.9rem;">
                    <span>🕹️ {trans_val}</span>
                    <span>🛣️ {mileage_val}</span>
                    <span>📍 {loc_val}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 14px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر المعروض:</span><br>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر العادل المقدر:</span><br>
                        <strong style="color: #38bdf8; font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
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
    <div style="text-align: center; color: #64748b; margin-top: 35px;">
        <p style="font-size: 1.05rem;">اكتب مواصفات العربية واضغط <strong>Enter</strong> مباشرة، أو اضغط على أيقونة الكاميرا 📷 لرفع صورة سيارة</p>
    </div>
    """, unsafe_allow_html=True)
