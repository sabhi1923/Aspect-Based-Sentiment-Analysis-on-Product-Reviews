import sys
import os

# ============================================================
# BASE PATH — Single source of truth for all file paths
# ============================================================
BASE_PATH = r"C:\\Users\\Lenovo\\Downloads\\asp\\asp"

sys.path.insert(0, BASE_PATH)

from predict import predict_review

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get username from URL
query_params = st.query_params

if "user" in query_params:
    st.session_state.username = query_params["user"]

# ============================================================
# FILE PATHS
# ============================================================
TRAIN_CSV  = f"{BASE_PATH}/data/Laptop_Train_v2.csv"
PRED_FILE  = f"{BASE_PATH}/data/predictions.csv"
INPUT_FILE = f"{BASE_PATH}/data/reviews_input.csv"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ABSA Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #f4f6f9; }

section[data-testid="stSidebar"] {
    background-color: #1e293b;
    color: white;
}
section[data-testid="stSidebar"] .stButton>button {
    background-color: #3b82f6;
    color: white;
    width: 100%;
    font: Times New Roman;
    border-radius: 8px;
    margin-bottom: 6px;
}
section[data-testid="stSidebar"] label {
    color: white !important;
}

.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
.sentiment-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 13px;
}
.badge-positive { background-color: #dcfce7; color: #166534; }
.badge-neutral  { background-color: #fef9c3; color: #854d0e; }
.badge-negative { background-color: #fee2e2; color: #991b1b; }

.aspect-card {
    background: white;
    border-radius: 10px;
    padding: 15px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
            
/* ============================
   INPUT FIELD BORDERS
============================ */

/* Selectbox */
div[data-baseweb="select"] > div {
    border: 2px solid #101213 !important;
    border-radius: 8px !important;
    padding: 2px !important;
}

/* Text Input */
input[type="text"] {
    border: 2px solid #101213 !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Text Area */
textarea {
    border: 2px solid #101213 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* Focus effect (when user clicks) */
input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: #101213 !important;
    box-shadow: 0 0 0 1px #4F46E5 !important;
}            
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

SENTIMENTS = ["positive", "neutral", "negative"]
COLORS = {
    "positive": "#22c55e",
    "neutral":  "#f59e0b",
    "negative": "#ef4444"
}


PRODUCT_CATEGORIES = [
    "-- Select Category --",
    "Laptop",  "Tablet", "Smartphone", "Other"
]

CATEGORY_BRANDS = {
    "Laptop": ["Lenovo", "HP", "Dell", "Asus", "Acer","Apple","Samsung","Microsoft"],
    "Smartphone": ["Apple", "Samsung", "OnePlus", "Xiaomi", "Realme","Vivo"],
    "Tablet": ["Apple", "Samsung", "Lenovo","Redmi","OnePlus","Realme","Xiaomi","Motorola"],
   "Other": ["Other"]
}

# ============================================================
# INIT FILES (create if missing)
# ============================================================
if not os.path.exists(INPUT_FILE):
    pd.DataFrame(
        columns=[
    "product_category", 
    "product",
    "brand",
    "model",
    "review_text"
]
    ).to_csv(INPUT_FILE, index=False)

if not os.path.exists(PRED_FILE):
    pd.DataFrame(
        columns=[
    "product_category",   
    "product",
    "brand",
    "model",
    "review_text",
    "category",
    "sentiment",
    "confidence",
    "positive",
    "neutral",
    "negative"
]
    ).to_csv(PRED_FILE, index=False)

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Review"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Review"

if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

if "username" not in st.session_state:
    st.session_state.username = "Guest"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Aspect Based Sentiment Analysis System")
    st.markdown("---")

    if st.button("📝 Review Submission"):
        st.session_state.page = "Review"

    if st.button("📈 Analytics Dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("🔍 Review History"):
        st.session_state.page = "History"

    # ============================================================
    # PUSH CONTENT DOWN (to place profile at bottom)
    # ============================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # 👤 USER PROFILE
    # ============================================================
    username = st.session_state.get("username", "Guest")

    # Toggle state
    if "show_menu" not in st.session_state:
        st.session_state.show_menu = False

    # --- STYLE ---
    st.markdown("""
    <style>
    div.stButton > button[kind="secondary"] {
        background-color: #111827;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-weight: 600;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #1f2937;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- PROFILE BUTTON ---
    if st.button(f"👤 {username}", key="profile_btn"):
        st.session_state.show_menu = not st.session_state.show_menu

    # --- LOGOUT OPTION ---
    if st.session_state.show_menu:
        if st.button("🚪 Logout", key="logout_unique"):

            st.session_state.clear()

            st.markdown("""
                <meta http-equiv="refresh" content="0; url='http://localhost:5500/ui/login.html'" />
            """, unsafe_allow_html=True)

            st.stop()
    


# ============================================================
# HELPERS
# ============================================================

def sentiment_badge(label: str) -> str:
    css_class = f"badge-{label}"
    icon = (
        "✅" if label == "positive"
        else ("⚠️" if label == "neutral" else "❌")
    )
    return (
        f'<span class="sentiment-badge {css_class}">'
        f'{icon} {label.capitalize()}</span>'
    )


def get_dominant(scores: dict) -> tuple:
    label = max(scores, key=lambda k: scores[k])
    return label, scores[label]


def save_prediction(
    product: str, brand: str, model_name: str,
    product_category: str,
    review: str, category_results: dict, overall_result: dict
):
    """Save one clean row per category to predictions.csv."""

    # ---- Save raw review ---- #
    new_input = pd.DataFrame([{
        "product":     product,
        "brand":       brand,
        "model":       model_name,
        "review_text": review,
        "product_category": product_category
    }])
    input_df = pd.read_csv(INPUT_FILE)
    input_df = pd.concat([input_df, new_input], ignore_index=True)
    input_df.to_csv(INPUT_FILE, index=False)

    # ---- Save one row per aspect category ---- #
    rows = []

    for category, scores in category_results.items():
        label, conf = get_dominant(scores)
        rows.append({
            "product_category": product_category, 
            "product":     product,
            "brand":       brand,
            "model":       model_name,
            "review_text": review,
            "category":    category,
            "sentiment":   label,
            "confidence":  round(conf, 2),
            "positive":    scores.get("positive", 0.0),
            "neutral":     scores.get("neutral",  0.0),
            "negative":    scores.get("negative", 0.0),
        })

    # ---- Save overall row ---- #
    overall_label, overall_conf = get_dominant(overall_result)
    rows.append({
        "product_category": product_category, 
        "product":     product,
        "brand":       brand,
        "model":       model_name,
        "review_text": review,
        "category":    "Overall",
        "sentiment":   overall_label,
        "confidence":  round(overall_conf, 2),
        "positive":    overall_result.get("positive", 0.0),
        "neutral":     overall_result.get("neutral",  0.0),
        "negative":    overall_result.get("negative", 0.0),
    })

    pred_df = pd.read_csv(PRED_FILE)
    pred_df = pd.concat([pred_df, pd.DataFrame(rows)], ignore_index=True)
    pred_df.to_csv(PRED_FILE, index=False)


# ============================================================
# PAGE: REVIEW SUBMISSION
# ============================================================
if st.session_state.page == "Review":

    st.title("📝 Submit a Product Review")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        product_category = st.selectbox(
            "Category",
            PRODUCT_CATEGORIES,
            index=0
        )

    with col2:
        if product_category != "-- Select Category --":
            filtered_brands = CATEGORY_BRANDS.get(product_category, [])
        else:
            filtered_brands = []

        brand = st.selectbox(
            "Brand",
            ["-- Select Brand --"] + filtered_brands,
            index=0
        )

    with col3:
        model_name = st.text_input(
            "Model name",
            placeholder="e.g. IdeaPad 5, MacBook Pro"
        )

    st.markdown("---")
    review = st.text_area(
        "Enter the Review here",
        height=140,
        placeholder=(
            "e.g. The camera quality is great but "
            "battery drains too fast and keyboard is uncomfortable..."
        )
    )

    
    predict_btn = st.button("🔍 Analyze Review", use_container_width=False)

    if predict_btn:

        # ---- Validation ---- #
        if brand == "-- Select Brand --":
            st.warning("⚠️ Please select a Brand.")
            st.stop()

        if product_category == "-- Select Category --":
            st.warning("⚠️ Please select a Product Category.")
            st.stop()
        if not model_name.strip():
            st.warning("⚠️ Please enter a Model name.")
            st.stop() 

        if not review.strip():
            st.warning("⚠️ Please enter a review before submitting.")
            st.stop()

        product_label = f"{brand} {model_name}".strip()

        with st.spinner("🤖 Analyzing with BERT ABSA model..."):
            category_results, overall_result = predict_review(review)



        # Save to disk
        save_prediction(
            product_label, brand, model_name,
            product_category,
            review, category_results, overall_result
)

        # Store in session
        st.session_state.last_prediction = {
            "review":           review,
            "brand":            brand,
            "model_name":       model_name,
            "category_results": category_results,
            "overall_result":   overall_result
        }

        st.success("✅ Analysis complete!")
        st.markdown("---")

        # ---- Overall ---- #
        overall_label, overall_conf = get_dominant(overall_result)

        st.subheader("🎯 Overall Sentiment")
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Overall Verdict",  overall_label.capitalize())
        oc2.metric("Confidence",       f"{overall_conf:.1f}%")
        oc3.metric("Aspects Detected", len(category_results))

        st.markdown("---")

        # ---- Aspect Cards ---- #
        if category_results:
            st.subheader("🧩 Aspect-Level Breakdown")

            for cat, scores in category_results.items():
                label, conf = get_dominant(scores)

                bar_html = ""
                for s in SENTIMENTS:
                    width = scores[s]
                    color = COLORS[s]
                    if width > 0:
                        bar_html += (
                            f'<div style="display:inline-block;'
                            f'width:{width:.1f}%;height:10px;'
                            f'background:{color};border-radius:5px;'
                            f'margin-right:2px;" '
                            f'title="{s}: {scores[s]}%"></div>'
                        )

                badge = sentiment_badge(label)

                st.markdown(f"""
                <div class="aspect-card">
                    <div>
                        <strong style="font-size:15px;">{cat}</strong><br>
                        <div style="margin-top:6px;">{bar_html}</div>
                        <small style="color:#6b7280;">
                            ✅ {scores['positive']}%&nbsp;
                            ⚠️ {scores['neutral']}%&nbsp;
                            ❌ {scores['negative']}%
                        </small>
                    </div>
                    <div>
                        {badge}<br>
                        <small style="color:#6b7280;">
                            {conf:.1f}% confidence
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info(
                "No specific aspects detected. "
                "Overall sentiment model was used."
            )

        st.markdown("---")
        if st.button("📈 View Full Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()


# ============================================================
# PAGE: ANALYTICS DASHBOARD
# ============================================================
elif st.session_state.page == "Dashboard":

    st.title("📈 Analytics Dashboard")

    if not os.path.exists(PRED_FILE):
        st.info("No predictions file found at: " + PRED_FILE)
        st.stop()

    df = pd.read_csv(PRED_FILE)

    if df.empty:
        st.info("No predictions available yet. Submit a review first.")
        st.stop()

    # ---- Product Filter ---- #
    # --- Category Filter ---
    categories = ["All"] + sorted(df["product_category"].dropna().unique().tolist())
    selected_category = st.selectbox("Select Category", categories)

    # --- Brand Filter ---
    if selected_category != "All":
        filtered_brands = CATEGORY_BRANDS.get(selected_category, [])
    else:
        filtered_brands = sorted(df["brand"].dropna().unique().tolist())

    selected_brand = st.selectbox("Select Brand", ["All"] + filtered_brands)

    # --- Apply filters ---
    filtered_df = df.copy()

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["product_category"] == selected_category]

    if selected_brand != "All":
        filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

    # --- Product filter ---
    all_products = filtered_df["product"].dropna().unique().tolist()

    if not all_products:
        st.warning("No products found for selected filters.")
        st.stop()

    selected_product = st.selectbox("Select Product", all_products)

    #----Use filtered Data----
    product_df = filtered_df[filtered_df["product"] == selected_product].copy()
    if product_df.empty:
        st.info("No data for selected product.")
        st.stop()

    st.markdown(f"### Results for **{selected_product}**")
    st.markdown("---")

    # ============================================================
    # OVERALL SENTIMENT SECTION
    # ============================================================
    st.header("🎯 Overall Sentiment Summary")

    overall_df    = product_df[product_df["category"] == "Overall"].copy()
    total_reviews = overall_df["review_text"].nunique()

    review_sentiments = []
    for review_text, grp in overall_df.groupby("review_text"):

        avg_pos = grp["positive"].mean()
        avg_neu = grp["neutral"].mean()
        avg_neg = grp["negative"].mean()

        review_sentiments.append({
            "review_text": review_text,
            "positive": avg_pos,
            "neutral":  avg_neu,
            "negative": avg_neg
        })

    sent_df     = pd.DataFrame(review_sentiments)
    total_reviews = len(sent_df)

   # --- dominant sentiment (for charts only) ---
    sent_df["dominant"] = sent_df[["positive", "neutral", "negative"]].idxmax(axis=1)
    sent_counts = sent_df["dominant"].value_counts().to_dict()

    # --- percentage (this is what you display) ---
    pos_pct = round(sent_df["positive"].mean(), 1)
    neu_pct = round(sent_df["neutral"].mean(), 1)
    neg_pct = round(sent_df["negative"].mean(), 1)

    # --- confidence ---
    avg_conf = round(overall_df["confidence"].mean(), 1) if not overall_df.empty else 0

    # --- UI ---
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Reviews", total_reviews)
    c2.metric("✅ Positive", f"{pos_pct}%")
    c3.metric("⚠️ Neutral",  f"{neu_pct}%")
    c4.metric("❌ Negative", f"{neg_pct}%")
    c5.metric("Avg Confidence", f"{avg_conf}%")

    st.markdown("---")
    # Center the pie chart
    st.subheader("Sentiment Distribution")
    left, center, right = st.columns([1,2,1])
    
    with center:
        labels_pie = ["Positive", "Neutral", "Negative"]

        values_pie = [
            sent_df["positive"].mean(),
            sent_df["neutral"].mean(),
            sent_df["negative"].mean()
        ]

        colors_pie = [
            COLORS["positive"],
            COLORS["neutral"],
            COLORS["negative"]
        ]

        if sum(values_pie) > 0:
            fig1, ax1 = plt.subplots(figsize=(5,5))

            ax1.pie(
                values_pie,
                labels=labels_pie,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors_pie,
                wedgeprops={"edgecolor": "white", "linewidth": 2}
            )

            ax1.set_title(selected_product, fontsize=12)

            st.pyplot(fig1)
                    
 

    st.markdown("---")

    # ============================================================
    # ASPECT SECTION
    # ============================================================
    st.header("🧩 Aspect-Level Sentiment Analysis")

    aspect_df = product_df[product_df["category"] != "Overall"].copy()

    if aspect_df.empty:
        st.info("No aspect-level data found.")
    else:
        pivot = (
            aspect_df
            .groupby("category")[["positive", "neutral", "negative"]]
            .mean()
            .round(2)
        )

        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)

      
        # ---- Raw Data Table ---- #
        st.subheader("Aspect Data Table")
        st.dataframe(
            pivot_pct.style
            .background_gradient(subset=["positive"], cmap="Greens")
            .background_gradient(subset=["negative"], cmap="Reds")
            .background_gradient(subset=["neutral"],  cmap="YlOrBr")
            .format("{:.1f}%"),
            use_container_width=True
        )

        st.markdown("---")

        # ---- Stacked Bar Chart ---- #
        st.subheader("Aspect-wise Sentiment (Stacked Bar)")

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bottoms   = np.zeros(len(pivot_pct))
        x         = range(len(pivot_pct))

        for s in SENTIMENTS:
            ax3.bar(
                x,
                pivot_pct[s],
                bottom=bottoms,
                color=COLORS[s],
                label=s.capitalize(),
                edgecolor="white"
            )
            bottoms += pivot_pct[s].values

        ax3.set_xticks(list(x))
        ax3.set_xticklabels(
            pivot_pct.index, rotation=30, ha="right", fontsize=10
        )
        ax3.set_ylabel("Confidence %")
        ax3.set_xlabel("Aspect Category")
        ax3.set_title(f"Aspect Sentiment Breakdown — {selected_product}")
        ax3.legend(loc="upper right")
        ax3.set_ylim(0, 110)
        st.pyplot(fig3)

        st.markdown("---")

          # ---- Aspect Summary Cards ---- #
        st.subheader("Aspect Summary Cards")

        for cat in pivot_pct.index:
            scores = {
                "positive": pivot_pct.loc[cat, "positive"],
                "neutral":  pivot_pct.loc[cat, "neutral"],
                "negative": pivot_pct.loc[cat, "negative"],
            }
            label, conf = get_dominant(scores)

            bar_html = ""
            for s in SENTIMENTS:
                width = scores[s]
                color = COLORS[s]
                if width > 0:
                    bar_html += (
                        f'<div style="display:inline-block;'
                        f'width:{width:.1f}%;height:14px;'
                        f'background:{color};border-radius:4px;'
                        f'margin-right:2px;" '
                        f'title="{s}: {scores[s]:.1f}%"></div>'
                    )

            badge = sentiment_badge(label)

            st.markdown(f"""
            <div class="aspect-card">
                <div style="flex:1">
                    <strong style="font-size:15px;">{cat}</strong><br>
                    <div style="margin-top:6px;width:100%">{bar_html}</div>
                    <small style="color:#6b7280;">
                        ✅ {scores['positive']:.1f}%&nbsp;&nbsp;&nbsp;
                        ⚠️ {scores['neutral']:.1f}%&nbsp;&nbsp;&nbsp;
                        ❌ {scores['negative']:.1f}%
                    </small>
                </div>
                <div style="text-align:right">
                    {badge}<br>
                    <small style="color:#6b7280;">
                        {conf:.1f}% confidence
                    </small>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

 # ---- Recent Reviews ---- #
    st.header("📋 Recent Reviews for This Product")

    if os.path.exists(INPUT_FILE):
        input_df      = pd.read_csv(INPUT_FILE)
        product_input = input_df[
            input_df["product"] == selected_product
        ][["review_text"]].copy()

        if not product_input.empty:
            product_input.index = range(1, len(product_input) + 1)
            st.dataframe(product_input, use_container_width=True)
        else:
            st.info("No reviews stored yet for this product.")


# ============================================================
# PAGE: REVIEW HISTORY
# ============================================================
elif st.session_state.page == "History":

    st.title("🔍 Full Review History")

    if not os.path.exists(PRED_FILE):
        st.info("No predictions file found.")
        st.stop()

    df = pd.read_csv(PRED_FILE)

    if df.empty:
        st.info("No prediction history yet. Submit a review first.")
        st.stop()

    # ---- Filters ---- #
    col1, col2, col3 = st.columns(3)

    with col1:
        brands = ["All"] + sorted(
            df["brand"].dropna().astype(str).unique().tolist()
        )
        selected_brand = st.selectbox("Filter by Brand", brands)

    with col2:
        categories = ["All"] + sorted([
            c for c in
            df["category"].dropna().astype(str).unique().tolist()
        ])
        selected_cat = st.selectbox("Filter by Category", categories)

    with col3:
        sentiments    = ["All", "positive", "neutral", "negative"]
        selected_sent = st.selectbox("Filter by Sentiment", sentiments)

    # ---- Apply Filters ---- #
    filtered = df.copy()

    if selected_brand != "All":
        filtered = filtered[
            filtered["brand"].astype(str) == selected_brand
        ]
    if selected_cat != "All":
        filtered = filtered[
            filtered["category"].astype(str) == selected_cat
        ]
    if selected_sent != "All":
        filtered = filtered[
            filtered["sentiment"].astype(str) == selected_sent
        ]

    # ---- Summary Metrics ---- #
    st.markdown("---")
    pos = len(filtered[filtered["sentiment"] == "positive"])
    neg = len(filtered[filtered["sentiment"] == "negative"])
    neu = len(filtered[filtered["sentiment"] == "neutral"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Entries", len(filtered))
    m2.metric("✅ Positive",   pos)
    m3.metric("⚠️ Neutral",    neu)
    m4.metric("❌ Negative",   neg)

    st.markdown("---")

    # ---- Display Table ---- #
    st.subheader("📋 Prediction Records")

    display_cols = [
        "product", "brand", "review_text",
        "category", "sentiment", "confidence"
    ]
    show_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].reset_index(drop=True),
        use_container_width=True
    )

    # ---- Download ---- #
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="absa_history.csv",
        mime="text/csv"
    )
