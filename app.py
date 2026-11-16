
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🎧",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #808080;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 23px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 12px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.35);
    margin-bottom: 15px;
}

.result-label {
    font-size: 14px;
    color: #808080;
}

.result-value {
    font-size: 22px;
    font-weight: 600;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #808080;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. PROJECT FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "ticket_type_model.pkl"
VECTORIZER_PATH = DATA_DIR / "ticket_type_tfidf.pkl"
KNOWLEDGE_BASE_PATH = DATA_DIR / "resolution_knowledge_base.csv"


# --------------------------------------------------
# 4. LOAD MODEL AND KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource
def load_model_and_vectorizer():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


@st.cache_data
def load_knowledge_base():

    return pd.read_csv(KNOWLEDGE_BASE_PATH)


try:

    model, tfidf = load_model_and_vectorizer()

    resolution_df = load_knowledge_base()

    problem_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    problem_vectors = problem_vectorizer.fit_transform(
        resolution_df["Problem"].fillna("")
    )

except Exception as e:

    st.error(f"Could not load project files: {e}")
    st.stop()


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# --------------------------------------------------
# 6. EXAMPLE TICKETS AND RESET FUNCTIONS
# --------------------------------------------------

example_tickets = {

    "💳 Billing": (
        "I paid my subscription twice, but the amount "
        "was deducted two times from my bank account."
    ),

    "❌ Cancellation": (
        "I want to cancel my subscription and stop "
        "the automatic renewal."
    ),

    "🔧 Technical": (
        "My WiFi connection keeps disconnecting "
        "every few minutes."
    ),

    "📦 Product": (
        "I would like to know whether this device "
        "is compatible with my laptop."
    ),

    "💰 Refund": (
        "I received a damaged product and would "
        "like to get my money refunded."
    )
}


def set_example_ticket(ticket):

    st.session_state.ticket_input = ticket

    # Clear previous analysis when selecting another example
    st.session_state.analysis_result = None


def reset_application():

    st.session_state.ticket_input = ""

    st.session_state.analysis_result = None


# --------------------------------------------------
# 7. APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎧 AI Customer Support Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Intelligent ticket classification and resolution recommendation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 8. TICKET INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Submit Customer Ticket</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the customer's issue below or select "
    "one of the example tickets."
)

st.markdown("**Try an example ticket:**")

example_items = list(example_tickets.items())

# First row: three example buttons
cols = st.columns(3)

for i in range(3):

    label, ticket = example_items[i]

    with cols[i]:

        st.button(
            label,
            key=f"example_{i}",
            use_container_width=True,
            on_click=set_example_ticket,
            args=(ticket,)
        )


# Second row: two example buttons
cols = st.columns(2)

for i in range(3, 5):

    label, ticket = example_items[i]

    with cols[i - 3]:

        st.button(
            label,
            key=f"example_{i}",
            use_container_width=True,
            on_click=set_example_ticket,
            args=(ticket,)
        )


st.divider()


# Customer ticket input
ticket_text = st.text_area(
    "Customer Ticket Description",
    placeholder="Enter the customer's issue here...",
    height=150,
    key="ticket_input"
)

st.caption(
    "Tip: Describe the customer's problem clearly "
    "for better classification and matching."
)


# Analyze and Reset buttons
col1, col2 = st.columns(2)

with col1:

    analyze_button = st.button(
        "🔍 Analyze Ticket",
        type="primary",
        use_container_width=True
    )

with col2:

    st.button(
        "🔄 Reset",
        use_container_width=True,
        on_click=reset_application
    )


# --------------------------------------------------
# 9. ANALYZE CUSTOMER TICKET
# --------------------------------------------------

if analyze_button:

    if not ticket_text.strip():

        st.warning(
            "Please enter a customer ticket description."
        )

    else:

        with st.spinner("Analyzing customer ticket..."):

            try:

                # Convert ticket text into TF-IDF features
                ticket_vector = tfidf.transform([ticket_text])

                # Predict ticket category
                predicted_category = model.predict(
                    ticket_vector
                )[0]

                # Calculate classification confidence
                probabilities = model.predict_proba(
                    ticket_vector
                )[0]

                predicted_index = list(model.classes_).index(
                    predicted_category
                )

                classification_confidence = probabilities[
                    predicted_index
                ]

                # Find matching knowledge-base category
                category_df = resolution_df[
                    resolution_df["Ticket Type"] == predicted_category
                ].copy()

                if category_df.empty:

                    st.session_state.analysis_result = {
                        "error": (
                            "No resolution entries are available "
                            "for this category. Please review manually."
                        ),
                        "predicted_category": predicted_category,
                        "confidence": classification_confidence
                    }

                else:

                    # Convert ticket text for problem matching
                    problem_vector = problem_vectorizer.transform(
                        [ticket_text]
                    )

                    category_indices = category_df.index.tolist()

                    category_vectors = problem_vectors[
                        category_indices
                    ]

                    # Calculate text similarity
                    similarities = cosine_similarity(
                        problem_vector,
                        category_vectors
                    ).flatten()

                    best_position = similarities.argmax()

                    best_index = category_indices[best_position]

                    best_match = resolution_df.loc[best_index]

                    similarity_score = similarities[best_position]

                    # Save analysis results
                    st.session_state.analysis_result = {

                        "predicted_category": predicted_category,

                        "confidence": classification_confidence,

                        "problem": str(best_match["Problem"]),

                        "similarity": float(similarity_score),

                        "resolution": str(
                            best_match["Resolution Steps"]
                        ),

                        "escalation": str(
                            best_match["Escalation Required"]
                        )
                    }

            except Exception as e:

                st.error(f"Ticket analysis failed: {e}")


# --------------------------------------------------
# 10. DISPLAY ANALYSIS RESULTS
# --------------------------------------------------

result = st.session_state.analysis_result

if result is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Analysis Results</div>',
        unsafe_allow_html=True
    )

    # Display errors, if any
    if "error" in result:

        st.metric(
            "Predicted Ticket Category",
            result["predicted_category"]
        )

        st.metric(
            "Classification Confidence",
            f'{result["confidence"]:.2%}'
        )

        st.warning(result["error"])

    else:

        # Predicted category and confidence
        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTED TICKET CATEGORY
                    </div>
                    <div class="result-value">
                        {result["predicted_category"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        CLASSIFICATION CONFIDENCE
                    </div>
                    <div class="result-value">
                        {result["confidence"]:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            float(result["confidence"]),
            text="Model classification confidence (estimated probability)"
        )

        # Best matching problem and similarity
        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        BEST MATCHING PROBLEM
                    </div>
                    <div class="result-value">
                        {result["problem"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        TEXT SIMILARITY SCORE
                    </div>
                    <div class="result-value">
                        {result["similarity"]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            float(result["similarity"]),
            text="Knowledge-base text similarity (not classification confidence)"
        )

        # Suggested resolution
        st.divider()

        st.markdown(
            '<div class="section-title">💡 Suggested Resolution</div>',
            unsafe_allow_html=True
        )

        st.success(result["resolution"])

        # Escalation guidance
        st.markdown(
            '<div class="section-title">🚨 Escalation Guidance</div>',
            unsafe_allow_html=True
        )

        st.info(result["escalation"])

        st.caption(
            "These recommendations come from the project's "
            "sample knowledge base. Review them before taking action."
        )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI Customer Support Assistant |
        Built with Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
