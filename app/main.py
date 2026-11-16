
"""Streamlit application for Sports Commentary Generator."""

import streamlit as st

from core import generate_commentary, load_trained_model

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sports Commentary Generator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #111827;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.result-card {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 25px;
    margin-top: 15px;
    color: #ffffff;
    font-size: 20px;
    line-height: 1.7;
}

.metric-card {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.stButton > button {
    background: #22c55e;
    color: #07111f;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover {
    background: #16a34a;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_trained_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🏆 Sports AI")
    st.caption("LoRA-powered Commentary Engine")

    st.divider()

    st.subheader("Model Information")
    st.write("Base Model: GPT-2")
    st.write("Fine-tuning: LoRA")
    st.write("Training Records: 200")
    st.write("Sports: Football & Cricket")

    st.divider()

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Sports Commentary Generator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Transform simulated sports events into dynamic AI-generated commentary.</div>',
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------
# EVENT INPUT
# --------------------------------------------------

st.subheader("⚡ Configure Match Event")

col1, col2 = st.columns(2)

with col1:
    sport = st.selectbox(
        "Select Sport",
        ["Football", "Cricket"],
    )

with col2:
    if sport == "Football":
        event_types = [
            "goal", "penalty", "save", "yellow_card",
            "red_card", "corner", "free_kick",
            "substitution", "missed_chance", "offside"
        ]
    else:
        event_types = [
            "six", "four", "wicket", "dot_ball",
            "single", "double", "wide", "no_ball",
            "fifty", "century"
        ]

    event_type = st.selectbox(
        "Event Type",
        event_types,
    )

col3, col4 = st.columns(2)

with col3:
    player = st.text_input("Player Name", placeholder="e.g. Rohit")

with col4:
    team = st.text_input(
        "Team Name",
        placeholder="e.g. India / Blue FC",
    )

match_context = st.text_input(
    "Match Context",
    placeholder="e.g. 15th over, score 120-2",
)

description = st.text_area(
    "Describe the Event",
    placeholder="Describe what happened during the match...",
    height=120,
)

# --------------------------------------------------
# GENERATION
# --------------------------------------------------

if st.button("⚡ Generate Commentary", use_container_width=True):

    if not all([
        player.strip(),
        team.strip(),
        match_context.strip(),
        description.strip(),
    ]):
        st.warning("Please complete all event details.")

    else:
        try:
            with st.spinner("Loading AI model and generating commentary..."):

                model, tokenizer = get_model()

                record = {
                    "sport": sport.lower(),
                    "event_type": event_type,
                    "description": description.strip(),
                    "team": team.strip(),
                    "player": player.strip(),
                    "match_context": match_context.strip(),
                }

                commentary = generate_commentary(
                    record,
                    model,
                    tokenizer,
                )

            if not commentary:
                st.warning("The model returned empty commentary.")
            else:
                st.session_state.history.insert(
                    0,
                    {
                        "sport": sport,
                        "event": event_type,
                        "player": player,
                        "commentary": commentary,
                    },
                )

                st.success("Commentary generated successfully!")

                st.subheader("🎙️ Generated Commentary")

                st.markdown(
                    f'<div class="result-card">{commentary}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as error:
            st.error(f"Generation failed: {error}")

# --------------------------------------------------
# GENERATION HISTORY
# --------------------------------------------------

st.divider()
st.subheader("📜 Recent Commentary")

if not st.session_state.history:
    st.info("Your generated commentary will appear here.")
else:
    for item in st.session_state.history[:5]:
        with st.expander(
            f"{item['sport']} | {item['event']} | {item['player']}"
        ):
            st.write(item["commentary"])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Sports Commentary Generator | GPT-2 + LoRA | "
    "Academic AI/ML Project"
)
