import os
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# FEDMED — PROFESSIONAL DASHBOARD
# ============================================================

st.set_page_config(
    page_title="FedMed | Federated Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_model.pth",
)

PREDICTION_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "predictions",
)


# ============================================================
# REAL PROJECT RESULTS
# ============================================================

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "experiment_results.csv",
)


def load_experiment_results():
    """Load verified experiment results from CSV."""

    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()

    return pd.read_csv(RESULTS_PATH)


RESULTS_DF = load_experiment_results()


# Use the latest verified DP-FedAvg experiment
if (
    not RESULTS_DF.empty
    and "experiment" in RESULTS_DF.columns
):
    DP_RESULTS = RESULTS_DF[
        RESULTS_DF["experiment"] == "DP-FedAvg"
    ].copy()
else:
    DP_RESULTS = pd.DataFrame()


if not DP_RESULTS.empty:

    HOSPITAL_DATA = {}

    for _, row in DP_RESULTS.iterrows():

        HOSPITAL_DATA[row["hospital"]] = {
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "status": "Completed",
        }

    AVERAGE_DICE = DP_RESULTS["dice"].mean()
    AVERAGE_IOU = DP_RESULTS["iou"].mean()
    AVERAGE_PRECISION = DP_RESULTS["precision"].mean()
    AVERAGE_RECALL = DP_RESULTS["recall"].mean()

else:

    HOSPITAL_DATA = {}
    AVERAGE_DICE = 0.0
    AVERAGE_IOU = 0.0
    AVERAGE_PRECISION = 0.0
    AVERAGE_RECALL = 0.0


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ============================================================
       FEDMED PROFESSIONAL UI
       ============================================================ */

    /* ---------- Global ---------- */

    .stApp {
        background: #f4f7fb;
        color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Improve default Streamlit text visibility */
    .stMarkdown,
    .stText,
    p,
    label {
        color: #334155;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] > div {
        background: #0b1220;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Sidebar radio navigation */

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.35rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        transition: all 0.2s ease;
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #172033;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #1e3a5f;
        color: #ffffff !important;
        font-weight: 700;
    }

    /* ---------- Brand ---------- */

    .brand {
        font-size: 2.45rem;
        font-weight: 850;
        letter-spacing: -1.5px;
        margin-bottom: 0;
        color: #0f172a !important;
        line-height: 1.05;
    }

    .subtitle {
        color: #64748b !important;
        font-size: 1rem;
        margin-top: 0.35rem;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }

    /* ---------- Section Headers ---------- */

    .section-title {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 1.2rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* ---------- KPI Cards ---------- */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem 1.4rem;
        min-height: 140px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.055);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.09);
    }

    .metric-label {
        color: #64748b !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #0f172a !important;
        font-size: 2.05rem;
        font-weight: 850;
        margin-top: 0.4rem;
        line-height: 1.15;
    }

    .metric-description {
        color: #94a3b8 !important;
        font-size: 0.78rem;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* ---------- Status Cards ---------- */

    .status-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
    }

    .status-title {
        font-weight: 750;
        font-size: 1rem;
        color: #0f172a !important;
    }

    .status-success {
        color: #15803d !important;
        font-weight: 750;
    }

    .status-warning {
        color: #b45309 !important;
        font-weight: 750;
    }

    .status-error {
        color: #b91c1c !important;
        font-weight: 750;
    }

    /* ---------- Hospital Cards ---------- */

    .hospital-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .hospital-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    }

    .hospital-name {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0f172a !important;
    }

    .hospital-status {
        color: #15803d !important;
        font-size: 0.82rem;
        font-weight: 750;
    }

    /* ---------- Architecture ---------- */

    .architecture {
        background: #0f172a;
        color: #e2e8f0 !important;
        border-radius: 18px;
        padding: 1.7rem;
        font-family: monospace;
        line-height: 1.9;
        border: 1px solid #1e293b;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }

    .architecture * {
        color: #e2e8f0 !important;
    }

    /* ---------- Tables ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #cbd5e1;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10);
    }

    /* ---------- Expanders ---------- */

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
    }

    /* ---------- Alerts ---------- */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---------- Images ---------- */

    [data-testid="stImage"] {
        border-radius: 12px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 0.78rem;
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }

    /* ---------- Responsive ---------- */

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .brand {
            font-size: 2rem;
        }

        .metric-value {
            font-size: 1.7rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.6rem;
            font-weight:800;
            color:white;
            margin-bottom:0.2rem;
        ">
            🏥 FedMed
        </div>

        <div style="
            color:#94a3b8;
            font-size:0.85rem;
            margin-bottom:2rem;
        ">
            Federated Medical AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Platform")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Hospital Performance",
            "Segmentation Results",
            "Federated Training",
            "Global Model",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        **System**

        🟢 Federated system ready

        🟢 Global model available

        🟢 3 hospitals participated

        🟢 Evaluation completed
        """
    )

    st.divider()

    st.caption(
        f"FedMed Dashboard\n\n"
        f"Updated: {datetime.now().strftime('%d %b %Y, %H:%M')}"
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="brand">FedMed</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Privacy-preserving federated learning for medical image segmentation'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Platform Overview")

    st.write(
        "Monitor the federated medical AI training pipeline, "
        "global model performance, and segmentation results."
    )

    st.write("")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">PARTICIPATING HOSPITALS</div>
                <div class="metric-value">3</div>
                <div class="metric-description">
                    Independent training sites
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">FEDERATED ROUNDS</div>
                <div class="metric-value">5</div>
                <div class="metric-description">
                    Completed training rounds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AVERAGE DICE</div>
                <div class="metric-value">{AVERAGE_DICE:.4f}</div>
                <div class="metric-description">
                    Global model evaluation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        model_status = (
            "Available"
            if os.path.exists(MODEL_PATH)
            else "Unavailable"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">GLOBAL MODEL</div>
                <div class="metric-value">
                    {"✓" if model_status == "Available" else "!"}
                </div>
                <div class="metric-description">
                    {model_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # System status
    st.subheader("System Status")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">
                    Federated Learning Infrastructure
                </div>
                <br>
                <span class="status-success">
                    ● Operational
                </span>
                <br><br>
                Flower server completed 5 federated rounds
                with all 3 hospitals participating.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">
                    Global Model
                </div>
                <br>
                <span class="status-success">
                    ● Available
                </span>
                <br><br>
                3D U-Net global model is stored and ready
                for evaluation and inference.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Federated Architecture")

    st.markdown(
        """
        <div class="architecture">

        Hospital 1 ───────────────┐<br>
        Hospital 2 ───────────────┼──► Federated Server<br>
        Hospital 3 ───────────────┘           │<br>
        <br>
        Local medical data                    │<br>
        remains at each hospital              ▼<br>
        <br>
                              Client-side Fixed Clipping<br>
                                             │<br>
                                             ▼<br>
                                      FedAvg Aggregation<br>
                                             │<br>
                                             ▼<br>
                                  Central Gaussian DP Noise<br>
                                             │<br>
                                             ▼<br>
                                      Global 3D U-Net<br>
                                             │<br>
                                             ▼<br>
                                  Model Evaluation<br>
                                             │<br>
                                             ▼<br>
                                  Segmentation Results

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOSPITAL PERFORMANCE
# ============================================================

elif page == "Hospital Performance":

    st.header("Hospital Performance")

    st.write(
        "Evaluation of the federated global model across "
        "each participating hospital."
    )

    st.write("")

    data = []

    for hospital, values in HOSPITAL_DATA.items():

        data.append(
            {
                "Hospital": hospital,
                "Dice Score": values["dice"],
                "IoU": values["iou"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "Status": values["status"],
            }
        )

    df = pd.DataFrame(data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Dice Score": st.column_config.NumberColumn(
                "Dice Score",
                format="%.4f",
            ),
            "IoU": st.column_config.NumberColumn(
                "IoU",
                format="%.4f",
            ),
            "Precision": st.column_config.NumberColumn(
                "Precision",
                format="%.4f",
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                format="%.4f",
            ),
        },
    )

    st.write("")

    st.subheader("Dice Score Comparison")

    chart_df = df.set_index("Hospital")[["Dice Score"]]

    st.bar_chart(chart_df)

    st.write("")

    c1, c2, c3 = st.columns(3)

    hospitals = list(HOSPITAL_DATA.keys())

    for column, hospital in zip(
        [c1, c2, c3],
        hospitals,
    ):

        with column:

            values = HOSPITAL_DATA[hospital]

            st.markdown(
                f"""
                <div class="hospital-card">

                <div class="hospital-name">
                    {hospital}
                </div>

                <div class="hospital-status">
                    ● {values["status"]}
                </div>

                <br>

                <div style="
                    font-size:2rem;
                    font-weight:800;
                    color:#0f172a;
                ">
                    {values["dice"]:.4f}
                </div>

                <div style="
                    color:#64748b;
                    font-size:0.82rem;
                ">
                    Dice Score
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# SEGMENTATION RESULTS
# ============================================================

elif page == "Segmentation Results":

    st.header("Segmentation Results")

    st.write(
        "Prediction visualizations generated using the "
        "federated global 3D U-Net model."
    )

    st.write("")

    hospitals = [
        "hospital1",
        "hospital2",
        "hospital3",
    ]

    display_names = {
        "hospital1": "Hospital-1",
        "hospital2": "Hospital-2",
        "hospital3": "Hospital-3",
    }

    cols = st.columns(3)

    for column, hospital in zip(cols, hospitals):

        image_path = os.path.join(
            PREDICTION_DIR,
            f"{hospital}_prediction.png",
        )

        with column:

            st.subheader(
                display_names[hospital]
            )

            if os.path.exists(image_path):

                image = Image.open(image_path)

                st.image(
                    image,
                    caption="Global model segmentation",
                    use_container_width=True,
                )

                score = HOSPITAL_DATA[
                    display_names[hospital]
                ]["dice"]

                st.metric(
                    "Dice",
                    f"{score:.4f}",
                )

            else:

                st.warning(
                    "Prediction image not found."
                )


# ============================================================
# FEDERATED TRAINING
# ============================================================

elif page == "Federated Training":

    st.header("Federated Training")

    st.write(
        "Verified training history for the standard FedAvg "
        "baseline and the privacy-preserving DP-FedAvg experiment."
    )

    st.write("")

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Federated Rounds",
            "5",
        )

    with c2:
        st.metric(
            "Hospitals",
            "3",
        )

    with c3:
        st.metric(
            "DP Noise Multiplier",
            "0.5",
        )

    with c4:
        st.metric(
            "DP Clipping Norm",
            "2.5",
        )

    st.divider()

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    st.subheader("Training Configuration")

    config_df = pd.DataFrame(
        {
            "Parameter": [
                "Baseline Algorithm",
                "Privacy-Preserving Algorithm",
                "Hospitals",
                "Clients per Round",
                "Federated Rounds",
                "Local Epochs",
                "Learning Rate",
                "Model",
                "Framework",
            ],
            "Configuration": [
                "FedAvg",
                "DP-FedAvg",
                "3",
                "3",
                "5",
                "1",
                "0.001",
                "3D U-Net",
                "PyTorch + MONAI",
            ],
        }
    )

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True,
    )

    st.write("")

    # --------------------------------------------------------
    # Verified round-by-round training results
    # --------------------------------------------------------

    st.subheader("Round-by-Round Training Results")

    training_history = pd.DataFrame(
        {
            "Round": [1, 2, 3, 4, 5],

            "FedAvg Loss": [
                0.7552829252,
                0.6478891638,
                0.5751358867,
                0.5184391075,
                0.4676197006,
            ],

            "FedAvg Dice": [
                0.0316425730,
                0.0295521288,
                0.0283128191,
                0.0356543437,
                0.0480200425,
            ],

            "DP-FedAvg Loss": [
                0.8260768851,
                0.6389293538,
                2.7583336035,
                3.3267206351,
                4.4625719918,
            ],

            "DP-FedAvg Dice": [
                0.0305071597,
                0.0313314293,
                0.0305207844,
                0.0310470524,
                0.0312029148,
            ],
        }
    )

    st.dataframe(
        training_history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "FedAvg Loss": st.column_config.NumberColumn(
                "FedAvg Loss",
                format="%.4f",
            ),
            "FedAvg Dice": st.column_config.NumberColumn(
                "FedAvg Dice",
                format="%.4f",
            ),
            "DP-FedAvg Loss": st.column_config.NumberColumn(
                "DP-FedAvg Loss",
                format="%.4f",
            ),
            "DP-FedAvg Dice": st.column_config.NumberColumn(
                "DP-FedAvg Dice",
                format="%.4f",
            ),
        },
    )

    st.write("")

    # --------------------------------------------------------
    # Training loss comparison
    # --------------------------------------------------------

    st.subheader("Training Loss Comparison")

    loss_chart = training_history.set_index("Round")[
        [
            "FedAvg Loss",
            "DP-FedAvg Loss",
        ]
    ]

    st.line_chart(loss_chart)

    st.write("")

    # --------------------------------------------------------
    # Dice comparison
    # --------------------------------------------------------

    st.subheader("Dice Score Comparison")

    dice_chart = training_history.set_index("Round")[
        [
            "FedAvg Dice",
            "DP-FedAvg Dice",
        ]
    ]

    st.line_chart(dice_chart)

    st.write("")

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    st.subheader("Experiment Summary")

    summary_df = pd.DataFrame(
        {
            "Experiment": [
                "FedAvg",
                "DP-FedAvg",
            ],
            "Final Train Loss": [
                0.4676197006,
                4.4625719918,
            ],
            "Final Dice": [
                0.0480200425,
                0.0312029148,
            ],
            "Noise Multiplier": [
                0.0,
                0.5,
            ],
            "Clipping Norm": [
                0.0,
                2.5,
            ],
        }
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Train Loss": st.column_config.NumberColumn(
                "Final Train Loss",
                format="%.4f",
            ),
            "Final Dice": st.column_config.NumberColumn(
                "Final Dice",
                format="%.4f",
            ),
            "Noise Multiplier": st.column_config.NumberColumn(
                "Noise Multiplier",
                format="%.2f",
            ),
            "Clipping Norm": st.column_config.NumberColumn(
                "Clipping Norm",
                format="%.2f",
            ),
        },
    )

# ============================================================
# GLOBAL MODEL
# ============================================================

elif page == "Global Model":

    st.header("Global Model")

    st.write(
        "Information about the aggregated federated model."
    )

    st.write("")

    if os.path.exists(MODEL_PATH):

        model_size = (
            os.path.getsize(MODEL_PATH)
            / (1024 * 1024)
        )

        st.success(
            "Global model successfully loaded."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Architecture",
                "3D U-Net",
            )

        with c2:
            st.metric(
                "Training Rounds",
                "5",
            )

        with c3:
            st.metric(
                "Model Size",
                f"{model_size:.2f} MB",
            )

        st.divider()

        st.subheader("Model Details")

        details = pd.DataFrame(
                {
                    "Property": [
                        "Model",
                        "Framework",
                        "Architecture",
                        "Input Channels",
                        "Output Classes",
                        "Spatial Dimensions",
                        "Federated Algorithm",
                        "Privacy Mechanism",
                        "Noise Multiplier",
                        "Clipping Norm",
                        "Model Status",
                    ],
                    "Value": [
                        "Global 3D U-Net",
                        "PyTorch + MONAI",
                        "U-Net",
                        "1",
                        "2",
                        "3D",
                        "DP-FedAvg",
                        "Client-side fixed clipping + central Gaussian noise",
                        "0.5",
                        "2.5",
                        "Available",
                    ],
                }
            )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.subheader("Model Artifact")

        st.code(
            "models/global_model.pth"
        )

        st.caption(
            "The global model contains the aggregated parameters "
            "produced through DP-FedAvg with client-side fixed clipping "
            "and central Gaussian noise."
        )

    else:

        st.error(
            "Global model not found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    FedMed · Federated Medical AI Platform<br>

    Privacy-preserving collaborative learning ·
    3D Medical Image Segmentation

    </div>
    """,
    unsafe_allow_html=True,
)