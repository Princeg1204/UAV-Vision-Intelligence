# ================================================================
# UAV VISION INTELLIGENCE
# Premium Streamlit Deployment
# Baseline YOLO26-L | VisDrone2019
# ================================================================

from pathlib import Path
from collections import Counter
from io import BytesIO
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from ultralytics import YOLO


# ================================================================
# 1. PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="UAV Vision Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# 2. PATHS
# ================================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"


# ================================================================
# 3. PROJECT CONSTANTS
# ================================================================

CLASS_NAMES = {
    0: "pedestrian",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


# Header / project imagery
HERO_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1527977966376-1c8408f9f108"
    "?auto=format&fit=crop&w=1500&q=90"
)

PROJECT_IMAGE_1 = (
    "https://images.unsplash.com/"
    "photo-1676036633191-ca2c6aac0b63"
    "?auto=format&fit=crop&w=1400&q=90"
)

PROJECT_IMAGE_2 = (
    "https://upload.wikimedia.org/wikipedia/commons/7/74/"
    "Cruce_de_Calzada_de_Tlalpan_y_Circuito_Interior_R%C3%ADo_Churubusco_desde_un_dron_04.jpg"
)


# ================================================================
# 4. PREMIUM CSS
# ================================================================

CSS = """
<style>

:root {
    --bg0: #040712;
    --bg1: #07101e;
    --card: rgba(12, 22, 40, 0.88);
    --card2: rgba(16, 27, 49, 0.82);

    --text: #f8fafc;
    --muted: #8da0b8;

    --cyan: #22d3ee;
    --blue: #3b82f6;
    --purple: #a855f7;
    --green: #22c55e;
    --amber: #f59e0b;

    --border: rgba(148,163,184,.13);
}

/* --------------------------------------------------------------
   GLOBAL
-------------------------------------------------------------- */

html, body, [class*="css"] {
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(37,99,235,.14),
            transparent 30%
        ),
        radial-gradient(
            circle at 92% 0%,
            rgba(168,85,247,.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(34,211,238,.05),
            transparent 35%
        ),
        #040712;

    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

/* --------------------------------------------------------------
   SIDEBAR
-------------------------------------------------------------- */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #06101d,
            #040814
        );

    border-right:
        1px solid rgba(148,163,184,.10);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #d8e2ef;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(148,163,184,.12);
}

/* --------------------------------------------------------------
   HERO
-------------------------------------------------------------- */

.uav-hero {
    position: relative;
    overflow: hidden;

    padding: 42px 44px;

    margin-bottom: 30px;

    border-radius: 28px;

    border:
        1px solid rgba(34,211,238,.18);

    background:
        linear-gradient(
            120deg,
            rgba(10,23,45,.98),
            rgba(8,13,29,.93)
        );

    box-shadow:
        0 30px 80px rgba(0,0,0,.30);
}

.uav-hero:before {
    content: "";
    position: absolute;

    width: 420px;
    height: 420px;

    right: -120px;
    top: -260px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(34,211,238,.22),
            transparent 68%
        );
}

.uav-hero:after {
    content: "";

    position: absolute;

    width: 340px;
    height: 340px;

    right: 230px;
    bottom: -270px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(168,85,247,.19),
            transparent 68%
        );
}

.hero-eyebrow {
    position: relative;
    z-index: 2;

    color: #67e8f9;

    font-size: 11px;
    font-weight: 800;

    letter-spacing: .16em;
    text-transform: uppercase;
}

.hero-title {
    position: relative;
    z-index: 2;

    margin-top: 15px;

    color: white;

    font-size: clamp(40px,5vw,72px);
    line-height: .98;

    font-weight: 900;
    letter-spacing: -.055em;
}

.gradient-text {
    background:
        linear-gradient(
            90deg,
            #22d3ee,
            #60a5fa,
            #c084fc
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-copy {
    position: relative;
    z-index: 2;

    max-width: 900px;

    margin-top: 18px;

    color: #aab8ca;

    font-size: 16px;
    line-height: 1.75;
}

.hero-tags {
    position: relative;
    z-index: 2;

    display: flex;
    flex-wrap: wrap;

    gap: 9px;

    margin-top: 22px;
}

.hero-chip {
    padding: 7px 12px;

    border-radius: 999px;

    border:
        1px solid rgba(148,163,184,.15);

    background:
        rgba(15,23,42,.70);

    color: #dbeafe;

    font-size: 11px;
    font-weight: 650;
}

/* --------------------------------------------------------------
   NATIVE STREAMLIT METRIC CARD
-------------------------------------------------------------- */

[data-testid="stMetric"] {
    position: relative;

    min-height: 120px;

    padding: 20px 21px;

    border-radius: 19px;

    border:
        1px solid rgba(148,163,184,.13);

    background:
        linear-gradient(
            145deg,
            rgba(15,26,47,.96),
            rgba(8,14,28,.94)
        );

    box-shadow:
        0 12px 35px rgba(0,0,0,.17);

    overflow: hidden;
}

[data-testid="stMetric"]:after {
    content: "";

    position: absolute;

    left: 18px;
    right: 18px;
    bottom: 0;

    height: 3px;

    border-radius: 99px;

    background:
        linear-gradient(
            90deg,
            #22d3ee,
            #3b82f6,
            #a855f7
        );
}

[data-testid="stMetricLabel"] {
    color: #8292a8 !important;

    font-size: 11px !important;

    text-transform: uppercase;

    letter-spacing: .07em;
}

[data-testid="stMetricValue"] {
    color: white !important;

    font-size: 28px !important;
    font-weight: 850 !important;
}

[data-testid="stMetricDelta"] {
    color: #67e8f9 !important;
}

/* --------------------------------------------------------------
   HEADINGS
-------------------------------------------------------------- */

h1, h2, h3 {
    color: #f8fafc !important;
}

h2 {
    letter-spacing: -.025em;
}

.section-kicker {
    color: #67e8f9;

    font-size: 10px;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: .15em;

    margin-bottom: 4px;
}

/* --------------------------------------------------------------
   UPLOADER
-------------------------------------------------------------- */

[data-testid="stFileUploader"] section {
    min-height: 150px;

    border:
        1px dashed rgba(34,211,238,.38);

    border-radius: 20px;

    background:
        rgba(9,18,34,.75);

    transition: .20s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color:
        rgba(34,211,238,.75);

    background:
        rgba(10,23,43,.85);
}

/* --------------------------------------------------------------
   BUTTONS
-------------------------------------------------------------- */

.stButton > button {
    width: 100%;

    min-height: 54px;

    border: 0 !important;
    border-radius: 15px !important;

    color: white !important;

    font-size: 14px !important;
    font-weight: 800 !important;

    background:
        linear-gradient(
            90deg,
            #0891b2 0%,
            #2563eb 46%,
            #7c3aed 100%
        ) !important;

    box-shadow:
        0 12px 33px rgba(37,99,235,.24);

    transition: all .18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 17px 40px rgba(37,99,235,.34);
}

.stDownloadButton > button {
    width: 100%;

    min-height: 48px;

    border-radius: 14px !important;

    border:
        1px solid rgba(34,211,238,.22) !important;

    background:
        rgba(15,23,42,.72) !important;

    color: #dbeafe !important;

    font-weight: 700 !important;
}

/* --------------------------------------------------------------
   TABS
-------------------------------------------------------------- */

button[data-baseweb="tab"] {
    font-weight: 750 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #67e8f9 !important;
}

/* --------------------------------------------------------------
   DATAFRAME
-------------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border:
        1px solid rgba(148,163,184,.12);

    border-radius: 17px;

    overflow: hidden;
}

/* --------------------------------------------------------------
   IMAGES
-------------------------------------------------------------- */

[data-testid="stImage"] img {
    border-radius: 18px;

    border:
        1px solid rgba(148,163,184,.12);

    box-shadow:
        0 13px 38px rgba(0,0,0,.18);
}

/* --------------------------------------------------------------
   ALERTS
-------------------------------------------------------------- */

[data-testid="stAlert"] {
    border-radius: 15px;
}

/* --------------------------------------------------------------
   FOOTER
-------------------------------------------------------------- */

.custom-footer {
    margin-top: 55px;

    padding-top: 24px;

    border-top:
        1px solid rgba(148,163,184,.10);

    text-align: center;

    color: #56667d;

    font-size: 11px;
    line-height: 1.8;
}

/* --------------------------------------------------------------
   REMOVE DEFAULT STREAMLIT FOOTER
-------------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ================================================================
# 5. PREMIUM HERO + PROJECT VISUALS
# ================================================================

hero_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, Arial, sans-serif; color: #f8fafc; }}
.wrap {{ width: 100%; }}
.hero {{ display: flex; min-height: 430px; overflow: hidden; border-radius: 28px; border: 1px solid rgba(34,211,238,.18); background: linear-gradient(120deg,#0a172d,#080d1d); box-shadow: 0 30px 80px rgba(0,0,0,.30); }}
.hero-content {{ width: 58%; padding: 52px 44px; display: flex; flex-direction: column; justify-content: center; }}
.eyebrow {{ color: #67e8f9; font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }}
.title {{ margin-top: 15px; color: #fff; font-size: clamp(40px,5vw,72px); line-height: .98; font-weight: 900; letter-spacing: -.055em; }}
.gradient {{ background: linear-gradient(90deg,#22d3ee,#60a5fa,#c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.copy {{ max-width: 760px; margin-top: 18px; color: #aab8ca; font-size: 15px; line-height: 1.75; }}
.tags {{ display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }}
.chip {{ padding: 7px 12px; border-radius: 999px; border: 1px solid rgba(148,163,184,.15); background: rgba(15,23,42,.70); color: #dbeafe; font-size: 11px; font-weight: 650; }}
.visual {{ width: 42%; position: relative; overflow: hidden; }}
.visual img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.visual:after {{ content: ''; position: absolute; inset: 0; background: linear-gradient(90deg,rgba(7,18,35,.72),rgba(7,18,35,.08)),linear-gradient(180deg,rgba(2,8,18,.02),rgba(2,8,18,.42)); }}
.label {{ position: absolute; left: 22px; bottom: 22px; z-index: 3; padding: 10px 15px; border-radius: 11px; border: 1px solid rgba(255,255,255,.13); background: rgba(5,12,24,.78); color: #fff; font-size: 11px; font-weight: 750; letter-spacing: .10em; text-transform: uppercase; }}
.project-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 20px; }}
.project {{ position: relative; height: 270px; overflow: hidden; border-radius: 20px; border: 1px solid rgba(255,255,255,.08); background: #101722; }}
.project img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.project:after {{ content: ''; position: absolute; inset: 0; background: linear-gradient(180deg,rgba(2,8,18,.02) 30%,rgba(2,8,18,.78) 68%,rgba(2,8,18,.96) 100%); }}
.overlay {{ position: absolute; z-index: 3; left: 18px; right: 18px; bottom: 16px; padding: 13px 15px; border-radius: 13px; background: rgba(3,10,22,.88); border: 1px solid rgba(255,255,255,.14); box-shadow: 0 10px 28px rgba(0,0,0,.28); backdrop-filter: blur(7px); }}
.ptitle {{ color: #ffffff; font-size: 14px; line-height: 1.35; font-weight: 850; letter-spacing: .055em; text-shadow: 0 1px 8px rgba(0,0,0,.55); }}
.pdesc {{ margin-top: 6px; color: #d9e5f3; font-size: 12px; line-height: 1.45; }}
.about {{ display: grid; grid-template-columns: 180px 1fr; gap: 25px; padding: 25px 28px; margin-top: 20px; border-radius: 18px; border: 1px solid rgba(255,255,255,.08); background: rgba(12,23,40,.72); }}
.about-label {{ color: #67e8f9; font-size: 11px; font-weight: 800; letter-spacing: .12em; }}
.about-text {{ color: #b9c7d8; font-size: 13px; line-height: 1.75; }}
@media(max-width:900px) {{ .hero {{ flex-direction: column-reverse; }} .hero-content,.visual {{ width:100%; }} .visual {{ min-height:280px; }} .project-grid {{ grid-template-columns:1fr; }} .about {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="hero-content">
      <div class="eyebrow">Deep Learning · Computer Vision · UAV Analytics</div>
      <div class="title">UAV Vision <span class="gradient">Intelligence</span></div>
      <div class="copy">Research-grade aerial object detection powered by a trained YOLO26-L model on the VisDrone2019 domain. Analyze dense UAV scenes, localize small targets, inspect class distributions, and export structured detection intelligence from one interface.</div>
      <div class="tags">
        <span class="chip">YOLO26-L</span><span class="chip">VisDrone2019</span><span class="chip">10 Object Classes</span><span class="chip">Small-Object Detection</span><span class="chip">Academic Research Deployment</span>
      </div>
    </div>
    <div class="visual">
      <img src="{HERO_IMAGE}" alt="UAV aerial intelligence">
      <div class="label">UAV Aerial Intelligence</div>
    </div>
  </div>
  <div class="project-grid">
    <div class="project"><img src="{PROJECT_IMAGE_1}" alt="Aerial urban scene"><div class="overlay"><div class="ptitle">AERIAL SCENE UNDERSTANDING</div><div class="pdesc">Dense urban environments captured from above</div></div></div>
    <div class="project"><img src="{PROJECT_IMAGE_2}" alt="UAV imagery"><div class="overlay"><div class="ptitle">UAV IMAGERY</div><div class="pdesc">Real-world aerial vision analysis</div></div></div>
  </div>
  <div class="about"><div class="about-label">ABOUT PROJECT</div><div class="about-text">This project is a research-grade YOLO26-L based UAV object detection system trained and evaluated on the VisDrone2019 dataset for small-object detection in aerial imagery. It detects and localizes 10 object classes: pedestrian, person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor, enabling fine-grained analysis of complex UAV scenes.</div></div>
</div>
</body>
</html>
"""

components.html(
    hero_html,
    height=760,
    scrolling=False,
)


# 6. VERIFY MODEL
# ================================================================

if not MODEL_PATH.exists():

    st.error(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

    st.stop()


# ================================================================
# 7. LOAD MODEL
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model():

    return YOLO(
        str(
            MODEL_PATH
        )
    )


# ================================================================
# 8. DEVICE
# ================================================================

if torch.cuda.is_available():

    DEVICE = 0

    DEVICE_NAME = (
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = "cpu"

    DEVICE_NAME = "CPU"


# ================================================================
# 9. DETECTION FUNCTION
# ================================================================

def run_detection(
    model,
    image,
    confidence,
    iou_threshold,
    image_size,
    max_detections,
):

    started = (
        time.perf_counter()
    )


    results = model.predict(
        source=np.array(image),
        imgsz=image_size,
        conf=confidence,
        iou=iou_threshold,
        max_det=max_detections,
        device=DEVICE,
        verbose=False,
    )


    wall_time_ms = (
        time.perf_counter()
        - started
    ) * 1000


    result = results[0]


    rows = []

    class_counter = Counter()


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        cls_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )


        scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        coords = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )


        for detection_id, (
            class_id,
            score,
            xyxy,
        ) in enumerate(
            zip(
                cls_ids,
                scores,
                coords
            ),
            start=1
        ):

            class_name = (
                result.names[
                    int(class_id)
                ]
            )


            class_counter[
                class_name
            ] += 1


            x1, y1, x2, y2 = (
                xyxy.tolist()
            )


            rows.append(
                {
                    "ID":
                        detection_id,

                    "Class":
                        class_name,

                    "Confidence":
                        float(score),

                    "X1":
                        round(x1, 1),

                    "Y1":
                        round(y1, 1),

                    "X2":
                        round(x2, 1),

                    "Y2":
                        round(y2, 1),
                }
            )


    detections = pd.DataFrame(
        rows
    )


    annotated_bgr = (
        result.plot()
    )


    annotated_rgb = (
        annotated_bgr[
            :,
            :,
            ::-1
        ]
    )


    annotated_image = (
        Image.fromarray(
            annotated_rgb
        )
    )


    native_speed = getattr(
        result,
        "speed",
        {}
    )


    inference_ms = float(
        native_speed.get(
            "inference",
            wall_time_ms
        )
    )


    if detections.empty:

        mean_conf = 0.0
        max_conf = 0.0
        min_conf = 0.0

    else:

        mean_conf = float(
            detections[
                "Confidence"
            ].mean()
        )

        max_conf = float(
            detections[
                "Confidence"
            ].max()
        )

        min_conf = float(
            detections[
                "Confidence"
            ].min()
        )


    return {
        "detections":
            detections,

        "counts":
            class_counter,

        "total":
            int(
                sum(
                    class_counter.values()
                )
            ),

        "annotated":
            annotated_image,

        "mean_conf":
            mean_conf,

        "max_conf":
            max_conf,

        "min_conf":
            min_conf,

        "inference_ms":
            inference_ms,

        "wall_time_ms":
            wall_time_ms,
    }


# ================================================================
# 10. IMAGE TO BYTES
# ================================================================

def image_to_png_bytes(
    image
):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ================================================================
# 11. SIDEBAR
# ================================================================

with st.sidebar:

    st.title(
        "UAV Vision"
    )

    st.caption(
        "RESEARCH CONTROL CENTER"
    )


    st.success(
        "● Model checkpoint online"
    )


    st.divider()


    st.caption(
        "DEPLOYMENT MODEL"
    )

    st.write(
        "**Baseline YOLO26-L**"
    )


    st.caption(
        "EXECUTION DEVICE"
    )

    st.write(
        f"**{DEVICE_NAME}**"
    )


    st.caption(
        "MODEL CHECKPOINT"
    )

    st.code(
        "models/best.pt",
        language=None
    )


    st.divider()


    st.subheader(
        "Detection Controls"
    )


    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help=(
            "Lower confidence finds more objects "
            "but can increase false positives."
        ),
    )


    iou_threshold = st.slider(
        "NMS IoU threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.70,
        step=0.05,
    )


    image_size = st.selectbox(
        "Inference resolution",
        options=[
            640,
            960
        ],
        index=0,
    )


    max_detections = st.select_slider(
        "Maximum detections",
        options=[
            100,
            300,
            500,
            1000
        ],
        value=1000,
    )


    if image_size == 960:

        st.warning(
            "960×960 can improve tiny-object sensitivity, "
            "but CPU inference will be significantly slower."
        )


    st.divider()


    st.caption(
        "VISDRONE2019 CLASSES"
    )


    st.write(
        "Pedestrian · Person · Bicycle · Car · Van · "
        "Truck · Tricycle · Awning-Tricycle · Bus · Motor"
    )


# ================================================================
# 12. INFERENCE WORKSPACE
# ================================================================

st.caption(
    "INFERENCE WORKSPACE"
)

st.header(
    "Analyze UAV Imagery"
)

st.caption(
    "Upload an aerial image and run the trained "
    "YOLO26-L detector."
)


uploaded_file = st.file_uploader(
    "Upload UAV image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
    label_visibility="collapsed",
)


# ================================================================
# 13. EMPTY STATE
# ================================================================

if uploaded_file is None:

    st.info(
        "🛰️ **Ready for aerial intelligence**\n\n"
        "Upload a UAV image above to begin object detection, "
        "class-level analysis, and scene intelligence."
    )


    st.markdown(
        '<div class="custom-footer">'
        '<strong>UAV Vision Intelligence</strong><br>'
        'Multi-Scale YOLO26 · VisDrone2019 · '
        'Deep Learning Research Deployment'
        '</div>',
        unsafe_allow_html=True
    )


    st.stop()


# ================================================================
# 14. LOAD IMAGE
# ================================================================

try:

    image = Image.open(
        BytesIO(
            uploaded_file.getvalue()
        )
    ).convert(
        "RGB"
    )


except Exception as error:

    st.error(
        f"Unable to open uploaded image: {error}"
    )

    st.stop()


# ================================================================
# 15. RESET RESULT ON NEW IMAGE
# ================================================================

file_key = (
    uploaded_file.name,
    uploaded_file.size
)


if (
    st.session_state.get(
        "active_file"
    )
    != file_key
):

    st.session_state[
        "active_file"
    ] = file_key

    st.session_state.pop(
        "prediction",
        None
    )


# ================================================================
# 16. IMAGE INFORMATION
# ================================================================

megapixels = (
    image.width
    * image.height
    / 1_000_000
)


image_info = st.columns(
    4
)


with image_info[0]:

    st.metric(
        "Image Width",
        f"{image.width}px"
    )


with image_info[1]:

    st.metric(
        "Image Height",
        f"{image.height}px"
    )


with image_info[2]:

    st.metric(
        "Megapixels",
        f"{megapixels:.2f} MP"
    )


with image_info[3]:

    st.metric(
        "Model Input",
        f"{image_size} × {image_size}"
    )


st.write("")


# ================================================================
# 17. RUN BUTTON
# ================================================================

run = st.button(
    "⚡ Run UAV Intelligence",
    use_container_width=True
)


if run:

    try:

        with st.spinner(
            "Running YOLO26-L inference..."
        ):

            model = (
                load_model()
            )


            prediction = (
                run_detection(
                    model=model,
                    image=image,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    max_detections=max_detections,
                )
            )


            st.session_state[
                "prediction"
            ] = prediction


    except Exception as error:

        st.error(
            "Model inference failed."
        )

        st.exception(
            error
        )

        st.stop()


# ================================================================
# 18. BEFORE DETECTION
# ================================================================

if (
    "prediction"
    not in st.session_state
):

    st.info(
        "Image uploaded successfully. "
        "Click **Run UAV Intelligence** to start detection."
    )


    st.image(
        image,
        caption="Uploaded UAV scene",
        use_container_width=True
    )


    st.stop()


# ================================================================
# 19. GET RESULTS
# ================================================================

prediction = (
    st.session_state[
        "prediction"
    ]
)


detections = (
    prediction[
        "detections"
    ]
)


counts = (
    prediction[
        "counts"
    ]
)


total_objects = (
    prediction[
        "total"
    ]
)


# ================================================================
# 20. RESULT SUMMARY
# ================================================================

st.write("")
st.write("")

st.caption(
    "SCENE INTELLIGENCE"
)

st.header(
    "Detection Result"
)


result_cols = st.columns(
    5
)


with result_cols[0]:

    st.metric(
        "Detected Objects",
        total_objects
    )


with result_cols[1]:

    st.metric(
        "Classes Present",
        len(
            counts
        )
    )


with result_cols[2]:

    st.metric(
        "Mean Confidence",
        f"{prediction['mean_conf'] * 100:.1f}%"
    )


with result_cols[3]:

    st.metric(
        "Peak Confidence",
        f"{prediction['max_conf'] * 100:.1f}%"
    )


with result_cols[4]:

    st.metric(
        "Inference",
        f"{prediction['inference_ms']:.1f} ms"
    )


st.write("")


# ================================================================
# 21. TABS
# ================================================================

tab_detection, tab_analytics, tab_records, tab_research = (
    st.tabs(
        [
            "🎯 Detection View",
            "📊 Scene Analytics",
            "📑 Detection Records",
            "🧠 Research Model Card",
        ]
    )
)


# ================================================================
# 22. DETECTION VIEW
# ================================================================

with tab_detection:

    st.write("")


    left, right = st.columns(
        2,
        gap="large"
    )


    with left:

        st.subheader(
            "Original UAV Scene"
        )


        st.image(
            image,
            use_container_width=True
        )


    with right:

        st.subheader(
            "YOLO26-L Detection"
        )


        st.image(
            prediction[
                "annotated"
            ],
            use_container_width=True
        )


    st.write("")


    download_cols = (
        st.columns(
            2
        )
    )


    with download_cols[0]:

        st.download_button(
            "⬇ Download Annotated Image",
            data=image_to_png_bytes(
                prediction[
                    "annotated"
                ]
            ),
            file_name=(
                Path(
                    uploaded_file.name
                ).stem
                + "_YOLO26_detection.png"
            ),
            mime="image/png",
            use_container_width=True,
        )


    with download_cols[1]:

        st.download_button(
            "⬇ Download Detection CSV",
            data=detections.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                Path(
                    uploaded_file.name
                ).stem
                + "_detections.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ================================================================
# 23. ANALYTICS
# ================================================================

with tab_analytics:

    st.write("")


    if total_objects == 0:

        st.warning(
            "No objects detected above the selected confidence threshold."
        )


    else:

        class_df = pd.DataFrame(
            sorted(
                counts.items(),
                key=lambda item:
                    item[1],
                reverse=True
            ),
            columns=[
                "Class",
                "Detected Objects"
            ]
        )


        analytics_left, analytics_right = (
            st.columns(
                [
                    0.9,
                    1.3
                ],
                gap="large"
            )
        )


        with analytics_left:

            st.subheader(
                "Class Distribution"
            )


            st.dataframe(
                class_df,
                use_container_width=True,
                hide_index=True,
            )


        with analytics_right:

            st.subheader(
                "Detection Frequency"
            )


            st.bar_chart(
                class_df.set_index(
                    "Class"
                ),
                use_container_width=True,
            )


        dominant_class = (
            class_df.iloc[0][
                "Class"
            ]
        )


        dominant_count = int(
            class_df.iloc[0][
                "Detected Objects"
            ]
        )


        dominance = (
            dominant_count
            / total_objects
            * 100
        )


        st.write("")


        insight_cols = (
            st.columns(
                3
            )
        )


        with insight_cols[0]:

            st.metric(
                "Dominant Class",
                dominant_class.title()
            )


        with insight_cols[1]:

            st.metric(
                "Dominant Count",
                dominant_count
            )


        with insight_cols[2]:

            st.metric(
                "Scene Dominance",
                f"{dominance:.1f}%"
            )


# ================================================================
# 24. DETECTION RECORDS
# ================================================================

with tab_records:

    st.write("")

    st.subheader(
        "Individual Detection Records"
    )


    if detections.empty:

        st.info(
            "No detection records available."
        )


    else:

        records_df = (
            detections.copy()
        )


        records_df[
            "Confidence (%)"
        ] = (

            records_df[
                "Confidence"
            ]
            * 100

        ).round(
            2
        )


        records_df = (
            records_df.drop(
                columns=[
                    "Confidence"
                ]
            )
        )


        st.dataframe(
            records_df,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "Bounding-box coordinates correspond to "
            "the original uploaded image."
        )


# ================================================================
# 25. RESEARCH MODEL CARD
# ================================================================

with tab_research:

    st.write("")


    left, right = (
        st.columns(
            2,
            gap="large"
        )
    )


    with left:

        st.subheader(
            "Deployment Architecture"
        )


        st.markdown(
            """
**Model:** Baseline YOLO26-L

**Task:** Multi-class UAV object detection

**Dataset:** VisDrone2019

**Standard input:** 640 × 640

**Object classes:** 10

**Research focus:** Small, tiny, and very-tiny object detection in dense aerial scenes.
            """
        )


    with right:

        st.subheader(
            "Deployment Scope"
        )


        st.markdown(
            """
The deployed checkpoint is configured for multi-class UAV object
detection on VisDrone2019 aerial imagery. The interface focuses on
practical inference, small-object localization, class-level analysis,
and structured detection export.
            """
        )


    st.divider()


    st.subheader(
        "Ablation Context"
    )


    ablation_df = pd.DataFrame(
        [
            {
                "Model":
                    "Baseline YOLO26-L",

                "P2":
                    "No",

                "BiFPN-Lite":
                    "No",

                "Research Role":
                    "Best overall deployment model",
            },

            {
                "Model":
                    "E1: YOLO26-L + P2",

                "P2":
                    "Yes",

                "BiFPN-Lite":
                    "No",

                "Research Role":
                    "Tiny-target specialization",
            },

            {
                "Model":
                    "E2: P2 + BiFPN-Lite",

                "P2":
                    "Yes",

                "BiFPN-Lite":
                    "Yes",

                "Research Role":
                    "Very-tiny sensitivity / highest compute",
            },
        ]
    )


    st.dataframe(
        ablation_df,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    st.subheader(
        "VisDrone2019 Object Classes"
    )


    class_cols = (
        st.columns(
            5
        )
    )


    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        with class_cols[
            class_id % 5
        ]:

            st.metric(
                f"Class {class_id}",
                class_name.title()
            )


# ================================================================
# 26. RESEARCH NOTE
# ================================================================

st.write("")
st.write("")


st.info(
    "ℹ️ **Research note:** The displayed object count represents "
    "model detections above the selected confidence threshold. "
    "It is not the same as ground-truth count or ground-truth object count."
)


# ================================================================
# 27. FOOTER
# ================================================================

st.markdown(
    '<div class="custom-footer">'
    '<strong style="color:#94a3b8;">UAV Vision Intelligence</strong>'
    '<br>'
    'Multi-Scale YOLO26 for UAV Small-Object Detection'
    '<br>'
    'VisDrone2019 · Deep Learning · Computer Vision · '
    'Academic Research Deployment'
    '</div>',
    unsafe_allow_html=True
)