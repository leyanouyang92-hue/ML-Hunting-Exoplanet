import streamlit as st

# ---------- UI: CSS & small components ----------
def inject_css():
    st.markdown("""
    <style>
      /* 顶栏透明，但保留，内容整体往下推一点 */
      header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
      .block-container { padding-top: 4.2rem !important; padding-bottom: .8rem; }

      /* Title */
      .big-title { font-size: 2.2rem; font-weight: 700; line-height: 1.25; letter-spacing: .3px; }
      .subtitle  { color: #9fb3c8; margin-top:.2rem; }

      /* Cards */
      .upload-card {
        background: linear-gradient(180deg, #101735 0%, #0B1020 100%);
        border: 1px solid #1D2A55; border-radius: 16px; padding: 14px; margin-bottom: 12px;
      }
      .kpi-card {
        background: #101735; border: 1px solid #1D2A55; border-radius: 16px; padding: 18px;
      }

      /* Slider */
      .stSlider > div > div > div[role='slider'] { background: #2E56A6; border: 2px solid #2E56A6; }
      .stSlider > div > div > div[data-baseweb="slider"] > div { background: #24407d; }

      /* Footer 隐藏 */
      footer {visibility: hidden;}

      /* 小屏适配：标题略小、顶距略小 */
      @media (max-width: 768px){
        .block-container { padding-top: 3.6rem !important; }
        .big-title { font-size: 1.8rem; }
      }
    </style>
    """, unsafe_allow_html=True)


def title_bar():
    st.markdown('<div class="big-title">🛰️ Exoplanet Mission Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Train → tune threshold → scout candidates</div>', unsafe_allow_html=True)

def upload_card(title, key, types):
    with st.expander(title, expanded=True):
        return st.file_uploader(" ", type=types, key=key, label_visibility="collapsed")

def kpi_grid(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    for (lbl, val), col in zip(metrics.items(), [c1,c2,c3,c4]):
        with col:
            st.markdown(f'<div class="kpi-card"><div style="color:#9fb3c8">{lbl}</div><div style="font-size:1.8rem;font-weight:700">{val}</div></div>', unsafe_allow_html=True)
