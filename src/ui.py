import streamlit as st
# src/ui.py 里新增
import json
import numpy as np
import pandas as pd
import re

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

def _pick_col_like(df: pd.DataFrame, keys: list[str]) -> str | None:
    lowers = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in lowers:
            return lowers[k.lower()]
    # 含义相近的兜底
    for c in df.columns:
        lc = c.lower().replace("_", " ")
        if any(k in lc for k in keys):
            return c
    return None

def _to_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """优先用 RA/Dec(/distance) 转 3D；否则把样本均匀撒在壳层上（可复现）。"""
    ra_col = _pick_col_like(df, ["ra", "ra_deg"])
    dec_col = _pick_col_like(df, ["dec", "dec_deg", "decl", "declination"])
    dist_col = _pick_col_like(df, ["dist", "distance", "st_dist", "parallax"])

    n = len(df)
    rng = np.random.default_rng(42)  # 固定种子，保证每次位置一致
    if ra_col is not None and dec_col is not None:
        ra = pd.to_numeric(df[ra_col], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(df[dec_col], errors="coerce").to_numpy(dtype=float)
        # 角度→弧度
        ra_r = np.deg2rad(np.nan_to_num(ra, nan=rng.uniform(0, 360, n)))
        dec_r = np.deg2rad(np.nan_to_num(dec, nan=rng.uniform(-90, 90, n)))
        # 距离缩放：可视化半径 120 左右
        if dist_col is not None:
            d_raw = pd.to_numeric(df[dist_col], errors="coerce").to_numpy(dtype=float)
            d = np.nan_to_num(d_raw, nan=1.0)
            d = 40 + 80 * (d / (np.nanpercentile(d, 95) + 1e-9))  # 截尾到 95 分位
            d = np.clip(d, 30, 120)
        else:
            d = rng.uniform(60, 120, n)
        x = d * np.cos(dec_r) * np.cos(ra_r)
        y = d * np.sin(dec_r)
        z = d * np.cos(dec_r) * np.sin(ra_r)
    else:
        # 没有 RA/Dec，就随机均匀撒点（壳层）
        r = rng.uniform(60, 120, n)
        theta = rng.uniform(0, 2*np.pi, n)
        phi = np.arccos(2*rng.random(n) - 1)
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.cos(phi)
        z = r*np.sin(phi)*np.sin(theta)

    out = pd.DataFrame({"x": x, "y": y, "z": z})
    return out

def render_universe_html(df: pd.DataFrame,
                         color_by: str = "label",
                         max_points: int = 5000,
                         point_size: float = 3.0) -> str:
    df = df.copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).reset_index(drop=True)

    # 位置：优先 RA/Dec，否则撒点
    pos = _to_xyz(df)

    # 颜色：调亮蓝色（更容易与灰色区分）
    col = df[color_by] if color_by in df.columns else pd.Series([np.nan]*len(df))
    colors = []
    for v in col:
        if pd.isna(v): colors.append("#8A93A6")      # unknown / candidate（灰）
        elif int(v) == 1: colors.append("#66B2FF")   # confirmed（更亮的蓝）
        else: colors.append("#E45757")               # false positive（红）

    # 名称 + 坐标（RA/Dec）用于标签
    name_col = _pick_name_col(df)
    names = (df[name_col].astype(str).fillna("").tolist()
             if name_col else [f"Obj {i}" for i in range(len(df))])

    # RA/Dec 尽量给出（没有就 NaN；标签会自动只显示名字）
    ra_col = _pick_col_like(df, ["ra", "ra_deg", "ra_str"])
    dec_col = _pick_col_like(df, ["dec", "dec_deg", "decl", "declination", "dec_str"])
    if ra_col is not None:
        ra_deg = _series_to_deg(df[ra_col], "ra").round(6).tolist()
    else:
        ra_deg = [float("nan")] * len(df)
    if dec_col is not None:
        dec_deg = _series_to_deg(df[dec_col], "dec").round(6).tolist()
    else:
        dec_deg = [float("nan")] * len(df)

    data = {
        "positions": pos[["x","y","z"]].round(3).to_numpy().tolist(),
        "colors": colors,
        "pointSize": float(point_size),
        "names": names,
        "ra": ra_deg,
        "dec": dec_deg,
    }
    jsdata = json.dumps(data)

    # HTML + Three.js（带最近 5 颗星高亮 + 标签 + Reset view）
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{ margin:0; height:100%; background:#0B1020; overflow:hidden; }}
  #c {{ width:100%; height:100%; display:block; }}
  .legend {{
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(11,16,32,.6); border:1px solid #1D2A55; color:#E6EDF3;
    padding:8px 10px; border-radius:10px; font:13px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial;
  }}
  .legend .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .reset-btn {{
    position:absolute; left:12px; z-index:11;
    background:#182448; border:1px solid #2E56A6; color:#E6EDF3;
    padding:6px 10px; border-radius:8px; font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial;
    cursor:pointer; user-select:none;
    transition:all .15s ease;
  }}
  .reset-btn:hover {{ filter:brightness(1.15); }}
  .labels {{ position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; z-index:12; }}
  .label {{
    position:absolute; transform:translate(-50%,-120%);
    padding:4px 6px; border-radius:6px; white-space:nowrap;
    background:rgba(13,22,46,.85); border:1px solid #2E56A6; color:#E6EDF3;
    font:12px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial;
    box-shadow:0 2px 6px rgba(0,0,0,.35);
  }}
</style>
</head>
<body>
<div class="legend" id="legend">
  <div><span class="dot" style="background:#66B2FF"></span>Confirmed / With exoplanet</div>
  <div><span class="dot" style="background:#E45757"></span>No exoplanet / False positive</div>
  <div><span class="dot" style="background:#8A93A6"></span>Unknown / Candidate</div>
  <div><span class="dot" style="background:#ffbb55"></span>Solar System</div>
</div>
<button class="reset-btn" id="resetBtn" style="top:72px;">Reset view</button>
<canvas id="c"></canvas>
<div class="labels" id="labels"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
  const DATA = {jsdata};
  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias:true }});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  const scene = new THREE.Scene();

  // 相机：球坐标控制（左键旋转、滚轮缩放）
  const camera = new THREE.PerspectiveCamera(60, 2, 0.1, 5000);
  let radius = 160, minR = 30, maxR = 500;
  let theta  = Math.PI/6;    // 水平角
  let phi    = Math.PI/3;    // 仰角（0~PI）
  const DEFAULT = {{ r:160, t:Math.PI/6, p:Math.PI/3 }};
  function updateCam(){{
    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);
    camera.position.set(x, y, z);
    camera.lookAt(0,0,0);
  }}
  function resetView(){{
    radius = DEFAULT.r; theta = DEFAULT.t; phi = DEFAULT.p; updateCam();
  }}
  resetView();

  // 灯光 & 太阳
  scene.add(new THREE.AmbientLight(0x99aadd, 0.6));
  const sunLight = new THREE.PointLight(0xffcc88, 1.2, 0, 2); scene.add(sunLight);
  const sun = new THREE.Mesh(new THREE.SphereGeometry(3,24,24), new THREE.MeshBasicMaterial({{ color:0xffbb55 }})); scene.add(sun);

  // 参考环（ecliptic）
  const ring = new THREE.Mesh(
      new THREE.TorusGeometry(120, 0.08, 8, 220),
      new THREE.MeshBasicMaterial({{ color:0x1d2a55 }})
  );
  ring.rotation.x = Math.PI/2; scene.add(ring);

  // 生成一个圆形、带柔和边缘的纹理，用作点精灵
  function makeCircleTexture(){{
    const s = 64;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = s;
    const g = canvas.getContext('2d');
    const r = s / 2;

    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    grd.addColorStop(0.0, 'rgba(255,255,255,1)');
    grd.addColorStop(1.0, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.beginPath(); g.arc(r, r, r, 0, Math.PI * 2); g.fill();

    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    return tex;
  }}

  // 点云（位置/颜色同原来）
  const N = DATA.positions.length;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);

  // 可选：极小抖动，弱化“网格感”（按需打开）
  // const JITTER = 0.05;

  for (let i = 0; i < N; i++) {{
    const p = DATA.positions[i];
    pos[3*i]   = p[0]; // + (Math.random()-0.5) * JITTER;
    pos[3*i+1] = p[1]; // + (Math.random()-0.5) * JITTER;
    pos[3*i+2] = p[2]; // + (Math.random()-0.5) * JITTER;

    const c = new THREE.Color(DATA.colors[i] || "#888888");
    col[3*i]   = c.r;
    col[3*i+1] = c.g;
    col[3*i+2] = c.b;
  }}
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));

  const dotTex = makeCircleTexture();
  const mat = new THREE.PointsMaterial({{
    size: DATA.pointSize,
    map: dotTex,               // 用圆形纹理做点
    transparent: true,
    alphaTest: 0.35,           // 过滤掉边缘透明像素，避免锯齿
    depthWrite: false,         // 不写深度，叠加更柔和
    blending: THREE.AdditiveBlending, // 发光叠加效果（想要更朴素可去掉这行）
    vertexColors: true,
    sizeAttenuation: true
  }});

  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // —— 最近 5 颗星：实心高亮 + 标签 —— //
  const HILIGHT_COUNT = 5;
  const highlightGroup = new THREE.Group(); scene.add(highlightGroup);
  const labelsRoot = document.getElementById('labels');
  let labelElems = [];  // {{ i, el }}

  function rebuildHighlights(indexes){{
    highlightGroup.clear();     // 保留接口，将来想画外圈可用
  labelsRoot.innerHTML = "";
  labelElems = [];

  indexes.forEach(i => {{
    // 仅创建 DOM 标签
    const el = document.createElement('div');
    el.className = "label";
    const name = DATA.names?.[i] || ("Obj " + i);

    const hasRA  = Number.isFinite(DATA.ra?.[i]);
    const hasDec = Number.isFinite(DATA.dec?.[i]);
    el.textContent = (hasRA && hasDec)
      ? `${{name}} (RA ${{DATA.ra[i].toFixed(3)}}°, Dec ${{DATA.dec[i].toFixed(3)}}°)`
      : name;

    labelsRoot.appendChild(el);
    labelElems.push({{ i, el }});
  }});
  }}

  const tmp = new THREE.Vector3();
  function pickNearestK(cameraPos){{
    // 简单 O(N) 取最近 K 个
    const idx = new Array(N).fill(0).map((_,i)=>i);
    idx.sort((a,b) => {{
      const pa=DATA.positions[a], pb=DATA.positions[b];
      const da=(pa[0]-cameraPos.x)**2+(pa[1]-cameraPos.y)**2+(pa[2]-cameraPos.z)**2;
      const db=(pb[0]-cameraPos.x)**2+(pb[1]-cameraPos.y)**2+(pb[2]-cameraPos.z)**2;
      return da - db;
    }});
    return idx.slice(0, HILIGHT_COUNT);
  }}

  function layoutLabels(){{
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight || 560;
    labelElems.forEach(({{ i, el }}) => {{
      const p = DATA.positions[i];
      tmp.set(p[0], p[1], p[2]).project(camera);
      const x = (tmp.x * .5 + .5) * w;
      const y = (-tmp.y * .5 + .5) * h;
      el.style.transform = `translate(${{x}}px, ${{y}}px) translate(-50%, -120%)`;
    }});
  }}

  // 交互：左键旋转，滚轮缩放
  canvas.style.cursor = 'grab';
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.style.cursor='grabbing'; e.preventDefault(); }});
  window.addEventListener('mouseup',   () => {{ dragging = false; canvas.style.cursor='grab'; }});
  window.addEventListener('mousemove', e => {{
    if(!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    const ROT = 0.005;
    theta -= dx * ROT;
    phi   -= dy * ROT;
    phi = Math.min(Math.PI - 0.1, Math.max(0.1, phi));
    updateCam();
  }});
  canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const zoom = Math.exp(e.deltaY * 0.001);
    radius = Math.min(maxR, Math.max(minR, radius * zoom));
    updateCam();
  }}, {{ passive:false }});

  // Reset 按钮（放在 Legend 下方）
  const legend = document.getElementById('legend');
  const resetBtn = document.getElementById('resetBtn');
  function placeReset(){{
    const box = legend.getBoundingClientRect();
    resetBtn.style.top = (box.bottom - box.top + 20) + "px";
  }}
  placeReset();
  window.addEventListener('resize', placeReset);
  resetBtn.addEventListener('click', () => {{ resetView(); }})

  // 自适应
  function setSize(){{
    const w = canvas.clientWidth  || document.body.clientWidth  || window.innerWidth;
    const h = canvas.clientHeight || document.body.clientHeight || window.innerHeight || 560;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }}
  setSize();
  window.addEventListener('resize', setSize);

  // 动画循环：每 10 帧更新一次“最近 5 颗”
  let frame = 0, lastPicked = [];
  function tick(){{
    frame++;
    sun.rotation.y += 0.002;
    if (frame % 10 === 0) {{
      const picked = pickNearestK(camera.position);
      // 避免没必要的 DOM/mesh 重建
      if (picked.join(",") !== lastPicked.join(",")) {{
        lastPicked = picked;
        rebuildHighlights(picked);
      }}
    }}
    layoutLabels();
    renderer.setClearColor(0x0B1020, 1);
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }}
  tick();
</script>
</body>
</html>
"""

def _parse_sexagesimal(s: str, kind: str) -> float | float:
    """
    把 '07h29m25.1s' / '07:29:25.1' / '07 29 25.1' / '-12d34m56' / '-12:34:56' 等
    解析为十进制度。kind='ra' 或 'dec'。
    """
    if s is None:
        return np.nan
    t = str(s).strip().lower()
    if not t or t in {"nan", "none"}:
        return np.nan
    # 统一分隔符：h/d/°/'/" 统统替换为冒号
    t = (t.replace("−", "-")
           .replace("h", ":")
           .replace("d", ":")
           .replace("m", ":")
           .replace("s", "")
        )
    t = re.sub(r"[°′’″\"]", ":", t)
    t = re.sub(r"\s+", ":", t)
    # 处理正负号
    sign = 1
    if t.startswith(("+", "-")):
        if t[0] == "-":
            sign = -1
        t = t[1:]
    parts = [p for p in t.split(":") if p != ""]
    if not parts:
        return np.nan
    a = float(parts[0])
    b = float(parts[1]) if len(parts) > 1 else 0.0
    c = float(parts[2]) if len(parts) > 2 else 0.0
    if kind == "ra":
        # RA: hour -> degree
        return (abs(a) + b/60 + c/3600) * 15.0
    else:
        return sign * (abs(a) + b/60 + c/3600)

def _series_to_deg(series: pd.Series, kind: str) -> pd.Series:
    """
    series 可能是十进制度、小时(ra)或六十进制字符串，统一转成十进制度。
    kind='ra' 或 'dec'
    """
    s = series.copy()
    # 先尝试当数值
    num = pd.to_numeric(s, errors="coerce")
    out = num.astype(float)

    # 对无法直接转数值的条目，尝试六十进制解析
    mask = num.isna()
    if mask.any():
        out.loc[mask] = s[mask].apply(
            lambda x: _parse_sexagesimal(x, kind) if isinstance(x, str) else np.nan
        )

    # RA 特例：如果是纯数字且 <=24，很可能是小时（没有任何分隔/单位）
    if kind == "ra":
        m_hour_like = out.notna() & (out <= 24) & (~s.astype(str).str.contains(r"[h:°d]", case=False, regex=True))
        out.loc[m_hour_like] = out.loc[m_hour_like] * 15.0

    # 规范范围
    if kind == "ra":
        out = out % 360.0
    else:
        out = out.clip(-90, 90)
    return out

def _pick_name_col(df: pd.DataFrame) -> str | None:
    """尽量找一个“名字/编号”列用于标签显示。"""
    candidates = [
        # 常见 exoplanet / TESS / Kepler 命名
        "pl_name", "pl_hostname",
        "koi_name", "kepoi_name", "koi", "kepid",
        "tic", "tic_id", "toi", "toi_id",
        "object", "obj_id", "id", "name", "designation", "hostname",
    ]
    lowers = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lowers:
            return lowers[k]
    # 兜底：挑一个最像“名字”的列
    for c in df.columns:
        if any(s in c.lower() for s in ["name", "id", "host", "obj", "tic", "kep"]):
            return c
    return None
