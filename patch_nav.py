"""
patch_nav.py - Patches app.py to replace sidebar radio nav with st.tabs()
"""
import re, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

with open('app.py', encoding='utf-8') as f:
    src = f.read()

# ── 1. Replace entire sidebar block (from "# SIDEBAR" header to end of with block)
#    and replace with: models loaded + sidebar stats only + tabs creation

old_sidebar = r"""# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:"""

# Find where sidebar starts and the FIRST page section begins
sidebar_start = src.find('# SIDEBAR\n')
if sidebar_start == -1:
    print("ERROR: Could not find '# SIDEBAR' marker")
    sys.exit(1)

# Find the next page section marker after sidebar
page_home_marker = '# PAGE: HOME\n'
page_home_start = src.find(page_home_marker)
if page_home_start == -1:
    print("ERROR: Could not find '# PAGE: HOME' marker")
    sys.exit(1)

# Go back to find the comment block before PAGE: HOME
# Find the line of dashes before "# PAGE: HOME"
pre_home_dashes = src.rfind('\n# ─', 0, page_home_start)

# The sidebar block is from sidebar_start to pre_home_dashes
sidebar_block = src[sidebar_start:pre_home_dashes+1]
print(f"Found sidebar block: {len(sidebar_block)} chars")
print(f"  starts: {sidebar_block[:60]!r}")
print(f"  ends:   {sidebar_block[-60:]!r}")

new_sidebar = '''# SIDEBAR — model status & KGE metrics only (navigation via tabs)
# ─────────────────────────────────────────────────────────────────────────────
models = load_models()
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.2rem;'>
      <div style='font-size:2.5rem;'>🌊</div>
      <div style='font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:0.4rem;'>Flood Prediction AI</div>
      <div style='font-size:0.70rem; color:#475569; margin-top:0.2rem;'>Ganga-Brahmaputra Basin · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    if models is not None:
        n_stations = len(models[2])
        st.markdown(f"""
        <div class='success-box' style='text-align:center; margin-bottom:1rem;'>
          ✅ <b>Models Loaded</b><br>
          <span style='font-size:0.76rem; color:#86efac;'>LGB + XGB global<br>{n_stations} station specialists</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box' style='text-align:center; margin-bottom:1rem;'>
          ⚡ <b>Demo Mode</b><br>
          <span style='font-size:0.76rem;'>Add model files to enable live predictions</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.73rem; color:#475569;'>
      <div style='font-weight:600; color:#64748b; margin-bottom:0.4rem;
           text-transform:uppercase; letter-spacing:1px;'>KGE Results</div>
      <div style='margin-bottom:0.22rem;'>🎯 Val KGE <b style='color:#22c55e; font-size:0.88rem;'>0.999875</b></div>
      <div style='margin-bottom:0.22rem;'>📈 r &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#60a5fa;'>0.9999</b></div>
      <div style='margin-bottom:0.22rem;'>📊 alpha &nbsp; <b style='color:#60a5fa;'>0.9999</b></div>
      <div style='margin-bottom:0.22rem;'>⚖️ beta &nbsp;&nbsp; <b style='color:#60a5fa;'>1.0000</b></div>
      <div style='margin-bottom:0.22rem;'>🔢 NSE &nbsp;&nbsp;&nbsp; <b style='color:#60a5fa;'>0.999827</b></div>
      <hr style='border-color:#1e3a5f; margin:0.6rem 0;'>
      <div style='font-weight:600; color:#64748b; margin-bottom:0.4rem;
           text-transform:uppercase; letter-spacing:1px;'>Architecture</div>
      <div style='margin-bottom:0.18rem;'>🌲 LightGBM (global)</div>
      <div style='margin-bottom:0.18rem;'>🌳 XGBoost (global)</div>
      <div style='margin-bottom:0.18rem;'>🔬 10 station specialists</div>
      <div style='margin-bottom:0.18rem;'>📡 367 gauge stations</div>
      <div style='margin-bottom:0.18rem;'>🔧 94 engineered features</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS — always-visible navigation (works at every screen size)
# ─────────────────────────────────────────────────────────────────────────────
tab_home, tab_predict, tab_explorer, tab_how = st.tabs(
    ["🏠 Home", "🔮 Predict", "📊 Explorer", "🧠 How It Works"]
)

'''

src = src[:sidebar_start] + new_sidebar + src[pre_home_dashes+1:]
print(f"After sidebar replacement: {len(src)} chars")

# ── 2. Replace "# PAGE: HOME\n...if page ==" with "with tab_home:\n"
src = re.sub(
    r'# ─+\n# PAGE: HOME\n# ─+\nif page == "[^"]+":',
    'with tab_home:',
    src
)

# ── 3. Replace "# PAGE: PREDICT\n...elif page ==" with "with tab_predict:\n"
src = re.sub(
    r'# ─+\n# PAGE: PREDICT\n# ─+\nelif page == "[^"]+":',
    'with tab_predict:',
    src
)

# ── 4. Replace "# PAGE: EXPLORER\n...elif page ==" with "with tab_explorer:\n"
src = re.sub(
    r'# ─+\n# PAGE: EXPLORER\n# ─+\nelif page == "[^"]+":',
    'with tab_explorer:',
    src
)

# ── 5. Replace "# PAGE: HOW IT WORKS\n...elif page ==" with "with tab_how:\n"
src = re.sub(
    r'# ─+\n# PAGE: HOW IT WORKS\n# ─+\nelif page == "[^"]+":',
    'with tab_how:',
    src
)

# ── 6. Remove stale `models = load_models()` calls inside tab blocks
#    (we already loaded models once before the sidebar)
src = re.sub(
    r'\n    models = load_models\(\)\n',
    '\n',
    src
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("✅ patch_nav.py complete")

# Verify syntax
import ast
try:
    ast.parse(src)
    print("✅ Syntax OK")
except SyntaxError as e:
    print(f"❌ SyntaxError: {e}")
