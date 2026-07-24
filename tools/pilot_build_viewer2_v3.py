"""Build viewer v2: adds a Stage-0 mask-overlay toggle on the input panel, a
Stage-2 graph-overlay toggle (kept/hachure-removed/closed-loop colored) on
each config panel, and synced zoom/pan across all 4 panels.
"""
import sqlite3, os, json, shutil
import cv2, numpy as np

RUNS = {
    "cnfusion": ("output/pilotv3_cnfusion", "CN-fusion (v3 filtered)"),
    "phaseA":   ("output/pilotv3_phaseA",   "Full-CNN phaseA + tiling"),
    "phaseB":   ("output/pilotv3_phaseB",   "Full-CNN phaseB + tiling"),
}
OUT = "output/pilotv3_viewer2"
ASSETS = f"{OUT}/assets"
os.makedirs(ASSETS, exist_ok=True)


def load_db(db_path):
    con = sqlite3.connect(db_path)
    cols = [c[1] for c in con.execute("PRAGMA table_info(results)").fetchall()]
    return {(r["patent_id"], r["sketch_id"]): r
            for r in (dict(zip(cols, row)) for row in con.execute("SELECT * FROM results"))}


runs_data = {key: load_db(f"{path}/results.db") for key, (path, _) in RUNS.items()}
all_keys = sorted(set().union(*[set(d.keys()) for d in runs_data.values()]))


def render_mask_overlay(tif_path, refjson_path, out_path, target_w):
    img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if os.path.exists(refjson_path):
        d = json.load(open(refjson_path))
        for lab in d.get("reference_labels", []):
            x, y, w, h = lab["bbox"]
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for ld in lab.get("leader_lines", []):
                p1 = tuple(map(int, ld["p1"])); p2 = tuple(map(int, ld["p2"]))
                cv2.line(rgb, p1, p2, (0, 140, 255), 2)
    H, W = rgb.shape[:2]
    sc = target_w / W
    cv2.imwrite(out_path, cv2.resize(rgb, (int(W * sc), int(H * sc))))
    return True


def render_graph_overlay(graph_path, out_path, target_w):
    if not os.path.exists(graph_path):
        return False
    g = json.load(open(graph_path))
    H, W = g["image_shape"]
    canvas = np.full((H, W, 3), 255, np.uint8)
    for e in g.get("edges", []):
        color = (200, 0, 0) if e.get("is_closed") else (0, 0, 0)  # closed=blue(BGR), open=black
        for x, y in e.get("pixels", []):
            if 0 <= int(y) < H and 0 <= int(x) < W:
                canvas[int(y), int(x)] = color
    for e in g.get("removed_hachures", []):
        for x, y in e.get("pixels", []):
            if 0 <= int(y) < H and 0 <= int(x) < W:
                canvas[int(y), int(x)] = (0, 140, 255)  # hachure-removed = orange
    sc = target_w / W
    cv2.imwrite(out_path, cv2.resize(canvas, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_NEAREST))
    return True


TARGET_W = 1400
figs = []
tif_cache = {}
for pat, sk in all_keys:
    entry = {"patent": pat, "sketch": sk, "configs": {}}
    tif_path = None
    ref_done = False
    for key, (root, label) in RUNS.items():
        r = runs_data[key].get((pat, sk))
        if r is None:
            entry["configs"][key] = {"status": "not_run"}
            continue
        tif_path = tif_path or r.get("input_path")
        status = r["status"]
        cfg_entry = {
            "status": status,
            "svg": None, "graph_png": None,
            "n_edges": r.get("s2_n_edges"), "n_prims": r.get("s3_n_primitives"),
            "keypoint_src": r.get("s2_keypoint_src"),
        }
        if status in ("ok",) or status.startswith("quality_gate"):
            svg_src = f"{root}/{pat}/vectors/{sk}.svg"
            if os.path.exists(svg_src):
                svg_name = f"{pat}_{sk}_{key}.svg"
                shutil.copy(svg_src, f"{ASSETS}/{svg_name}")
                cfg_entry["svg"] = f"assets/{svg_name}"
            graph_src = f"{root}/{pat}/graphs/{sk}_graph.json"
            graph_png = f"{pat}_{sk}_{key}_graph.png"
            if render_graph_overlay(graph_src, f"{ASSETS}/{graph_png}", TARGET_W):
                cfg_entry["graph_png"] = f"assets/{graph_png}"
        entry["configs"][key] = cfg_entry

        # Stage-0 outputs are identical across configs (same stage0 settings);
        # generate once per figure from whichever run has them.
        if not ref_done:
            refjson = f"{root}/{pat}/references/{sk}_references.json"
            norefs = f"{root}/{pat}/references/{sk}_norefs.png"
            if os.path.exists(refjson) and tif_path:
                mask_png = f"{pat}_{sk}_mask.png"
                if render_mask_overlay(tif_path, refjson, f"{ASSETS}/{mask_png}", TARGET_W):
                    entry["mask_png"] = f"assets/{mask_png}"
                    ref_done = True
            if os.path.exists(norefs):
                img = cv2.imread(norefs, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    H, W = img.shape
                    sc = TARGET_W / W
                    norefs_name = f"{pat}_{sk}_norefs.png"
                    cv2.imwrite(f"{ASSETS}/{norefs_name}",
                               cv2.resize(img, (int(W * sc), int(H * sc))))
                    entry["norefs_png"] = f"assets/{norefs_name}"

    if tif_path and os.path.exists(tif_path) and tif_path not in tif_cache:
        img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            H, W = img.shape
            sc = TARGET_W / W
            png_name = f"{pat}_{sk}_input.png"
            cv2.imwrite(f"{ASSETS}/{png_name}", cv2.resize(img, (int(W * sc), int(H * sc))))
            tif_cache[tif_path] = f"assets/{png_name}"
    entry["input_png"] = tif_cache.get(tif_path)
    entry["patent_dir"] = os.path.basename(os.path.dirname(tif_path)) if tif_path else ""
    figs.append(entry)
    if len(figs) % 20 == 0:
        print(f"  {len(figs)}/{len(all_keys)} figures processed", flush=True)

print(f"{len(figs)} figures indexed; assets: {len(os.listdir(ASSETS))} files")

CONFIG_LABELS = {k: v[1] for k, v in RUNS.items()}

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Stage 2 Pilot — 3-way Comparison v2</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background:#111; color:#eee; }
  header { padding: 8px 16px; display:flex; align-items:center; gap:12px; background:#1a1a1a; border-bottom:1px solid #333; position:sticky; top:0; z-index:10; flex-wrap:wrap;}
  header h1 { font-size:14px; margin:0; font-weight:600; flex:1; }
  .navbtn { background:#333; color:#eee; border:1px solid #555; border-radius:6px; padding:5px 12px; cursor:pointer; font-size:13px; }
  .navbtn:hover { background:#444; }
  .counter { font-variant-numeric: tabular-nums; min-width:100px; text-align:center; font-size:13px;}
  .jump { width:55px; background:#222; color:#eee; border:1px solid #555; border-radius:4px; padding:4px; }
  main { padding:10px 14px 30px; }
  .figtitle { font-size:13px; margin-bottom:8px; color:#aaa; }
  .figtitle b { color:#fff; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; }
  .panel { background:#1c1c1c; border:1px solid #333; border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }
  .panel .hdr { padding:5px 8px; font-size:11px; font-weight:600; background:#252525; border-bottom:1px solid #333; display:flex; justify-content:space-between; align-items:center; gap:6px;}
  .panel .hdr select { font-size:10px; background:#333; color:#eee; border:1px solid #555; border-radius:4px; }
  .viewport { flex:1; min-height:560px; background:#fff; overflow:hidden; position:relative; cursor:grab; }
  .viewport.dragging { cursor:grabbing; }
  .zoomlayer { position:absolute; top:0; left:0; transform-origin:0 0; }
  .zoomlayer img, .zoomlayer object { display:block; max-width:none; }
  .stub { color:#e88; font-size:12px; padding:20px; text-align:center; }
  .stats { font-size:10px; color:#888; padding:5px 8px; border-top:1px solid #333; }
  .legend { font-size:10px; color:#999; padding:4px 8px; }
  .legend span { margin-right:8px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:3px; vertical-align:middle;}
  footer { padding:8px 16px; font-size:11px; color:#777; }
  kbd { background:#333; border:1px solid #555; border-radius:3px; padding:1px 6px; font-size:11px; }
</style></head>
<body>
<header>
  <button class="navbtn" id="prev">&larr; Prev</button>
  <span class="counter" id="counter"></span>
  <button class="navbtn" id="next">Next &rarr;</button>
  <input class="jump" id="jump" type="number" min="1"> <button class="navbtn" id="go">Go</button>
  <button class="navbtn" id="resetzoom">Reset zoom</button>
  <span style="font-size:11px;color:#999;">scroll = zoom (synced) &middot; drag = pan (synced)</span>
  <h1 id="figlabel"></h1>
</header>
<main>
  <div class="figtitle" id="subtitle"></div>
  <div class="grid" id="grid"></div>
</main>
<footer>Keyboard: <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> to navigate figures. Input panel: toggle Original / Stage-0 mask+leaders / Cleaned(norefs). Config panels: toggle Final SVG / Stage-2 graph (black=kept open edge, blue=closed loop, orange=hachure-removed).</footer>
<script>
const CONFIG_ORDER = """ + json.dumps(list(RUNS.keys())) + """;
const CONFIG_LABELS = """ + json.dumps(CONFIG_LABELS) + """;
let FIGS = __FIGS_JSON__;
let i = 0;

// shared zoom/pan state, synced across all 4 viewports
let scale = 1, panX = 0, panY = 0;
let dragging = false, dragStartX = 0, dragStartY = 0, panStartX = 0, panStartY = 0;

function applyTransform() {
  document.querySelectorAll('.zoomlayer').forEach(el => {
    el.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  });
}
function resetZoom() { scale = 1; panX = 0; panY = 0; applyTransform(); }

function statusNote(s) {
  const map = {
    "quality_gate_stage1": "stopped: Stage-1 quality gate",
    "quality_gate_stage2": "stopped: Stage-2 quality gate",
    "quality_gate_stage3": "stopped: Stage-3 quality gate",
    "quality_gate_stage4": "stopped: Stage-4 quality gate",
    "stage0": "error in Stage 0", "stage1": "error in Stage 1",
    "stage2": "error in Stage 2", "stage3": "error in Stage 3",
    "stage4": "error in Stage 4", "not_run": "not processed",
  };
  return map[s] || s;
}

function viewportHTML(id, contentHTML) {
  return `<div class="viewport" data-vp="${id}"><div class="zoomlayer">${contentHTML}</div></div>`;
}

function render() {
  const f = FIGS[i];
  document.getElementById("counter").textContent = (i+1) + " / " + FIGS.length;
  document.getElementById("jump").value = i+1;
  document.getElementById("figlabel").textContent = f.patent_dir + " / " + f.sketch;
  document.getElementById("subtitle").innerHTML = "<b>" + f.patent_dir + "</b> — sketch " + f.sketch;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  // ---- Input panel ----
  const p0 = document.createElement("div"); p0.className = "panel";
  const inputModes = {orig: f.input_png, mask: f.mask_png, norefs: f.norefs_png};
  let sel0 = `<select data-role="input-mode">
      <option value="orig">Original</option>
      ${f.mask_png ? '<option value="mask">Stage-0 mask+leaders</option>' : ''}
      ${f.norefs_png ? '<option value="norefs">Cleaned (norefs)</option>' : ''}
    </select>`;
  p0.innerHTML = `<div class="hdr"><span>Input</span>${sel0}</div>
    <div id="vp-input"></div>
    <div class="stats">&nbsp;</div>`;
  grid.appendChild(p0);
  const setInputPanelBody = (mode) => {
    const src = inputModes[mode] || f.input_png;
    const holder = p0.querySelector('[data-vpholder="input"]') || document.createElement('div');
    const body = src ? viewportHTML('input', `<img src="${src}">`) : `<div class="stub">no input</div>`;
    p0.querySelector('.hdr').nextElementSibling.outerHTML = body;
  };
  setInputPanelBody("orig");
  p0.querySelector('select[data-role="input-mode"]').addEventListener('change', (e) => {
    setInputPanelBody(e.target.value); wireViewports(); applyTransform();
  });

  // ---- Config panels ----
  for (const key of CONFIG_ORDER) {
    const c = f.configs[key] || {status: "not_run"};
    const p = document.createElement("div"); p.className = "panel";
    const hasSvg = !!c.svg, hasGraph = !!c.graph_png;
    let sel = `<select data-role="cfg-mode">
        ${hasSvg ? '<option value="svg">Final SVG</option>' : ''}
        ${hasGraph ? '<option value="graph">Stage-2 graph</option>' : ''}
      </select>`;
    let stats = c.status === "ok"
      ? `<span>edges: ${c.n_edges ?? "-"}</span><span>prims: ${c.n_prims ?? "-"}</span>` +
        (c.keypoint_src ? `<span>kp: ${c.keypoint_src}</span>` : "")
      : `<span style="color:#e88">${statusNote(c.status)}</span>`;
    let body;
    if (hasSvg) body = viewportHTML(key, `<object type="image/svg+xml" data="${c.svg}"></object>`);
    else if (hasGraph) body = viewportHTML(key, `<img src="${c.graph_png}">`);
    else body = `<div class="stub">${statusNote(c.status)}</div>`;
    p.innerHTML = `<div class="hdr"><span>${CONFIG_LABELS[key]}</span>${(hasSvg&&hasGraph)?sel:''}</div>
      ${body}
      <div class="legend">${hasGraph ? '<span><i class="dot" style="background:#000"></i>open</span><span><i class="dot" style="background:#0000c8"></i>closed-loop</span><span><i class="dot" style="background:#ff8c00"></i>hachure-removed</span>' : ''}</div>
      <div class="stats">${stats}</div>`;
    grid.appendChild(p);
    if (hasSvg && hasGraph) {
      p.querySelector('select[data-role="cfg-mode"]').addEventListener('change', (e) => {
        const content = e.target.value === 'graph'
          ? `<img src="${c.graph_png}">`
          : `<object type="image/svg+xml" data="${c.svg}"></object>`;
        p.querySelector('.viewport').outerHTML = viewportHTML(key, content);
        wireViewports(); applyTransform();
      });
    }
  }
  wireViewports();
  applyTransform();
}

function wireViewports() {
  document.querySelectorAll('.viewport').forEach(vp => {
    vp.onwheel = (e) => {
      e.preventDefault();
      const rect = vp.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
      const newScale = Math.max(0.2, Math.min(20, scale * factor));
      // zoom toward cursor position (approx, shared across panels)
      panX = mx - (mx - panX) * (newScale / scale);
      panY = my - (my - panY) * (newScale / scale);
      scale = newScale;
      applyTransform();
    };
    vp.onmousedown = (e) => {
      dragging = true; vp.classList.add('dragging');
      dragStartX = e.clientX; dragStartY = e.clientY;
      panStartX = panX; panStartY = panY;
    };
  });
  window.onmousemove = (e) => {
    if (!dragging) return;
    panX = panStartX + (e.clientX - dragStartX);
    panY = panStartY + (e.clientY - dragStartY);
    applyTransform();
  };
  window.onmouseup = () => {
    dragging = false;
    document.querySelectorAll('.viewport').forEach(vp => vp.classList.remove('dragging'));
  };
}

function go(delta) { i = Math.max(0, Math.min(FIGS.length-1, i+delta)); resetZoom(); render(); }
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("go").onclick = () => { i = Math.max(0, Math.min(FIGS.length-1, (+document.getElementById("jump").value)-1)); resetZoom(); render(); };
document.getElementById("resetzoom").onclick = () => { resetZoom(); };
window.addEventListener("keydown", (e) => {
  if (["INPUT","SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowLeft") go(-1);
  if (e.key === "ArrowRight") go(1);
});

render();
</script>
</body></html>
"""
HTML = HTML.replace("__FIGS_JSON__", json.dumps(figs))
open(f"{OUT}/index.html", "w").write(HTML)
print(f"viewer -> {OUT}/index.html  ({os.path.getsize(f'{OUT}/index.html')/1e6:.2f} MB)")
