"""Build a self-contained HTML viewer for paired PatentData model runs."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import cairosvg
from PIL import Image

from tools.make_patent_comparison_sheet import _load, _load_graph, _load_raster


TARGET_WIDTH = 1400


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.joinpath("results.db").exists():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return name, path


def _save_width(image: Image.Image, path: Path, width: int = TARGET_WIDTH) -> None:
    scale = width / image.width
    resized = image.resize(
        (width, max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    resized.save(path, optimize=True)


def _render_svg(path: Path) -> Image.Image:
    png = cairosvg.svg2png(
        url=str(path), output_width=TARGET_WIDTH, background_color="white",
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def _metric_text(row: dict[str, Any]) -> str:
    edges = row.get("s2_n_edges")
    median = row.get("s2_median_edge_len")
    micro = row.get("s2_micro_edge_ratio")
    short = row.get("s2_short_edge_ratio")
    values = [f"edges {edges}" if edges is not None else "edges -"]
    values.append(f"median {median:.1f}" if median is not None else "median -")
    values.append(f"micro {micro:.3f}" if micro is not None else "micro -")
    values.append(f"short {short:.3f}" if short is not None else "short -")
    return " | ".join(values)


def build(runs: list[tuple[str, Path]], output: Path) -> None:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    data = {name: _load(path) for name, path in runs}
    run_names = [name for name, _path in runs]
    run_dirs = dict(runs)
    keys = sorted(set.intersection(*(set(data[name]) for name in run_names)))
    figures: list[dict[str, Any]] = []

    for index, (patent_id, sketch_id) in enumerate(keys):
        base_row = data[run_names[0]][(patent_id, sketch_id)]
        prefix = f"{index:03d}_{patent_id}_{sketch_id}"
        input_asset = assets / f"{prefix}_input.png"
        cleaned_asset = assets / f"{prefix}_cleaned.png"
        _save_width(_load_raster(Path(str(base_row["input_path"]))), input_asset)
        source_cleaned = (
            run_dirs[run_names[0]] / patent_id / "references"
            / f"{sketch_id}_norefs.png"
        )
        cleaned_value = None
        if source_cleaned.exists():
            _save_width(_load_raster(source_cleaned), cleaned_asset)
            cleaned_value = f"assets/{cleaned_asset.name}"

        figure: dict[str, Any] = {
            "patent": patent_id,
            "sketch": sketch_id,
            "input": f"assets/{input_asset.name}",
            "cleaned": cleaned_value,
            "runs": {},
        }
        for run_name in run_names:
            row = data[run_name][(patent_id, sketch_id)]
            run_dir = run_dirs[run_name]
            svg = run_dir / patent_id / "vectors" / f"{sketch_id}.svg"
            graph = run_dir / patent_id / "graphs" / f"{sketch_id}_graph.json"
            final_value = None
            graph_value = None
            if svg.exists():
                final_asset = assets / f"{prefix}_{run_name}_final.png"
                _render_svg(svg).save(final_asset, optimize=True)
                final_value = f"assets/{final_asset.name}"
            if graph.exists():
                graph_asset = assets / f"{prefix}_{run_name}_graph.png"
                _save_width(_load_graph(graph), graph_asset)
                graph_value = f"assets/{graph_asset.name}"
            figure["runs"][run_name] = {
                "status": row.get("status"),
                "metrics": _metric_text(row),
                "final": final_value,
                "graph": graph_value,
            }
        figures.append(figure)
        if (index + 1) % 20 == 0:
            print(f"rendered {index + 1}/{len(keys)}", flush=True)

    labels = {name: name for name in run_names}
    html = HTML.replace("__FIGURES__", json.dumps(figures))
    html = html.replace("__RUN_NAMES__", json.dumps(run_names))
    html = html.replace("__RUN_LABELS__", json.dumps(labels))
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(html)
    print(f"viewer: {output / 'index.html'}")
    print(f"figures: {len(figures)}; assets: {len(list(assets.iterdir()))}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PatentData paired model audit</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:Arial,sans-serif; background:#161616; color:#f2f2f2; letter-spacing:0; }
  header { position:sticky; top:0; z-index:5; min-height:50px; padding:8px 12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; background:#202020; border-bottom:1px solid #555; }
  button, input, select { height:32px; border:1px solid #666; border-radius:4px; background:#303030; color:#fff; padding:0 9px; font-size:13px; }
  button { cursor:pointer; }
  button:hover { background:#454545; }
  input { width:64px; }
  #identity { flex:1; min-width:220px; font-size:14px; font-weight:600; text-align:right; }
  main { padding:10px; }
  #grid { display:grid; grid-template-columns:repeat(var(--columns),minmax(260px,1fr)); gap:8px; }
  .panel { min-width:0; border:1px solid #555; border-radius:6px; overflow:hidden; background:#242424; }
  .panel-header { min-height:54px; padding:6px 8px; display:flex; align-items:flex-start; justify-content:space-between; gap:6px; border-bottom:1px solid #555; }
  .label { min-width:0; font-size:12px; font-weight:600; }
  .metrics { margin-top:5px; color:#b8b8b8; font-size:10px; white-space:normal; }
  .status-ok { color:#74d28c; }
  .status-gate, .status-error { color:#ff9a76; }
  .viewport { position:relative; height:calc(100vh - 145px); min-height:480px; overflow:hidden; background:#fff; cursor:grab; }
  .viewport.dragging { cursor:grabbing; }
  .content { position:absolute; left:0; top:0; transform-origin:0 0; }
  .content img { display:block; width:1400px; max-width:none; height:auto; }
  .empty { padding:24px; color:#a00000; font-size:13px; }
  #counter { min-width:90px; text-align:center; font-variant-numeric:tabular-nums; }
  @media (max-width:900px) {
    #grid { grid-template-columns:1fr; }
    .viewport { height:62vh; min-height:360px; }
    #identity { width:100%; text-align:left; }
  }
</style></head><body>
<header>
  <button id="prev" title="Previous drawing">Prev</button>
  <span id="counter"></span>
  <button id="next" title="Next drawing">Next</button>
  <input id="jump" type="number" min="1" title="Drawing number">
  <button id="go" title="Open drawing number">Go</button>
  <button id="reset" title="Reset synchronized zoom and pan">Reset</button>
  <span id="identity"></span>
</header>
<main><div id="grid"></div></main>
<script>
const FIGURES=__FIGURES__;
const RUN_NAMES=__RUN_NAMES__;
const RUN_LABELS=__RUN_LABELS__;
let index=0, scale=1, panX=0, panY=0, dragging=false;
let dragX=0, dragY=0, startX=0, startY=0;

function statusClass(status) {
  if (status === 'ok') return 'status-ok';
  if ((status || '').startsWith('quality_gate')) return 'status-gate';
  return 'status-error';
}
function viewport(src, id) {
  if (!src) return '<div class="viewport"><div class="empty">No output for this stage</div></div>';
  return `<div class="viewport" data-id="${id}"><div class="content"><img src="${src}"></div></div>`;
}
function applyTransform() {
  document.querySelectorAll('.content').forEach(el => {
    el.style.transform=`translate(${panX}px,${panY}px) scale(${scale})`;
  });
}
function resetTransform() { scale=1; panX=0; panY=0; applyTransform(); }
function wireViewports() {
  document.querySelectorAll('.viewport').forEach(vp => {
    vp.onwheel=e => {
      e.preventDefault();
      const rect=vp.getBoundingClientRect();
      const x=e.clientX-rect.left, y=e.clientY-rect.top;
      const factor=e.deltaY<0 ? 1.15 : 1/1.15;
      const next=Math.max(0.15,Math.min(25,scale*factor));
      panX=x-(x-panX)*(next/scale); panY=y-(y-panY)*(next/scale); scale=next;
      applyTransform();
    };
    vp.onmousedown=e => {
      dragging=true; dragX=e.clientX; dragY=e.clientY; startX=panX; startY=panY;
      document.querySelectorAll('.viewport').forEach(el=>el.classList.add('dragging'));
    };
  });
}
window.onmousemove=e => {
  if (!dragging) return;
  panX=startX+e.clientX-dragX; panY=startY+e.clientY-dragY; applyTransform();
};
window.onmouseup=() => {
  dragging=false;
  document.querySelectorAll('.viewport').forEach(el=>el.classList.remove('dragging'));
};

function render() {
  const figure=FIGURES[index];
  document.getElementById('counter').textContent=`${index+1} / ${FIGURES.length}`;
  document.getElementById('jump').value=index+1;
  document.getElementById('jump').max=FIGURES.length;
  document.getElementById('identity').textContent=`${figure.patent} / ${figure.sketch}`;
  const grid=document.getElementById('grid');
  grid.style.setProperty('--columns',RUN_NAMES.length+1);
  grid.innerHTML='';

  const input=document.createElement('section'); input.className='panel';
  const inputOptions=figure.cleaned ? '<option value="cleaned">Reference-free</option>' : '';
  input.innerHTML=`<div class="panel-header"><div class="label">Input</div><select><option value="input">Original</option>${inputOptions}</select></div>${viewport(figure.input,'input')}`;
  grid.appendChild(input);
  input.querySelector('select').onchange=e => {
    input.querySelector('.viewport').outerHTML=viewport(figure[e.target.value], 'input');
    wireViewports(); applyTransform();
  };

  RUN_NAMES.forEach(name => {
    const run=figure.runs[name];
    const panel=document.createElement('section'); panel.className='panel';
    let options='';
    if (run.final) options+='<option value="final">Final vector</option>';
    if (run.graph) options+='<option value="graph">Stage 2 graph</option>';
    const initial=run.final ? 'final' : 'graph';
    panel.innerHTML=`<div class="panel-header"><div class="label">${RUN_LABELS[name]} <span class="${statusClass(run.status)}">${run.status}</span><div class="metrics">${run.metrics}</div></div><select>${options}</select></div>${viewport(run[initial],name)}`;
    grid.appendChild(panel);
    const select=panel.querySelector('select');
    if (!run.final || !run.graph) select.disabled=true;
    select.onchange=e => {
      panel.querySelector('.viewport').outerHTML=viewport(run[e.target.value],name);
      wireViewports(); applyTransform();
    };
  });
  wireViewports(); applyTransform();
}
function move(delta) {
  index=Math.max(0,Math.min(FIGURES.length-1,index+delta)); resetTransform(); render();
}
document.getElementById('prev').onclick=()=>move(-1);
document.getElementById('next').onclick=()=>move(1);
document.getElementById('go').onclick=()=>{
  const value=Number(document.getElementById('jump').value)-1;
  if (Number.isFinite(value)) { index=Math.max(0,Math.min(FIGURES.length-1,value)); resetTransform(); render(); }
};
document.getElementById('reset').onclick=resetTransform;
window.onkeydown=e=>{
  if (e.key==='ArrowLeft') move(-1);
  if (e.key==='ArrowRight') move(1);
};
render();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")
    if len(dict(args.run)) != len(args.run):
        parser.error("run names must be unique")
    build(args.run, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
