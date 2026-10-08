"""Build the decoding-scheme page from schemes_results.json.

Form: the job is COMPARE MAGNITUDE (win rate) across a handful of named schemes,
faceted by the question each family answers, so horizontal bars per family on one
shared x-axis, with the +/-1 SE interval drawn as a thin line at the bar end. The
emphasis form: the best arm of each family carries the accent hue, the rest are
gray. Showdown validity is a second measure, so it stays in the row label and the
table rather than on a second axis. The reference line is a real corpus team.
"""
import json, os
from pathlib import Path

SRC = "/tmp/vgc-pilot/schemes_results.json"
OUT = Path("/tmp/vgc-pilot/gui/static/schemes.html")
REAL, REAL_MEDIAN = 0.470, 0.458

LABEL = {
    "deporder_g2":  ("dependency order", "species → ability → item → moves → nature; temp 1.0 (shipped)"),
    "confidence":   ("most-confident column first", "MaskGIT-style: unmask the column the model is surest of"),
    "random_order": ("random column order", ""),
    "entropy_hi":   ("highest-entropy column first", "unmask the least certain column first"),
    "parallel8":    ("8 parallel steps", "unmask 1/8 of the grid per step"),
    "remask8":      ("8 parallel steps + remasking", "LLaDA-style low-confidence remasking"),
    "lowtemp":      ("temp 0.7 · top-p 0.9", "dependency order, sharper token draws"),
    "fill_move":    ("regenerate one move", "a real team with one move slot masked"),
    "fill_slot":    ("regenerate one candidate", "a real team with one whole slot masked"),
    "fill_pin":     ("build around a pinned Garchomp", "everything else generated, temp 1.0"),
    "fill_pin_lowtemp": ("pinned Garchomp · temp 0.7 · top-p 0.9", ""),
    "unconstrained": ("no constraints", "reject-and-resample against Showdown only"),
    "unconstrained_lowtemp": ("no constraints · temp 0.5", ""),
    "clause_only":  ("team clauses only", "Species + Item clauses kept, per-species legality dropped"),
    "no_learnset":  ("no learnset check", "everything except move legality"),
}
FAMILIES = [
    ("Which column to unmask next", "same weights, temp 1.0, guidance 2, top win-rate bin",
     ["deporder_g2", "confidence", "random_order", "entropy_hi", "parallel8", "remask8"]),
    ("How sharply to draw each token", "dependency order; only the sampling temperature changes",
     ["deporder_g2", "lowtemp"]),
    ("Fill in the blank", "start from a real team or a pinned candidate instead of an empty grid",
     ["fill_move", "fill_slot", "fill_pin_lowtemp", "fill_pin"]),
    ("Without the legality mask", "20,000 samples each — every arm produced zero Showdown-valid teams",
     ["unconstrained", "unconstrained_lowtemp", "clause_only", "no_learnset"]),
]

def main():
    R = json.load(open(SRC))
    data = []
    for title, sub, arms in FAMILIES:
        rows = []
        for a in arms:
            r = R[a]; name, note = LABEL[a]
            rows.append(dict(key=a, name=name, note=note, wr=r["win_rate"], se=r["se"],
                             n=r["n"], attempts=r["attempts"], acceptance=r["acceptance"],
                             p90=r["p90"], mx=r["max"], ge=r["above_real_median"],
                             surprise=r["surprise_mean"]))
        scored = [x for x in rows if x["wr"] is not None]
        # "best" = every arm within one standard error of the family's top arm, so a
        # 0.002 gap (0.16 sigma) is shown as the tie it is, not as a winner
        best = []
        if scored:
            top = max(scored, key=lambda x: x["wr"])
            best = [x["key"] for x in scored if x["wr"] >= top["wr"] - top["se"]]
        data.append(dict(title=title, sub=sub, rows=rows, best=best))
    meta = dict(real=REAL, real_median=REAL_MEDIAN, battles=24, per=150)
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__META__", json.dumps(meta))
    OUT.write_text(html)
    print("wrote", OUT, len(html), "bytes")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decoding schemes — win rate against the top 50</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8985;--grid:#ebebe8;
 --series:#2a78d6;--series-hover:#3987e5;--quiet:#c9c8c3;--quiet-hover:#b3b2ac;--rule:#e2e1dd}
@media(prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
 --ink3:#8b8a83;--grid:#2a2a28;--series:#3987e5;--series-hover:#5598e7;--quiet:#4a4a47;--quiet-hover:#5c5c58;--rule:#2f2f2d}}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--page);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;
 line-height:1.5;padding:clamp(1.2rem,4vw,3rem);max-width:1100px;margin:0 auto}
h1{font-family:"Bricolage Grotesque",sans-serif;font-weight:800;font-size:clamp(1.3rem,3vw,1.9rem);letter-spacing:-.02em}
.lede{color:var(--ink2);max-width:70ch;margin:.3rem 0 1.6rem;font-size:.92rem}
.controls{display:flex;gap:.6rem;align-items:center;margin-bottom:1.4rem;font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--ink2)}
.controls button{background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:3px;padding:.4rem .7rem;font:inherit;cursor:pointer}
.controls button[aria-pressed=true]{border-color:var(--series);color:var(--series)}
.controls button:focus-visible{outline:2px solid var(--series);outline-offset:2px}
section{background:var(--surface);border:1px solid var(--rule);border-radius:4px;padding:1.1rem 1.2rem 1rem;margin-bottom:1.6rem}
h2{font-family:"Bricolage Grotesque",sans-serif;font-weight:600;font-size:1.05rem;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:.85rem;margin:.1rem 0 .9rem}
.row{display:grid;grid-template-columns:minmax(170px,250px) 1fr;gap:.9rem;align-items:center;padding:.3rem 0;border-top:1px solid var(--grid)}
.row:first-of-type{border-top:0}
@media(max-width:720px){.row{grid-template-columns:1fr;gap:.15rem}}
.lab{font-size:.82rem;line-height:1.3}
.lab b{display:block;font-weight:600}
.lab small{display:block;color:var(--ink3);font-family:"IBM Plex Mono",monospace;font-size:.66rem;margin-top:.1rem}
.axis-row{display:grid;grid-template-columns:minmax(170px,250px) 1fr;gap:.9rem;margin-top:.4rem}
@media(max-width:720px){.axis-row{grid-template-columns:1fr}}
svg{display:block;width:100%;overflow:visible}
.bar{cursor:default;fill:var(--quiet)}
.bar.best{fill:var(--series)}
.bar:hover,.bar:focus{fill:var(--quiet-hover);outline:none}
.bar.best:hover,.bar.best:focus{fill:var(--series-hover)}
.bar:focus-visible{stroke:var(--ink);stroke-width:1}
.se{stroke:var(--ink);stroke-width:1.5}
.grid line{stroke:var(--grid);stroke-width:1}
.axis text,.val{fill:var(--ink3);font-family:"IBM Plex Mono",monospace;font-size:10px}
.val.best{fill:var(--ink2)}
.ref line{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}
.ref text{fill:var(--ink2);font-family:"IBM Plex Mono",monospace;font-size:10px}
.zero{fill:var(--ink3);font-family:"IBM Plex Mono",monospace;font-size:10px}
.legend{display:flex;gap:1.2rem;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:var(--ink2);margin:0 0 1.2rem}
.legend span::before{content:"";display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:.4rem;vertical-align:middle;background:var(--quiet)}
.legend span.b::before{background:var(--series)}
.legend span.r::before{height:0;width:14px;border-top:1px dashed var(--ink2);border-radius:0}
.legend span.s::before{height:0;width:14px;border-top:1.5px solid var(--ink);border-radius:0}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-family:"IBM Plex Mono",monospace;font-size:.7rem;padding:.4rem .55rem;border-radius:3px;display:none;z-index:9;white-space:nowrap;line-height:1.35}
table{border-collapse:collapse;width:100%;font-family:"IBM Plex Mono",monospace;font-size:.7rem;margin-top:.6rem}
th,td{text-align:right;padding:.25rem .45rem;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:500}
.hidden{display:none}
.foot{color:var(--ink3);font-size:.78rem;margin-top:.5rem;max-width:74ch}
</style></head>
<body>
<h1>Decoding schemes — win rate against the top 50</h1>
<p class="lede">One fixed checkpoint (<code>wrdiffusion.pt</code>, top win-rate bin, guidance 2). Each scheme decoded
<b>150 Showdown-valid teams</b>; each team played <b>24 battles</b> against the top-50 meta with the behaviour-cloning
policy on both sides. Bars are the mean win rate over the 150 teams; the black tick is ±1 standard error clustered by
team. The dashed line is an untouched real corpus team.</p>
<div class="controls" role="group" aria-label="View">
  <span>view</span>
  <button id="bChart" aria-pressed="true">chart</button>
  <button id="bTable" aria-pressed="false">table</button>
</div>
<div class="legend"><span class="b">best of its family (ties within 1 SE)</span><span>other schemes</span><span class="s">±1 SE (clustered by team)</span><span class="r">real corpus team 0.470</span></div>
<div id="root"></div>
<div id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const META = __META__;
const H = 30, BAR = 18, PADL = 4, PADR = 64, XMAX = 0.55;
let W = 760;
const $ = s => document.querySelector(s);
const tip = $("#tip");
const fmt = v => v == null ? "—" : v.toFixed(3);
const pct = v => v == null ? "—" : Math.round(v * 100) + "%";
function x(v){ return PADL + (W - PADL - PADR) * Math.min(v, XMAX) / XMAX; }
function ticks(){ return [0, .1, .2, .3, .4]; }

function barSvg(r, best, showGrid){
  const isBest = best.includes(r.key);
  if (r.wr == null){
    return `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${r.name}: no valid team in ${r.attempts.toLocaleString()} samples">
      <g class="grid">${ticks().map(t=>`<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H}"/>`).join("")}</g>
      <text class="zero" x="${PADL+6}" y="${H/2+4}">0 valid teams in ${r.attempts.toLocaleString()} samples — nothing to battle</text>
      <g class="ref"><line x1="${x(META.real)}" x2="${x(META.real)}" y1="0" y2="${H}"/></g></svg>`;
  }
  const w = Math.max(x(r.wr) - PADL, 4), y0 = (H - BAR) / 2;
  const lo = x(Math.max(0, r.wr - r.se)), hi = x(r.wr + r.se);
  const vlab = isBest || r.key === "deporder_g2" ? `<text class="val ${isBest?'best':''}" x="${hi + 6}" y="${H/2+4}">${fmt(r.wr)} ± ${fmt(r.se)}</text>` : "";
  return `<svg viewBox="0 0 ${W} ${H}" height="${H}">
    <g class="grid">${ticks().map(t=>`<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H}"/>`).join("")}</g>
    <path class="bar ${isBest?'best':''}" tabindex="0" data-key="${r.key}"
      d="M${PADL} ${y0} H${PADL+w-4} a4 4 0 0 1 4 4 V${y0+BAR-4} a4 4 0 0 1 -4 4 H${PADL} Z"
      aria-label="${r.name}: win rate ${fmt(r.wr)} plus or minus ${fmt(r.se)}"></path>
    <line class="se" x1="${lo}" x2="${hi}" y1="${H/2}" y2="${H/2}"/>
    <line class="se" x1="${lo}" x2="${lo}" y1="${H/2-4}" y2="${H/2+4}"/>
    <line class="se" x1="${hi}" x2="${hi}" y1="${H/2-4}" y2="${H/2+4}"/>
    ${vlab}
    <g class="ref"><line x1="${x(META.real)}" x2="${x(META.real)}" y1="0" y2="${H}"/></g>
  </svg>`;
}
function axisSvg(){
  return `<svg viewBox="0 0 ${W} 18" height="18" class="axis">
    ${ticks().map(t=>`<text x="${x(t)}" y="12" text-anchor="middle">${t.toFixed(1)}</text>`).join("")}
    <g class="ref"><text x="${x(META.real)}" y="12" text-anchor="middle">real 0.470</text></g></svg>`;
}
function render(){
  W = Math.max(420, Math.min(760, (document.body.clientWidth - 64) - 250));
  const root = $("#root"); root.innerHTML = "";
  for (const fam of DATA){
    const sec = document.createElement("section");
    sec.innerHTML = `<h2>${fam.title}</h2><p class="sub">${fam.sub}</p>` +
      fam.rows.map(r => `<div class="row"><div class="lab"><b>${r.name}</b>${r.note?`<span style="color:var(--ink2);font-size:.76rem">${r.note}</span>`:""}
        <small>${r.wr==null ? `0 of ${r.attempts.toLocaleString()} samples valid` : `${pct(r.acceptance)} Showdown-valid · ${r.n} teams`}</small></div>
        <div class="chart">${barSvg(r, fam.best)}</div></div>`).join("") +
      `<div class="axis-row"><div></div><div>${axisSvg()}</div></div>`;
    root.appendChild(sec);
  }
  const tbl = document.createElement("section"); tbl.id = "tbl"; tbl.className = "hidden";
  tbl.innerHTML = `<h2>Table</h2><table><thead><tr><th>scheme</th><th>family</th><th>win rate</th><th>SE</th><th>p90</th><th>max</th><th>≥ real median</th><th>valid</th><th>teams</th><th>samples</th><th>PMI surprise</th></tr></thead><tbody>` +
    DATA.flatMap(f => f.rows.map(r => `<tr><td>${r.name}</td><td>${f.title}</td><td>${fmt(r.wr)}</td><td>${fmt(r.se)}</td><td>${fmt(r.p90)}</td><td>${fmt(r.mx)}</td><td>${pct(r.ge)}</td><td>${pct(r.acceptance)}</td><td>${r.n}</td><td>${r.attempts.toLocaleString()}</td><td>${r.surprise==null?"—":r.surprise.toFixed(2)}</td></tr>`)).join("") +
    `</tbody></table><p class="foot">PMI surprise of a real corpus team is 1.84; ≥ real median counts teams at or above 0.458, the median untouched real team.</p>`;
  root.appendChild(tbl);
  root.querySelectorAll(".bar").forEach(b => {
    const r = DATA.flatMap(f=>f.rows).find(r=>r.key===b.dataset.key);
    const show = e => { tip.style.display="block";
      tip.innerHTML = `<b>${r.name}</b><br>win rate ${fmt(r.wr)} ± ${fmt(r.se)}<br>p90 ${fmt(r.p90)} · max ${fmt(r.mx)}<br>${pct(r.ge)} ≥ real median · ${pct(r.acceptance)} valid`;
      move(e); };
    const move = e => { const px = e.clientX ?? 0, py = e.clientY ?? 0;
      tip.style.left = Math.min(px + 14, window.innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = (py + 14) + "px"; };
    b.addEventListener("mouseenter", show); b.addEventListener("mousemove", move);
    b.addEventListener("focus", e => { const bb = b.getBoundingClientRect(); show({clientX: bb.right, clientY: bb.top}); });
    b.addEventListener("mouseleave", () => tip.style.display="none"); b.addEventListener("blur", () => tip.style.display="none");
  });
}
function setView(chart){
  $("#bChart").setAttribute("aria-pressed", chart); $("#bTable").setAttribute("aria-pressed", !chart);
  document.querySelectorAll("section").forEach(s => s.classList.toggle("hidden", (s.id === "tbl") === chart));
}
$("#bChart").onclick = () => setView(true); $("#bTable").onclick = () => setView(false);
window.addEventListener("resize", () => { render(); setView($("#bChart").getAttribute("aria-pressed")==="true"); });
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
