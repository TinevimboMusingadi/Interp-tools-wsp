// Interactive figures for the p2 post: results by cell, tool use by cell, and the embedded Studio replay.
// Numbers come from experiments/p2_loop_fix/results/*.json and the run transcripts.
(function () {
  const BASE = { math: 0.41, lim: 0.28, gsm: 0.58, ife: 0.67 };
  const CELLS = [
    { id: "c0_b10_carbon_s0", c: "C0", agent: "Carbon", b: 10, math: 0.47, lim: 0.20, gsm: 0.58, ife: 0.67, w: 8155, fwd: 8, usd: 8.87,
      what: "Shrank the input embeddings of 8 “second-guessing” tokens (“Wait”, “Alternatively”, “-check”, “-read”) to 72–80% and nudged them toward the average token. The head is tied, so the same 8 output rows changed too." },
    { id: "c0_b10_argon_s0", c: "C0", agent: "Argon", b: 10, math: 0.45, lim: 0.34, gsm: 0.58, ife: 0.67, w: 15360, fwd: 6, usd: 8.41,
      what: "Edited two MLP neurons in layer 22 (573 and 870, which write “wait” and “alternative”) plus 13 columns of layer 23’s attention output." },
    { id: "c0_b25_carbon_s0", c: "C0", agent: "Carbon", b: 25, math: 0.41, lim: 0.28, gsm: 0.58, ife: 0.67, w: 0, fwd: 10, usd: 8.72,
      what: "Tried several edits but restored the original weights each time. It ran out of tokens with nothing kept." },
    { id: "c0_b25_argon_s0", c: "C0", agent: "Argon", b: 25, math: 0.42, lim: 0.40, gsm: 0.59, ife: 0.65, w: 6144, fwd: 9, usd: 8.60, sub: true,
      what: "The only agent that submitted. It found neurons in layer 22 that write “alternative”, “wait” and push away from “conclusion”, then silenced or flipped 6 of them. Its practice problems got fixed, but held-out MATH didn’t move, and more answers hit the length limit." },
    { id: "c1_b10_carbon_s0", c: "C1", agent: "Carbon", b: 10, math: null, lim: null, gsm: null, ife: null, w: 0, fwd: 0, usd: 0.21, bad: true,
      what: "Invalid. Gemini sent back an empty reply at turn 8 and the harness counted it as finished. This is a harness bug, so the cell gets re-run." },
    { id: "c1_b10_argon_s0", c: "C1", agent: "Argon", b: 10, math: 0.40, lim: 0.29, gsm: 0.60, ife: 0.63, w: 5118, fwd: 6, usd: 8.43,
      what: "The heaviest tool user: 39 neuron_logits calls and 5 writers_of calls. It edited 5 MLP neuron columns in layers 21–23 that fire on “\\n\\n###” section breaks." },
    { id: "c1_b25_carbon_s0", c: "C1", agent: "Carbon", b: 25, math: 0.41, lim: 0.61, gsm: 0.69, ife: 0.65, w: 82188435, fwd: 8, usd: 8.87,
      what: "Captured the residual stream, built a direction from the difference between two groups of prompts, and removed it from every attention and MLP output in layers 6–22. That changed 82M weights. GSM8K rose 11 points, but 61% of MATH answers now run to the length limit." },
    { id: "c1_b25_argon_s0", c: "C1", agent: "Argon", b: 25, math: 0.41, lim: 0.28, gsm: 0.58, ife: 0.67, w: 0, fwd: 8, usd: 8.53,
      what: "Inspected neurons with the tools but kept no edit." },
  ];
  const TOOLS = {
    c0_b10_carbon_s0: { python: 50, generate: 1 },
    c0_b10_argon_s0: { python: 63, generate: 6, edit_report: 2 },
    c0_b25_carbon_s0: { python: 48, generate: 1 },
    c0_b25_argon_s0: { python: 41, generate: 5, submit: 1 },
    c1_b10_carbon_s0: { python: 7 },
    c1_b10_argon_s0: { python: 55, generate: 5, neuron_logits: 39, writers_of: 5, capture: 1, direction_logits: 1 },
    c1_b25_carbon_s0: { python: 43, generate: 7, capture: 1 },
    c1_b25_argon_s0: { python: 42, generate: 7, neuron_logits: 6, writers_of: 1, capture: 1 },
  };
  const TOOL_COL = { python: "#94a3b8", generate: "#16a34a", edit_report: "#f59e0b", submit: "#10b981",
    neuron_logits: "#ec4899", writers_of: "#f97316", capture: "#0ea5e9", direction_logits: "#7c3aed" };
  const METRICS = [
    ["math", "MATH-500", "higher is better"],
    ["lim", "MATH answers hitting the length limit", "lower is better: the loop the agents were asked to fix"],
    ["gsm", "GSM8K", "higher is better"],
    ["ife", "IFEval subset", "side effects: higher is better"],
  ];
  const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const label = (c) => `<b>${c.c}</b> · ${c.agent} · ${c.b} passes`;

  function studio(id) {
    const box = document.getElementById("p2-studio");
    if (!box) return;
    const base = box.dataset.src;
    const run = "p2_loop_fix/" + id;
    box.querySelector("iframe").src = `${base}?embed=1&run=${encodeURIComponent(run)}`;
    box.querySelector("a.full").href = `${base}?run=${encodeURIComponent(run)}`;
    box.querySelector(".now").textContent = "Showing " + label(CELLS.find((c) => c.id === id)).replace(/<\/?b>/g, "");
  }

  function results(root) {
    let metric = "math";
    let sel = "c0_b10_carbon_s0";
    root.innerHTML = "";
    root.append(el("h4", null, "Every cell, one seed each"), el("p", "sub", "Click a row to see what that agent changed, then replay it below."));
    const seg = el("div", "seg");
    const note = el("p", "sub");
    const rows = el("div", "rows");
    const card = el("div", "card");
    root.append(seg, note, rows, el("div", "legend",
      '<span><i style="background:#64748b"></i>C0: raw weights</span><span><i style="background:#4f46e5"></i>C1: + interp tools</span><span><i style="border:2px dashed #0f172a;width:0;height:12px;border-radius:0;opacity:.55"></i>unedited model</span>'), card);
    METRICS.forEach(([k, name]) => {
      const b = el("button", null, name.replace(" answers hitting the length limit", " hit limit"));
      b.onclick = () => { metric = k; draw(); };
      b.dataset.k = k;
      seg.append(b);
    });
    function draw() {
      seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === metric));
      const m = METRICS.find((x) => x[0] === metric);
      note.textContent = `${m[1]} · ${m[2]} · 100 held-out problems per suite, so ±5 points is noise`;
      rows.innerHTML = "";
      const max = metric === "lim" ? 0.7 : 0.75;
      [10, 25].forEach((b) => {
        rows.append(el("div", "grp", `${b} forward passes`));
        CELLS.filter((c) => c.b === b).forEach((c) => {
          const r = el("div", "row" + (c.id === sel ? " sel" : ""));
          r.append(el("div", "lab", label(c)));
          const t = el("div", "track");
          const v = c[metric];
          const bar = el("div", "bar " + (c.bad ? "bad" : c.c.toLowerCase()));
          bar.style.width = "0%";
          t.append(bar, Object.assign(el("div", "base"), { style: `left:${(BASE[metric] / max) * 100}%` }));
          requestAnimationFrame(() => (bar.style.width = (c.bad ? 100 : ((v ?? 0) / max) * 100) + "%"));
          r.append(t, el("div", "val", c.bad ? "n/a" : pct(v)));
          r.onclick = () => { sel = c.id; draw(); };
          rows.append(r);
        });
      });
      const c = CELLS.find((x) => x.id === sel);
      card.innerHTML = `<h5>${label(c)}${c.sub ? " · submitted" : ""}</h5>
        <div class="stats"><span class="stat">MATH <b>${pct(c.math)}</b></span><span class="stat">hit limit <b>${pct(c.lim)}</b></span>
        <span class="stat">GSM8K <b>${pct(c.gsm)}</b></span><span class="stat">weights changed <b>${c.w.toLocaleString()}</b></span>
        <span class="stat">passes used <b>${c.fwd}/${c.b}</b></span><span class="stat">API cost <b>$${c.usd.toFixed(2)}</b></span></div>
        <div>${c.what}</div>`;
      const go = el("button", "go", "▶ Replay this run in the Studio");
      go.onclick = () => { studio(c.id); document.getElementById("p2-studio")?.scrollIntoView({ behavior: "smooth", block: "center" }); };
      card.append(go);
    }
    draw();
  }

  function tools(root) {
    root.innerHTML = "";
    root.append(el("h4", null, "What the agents spent their calls on"), el("p", "sub", "Tool calls per run. Grey is raw Python; colours are the interp tools (C1 only)."));
    const rows = el("div", "rows");
    root.append(rows);
    const max = Math.max(...Object.values(TOOLS).map((t) => Object.values(t).reduce((a, b) => a + b, 0)));
    CELLS.forEach((c) => {
      const t = TOOLS[c.id];
      const n = Object.values(t).reduce((a, b) => a + b, 0);
      const r = el("div", "row");
      r.style.cursor = "default";
      r.append(el("div", "lab", label(c)));
      const s = el("div", "stack");
      s.style.width = (n / max) * 100 + "%";
      Object.entries(t).forEach(([k, v]) => {
        const sp = el("span");
        sp.style.width = (v / n) * 100 + "%";
        sp.style.background = TOOL_COL[k] ?? "#cbd5e1";
        sp.title = `${k}: ${v}`;
        s.append(sp);
      });
      const wrap = el("div");
      wrap.append(s);
      r.append(wrap, el("div", "val", String(n)));
      rows.append(r);
    });
    root.append(el("div", "legend", Object.entries(TOOL_COL).map(([k, v]) => `<span><i style="background:${v}"></i>${k}</span>`).join("")));
  }

  function init() {
    const r = document.getElementById("p2-results");
    if (r) results(r);
    const t = document.getElementById("p2-tools");
    if (t) tools(t);
    studio("c0_b10_carbon_s0");
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
