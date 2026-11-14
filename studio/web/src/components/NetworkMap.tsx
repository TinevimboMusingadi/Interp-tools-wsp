// The network, drawn on a canvas: embedding -> layer columns (attention block over MLP block, residual
// stream between) -> unembedding. Lasting edit heat in orange, the current step's touches as glows.
import { useEffect, useRef, useState } from "react";
import { COLORS, NEURON_BINS, binOf, key, type Heat } from "../heat";
import type { Arch, Touch } from "../types";

type Props = {
  arch: Arch;
  heat: Heat;
  flash: Touch[];
  flashKey: string | number;
  caption: string;
  sub?: string;
};

type Hit = { x: number; y: number; w: number; h: number; label: string };

export function NetworkMap({ arch, heat, flash, flashKey, caption, sub }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const hits = useRef<Hit[]>([]);
  const start = useRef(performance.now());
  const [size, setSize] = useState({ w: 800, h: 420 });
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    start.current = performance.now();
  }, [flashKey]);

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const c = canvas.current;
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== size.w * dpr || c.height !== size.h * dpr) {
        c.width = size.w * dpr;
        c.height = size.h * dpr;
      }
      const g = c.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const age = (performance.now() - start.current) / 1000;
      hits.current = render(g, size.w, size.h, arch, heat, flash, age, caption, sub ?? "");
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size, arch, heat, flash, caption, sub]);

  const onMove = (e: React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const h = hits.current.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
    setTip(h ? { x, y, text: h.label } : null);
  };

  return (
    <div ref={wrap} className="netmap" onMouseMove={onMove} onMouseLeave={() => setTip(null)}>
      <canvas id="netmap-canvas" ref={canvas} style={{ width: size.w, height: size.h }} />
      {tip && (
        <div className="tip" style={{ left: Math.min(tip.x + 14, size.w - 240), top: tip.y + 14 }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}

function rr(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.roundRect(x, y, w, h, r);
}

function render(
  g: CanvasRenderingContext2D, W: number, H: number, arch: Arch, heat: Heat, flash: Touch[], age: number,
  caption: string, sub: string,
): Hit[] {
  const out: Hit[] = [];
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#fbfbfc";
  g.fillRect(0, 0, W, H);
  g.fillStyle = "#e3e6eb"; // faint dot grid
  for (let gx = 12; gx < W; gx += 22) for (let gy = 12; gy < H; gy += 22) g.fillRect(gx, gy, 1.2, 1.2);

  const n = Math.max(1, arch.n_layers);
  const pad = 16;
  const top = pad + 46;
  const bottom = H - pad - 26;
  const hh = bottom - top;
  const sideW = Math.max(64, Math.min(120, W * 0.1));
  const x0 = pad + sideW + 22;
  const x1 = W - pad - sideW - 22;
  const colW = (x1 - x0) / n;
  const bw = Math.max(6, colW * 0.68);
  const attnY = top;
  const attnH = hh * 0.36;
  const residY = top + hh * 0.445;
  const mlpY = top + hh * 0.53;
  const mlpH = bottom - mlpY;
  const colX = (L: number) => x0 + colW * L + (colW - bw) / 2;

  // caption
  g.fillStyle = "#0f172a";
  g.font = "600 15px 'Inter Variable', Inter, system-ui, sans-serif";
  g.fillText(caption, pad, pad + 16, W - 260);
  g.fillStyle = "#64748b";
  g.font = "12px 'Inter Variable', Inter, system-ui, sans-serif";
  g.fillText(sub, pad, pad + 34, W - 260);
  legend(g, W - pad, pad + 12);

  const flashFor = (part: string, layer: number | null) =>
    flash.filter((t) => t.part === part && (t.layer === layer || (t.layer === null && layer !== null && part !== "embed" && part !== "head")));
  const pulse = 0.55 + 0.45 * Math.sin(age * 5);
  const fresh = Math.max(0, 1 - age / 1.2);

  // residual stream
  g.strokeStyle = "#d5d9e0";
  g.lineWidth = 3;
  g.beginPath();
  g.moveTo(pad + sideW, residY);
  g.lineTo(W - pad - sideW, residY);
  g.stroke();

  // a forward pass: a bright band sweeps embed -> head
  const runs = flash.some((t) => t.mode === "run" || t.mode === "steer");
  if (runs) {
    const p = (age % 1.6) / 1.6;
    const sx = pad + (W - 2 * pad) * p;
    const grad = g.createLinearGradient(sx - 80, 0, sx + 20, 0);
    grad.addColorStop(0, "rgba(22,163,74,0)");
    grad.addColorStop(1, "rgba(22,163,74,0.13)");
    g.fillStyle = grad;
    g.fillRect(sx - 80, top - 6, 100, hh + 12);
  }

  // embedding + head columns
  const side = (x: number, part: "embed" | "head", label: string) => {
    const k = key(part, null);
    base(g, x, top, sideW, hh, heat.edit.get(k) ?? 0, heat.everRead.get(k) ?? 0, false);
    const f = flashFor(part, null);
    for (const t of f) glow(g, x, top, sideW, hh, COLORS[t.mode], pulse);
    g.fillStyle = "#64748b";
    g.font = "11px 'Inter Variable', Inter, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(label, x + sideW / 2, bottom + 17);
    g.textAlign = "left";
    // token chips: rows edited so far, plus rows touched now
    const chips = new Map<string, string>();
    heat.tokens.forEach((v) => chips.set(v.label, v.mode === "state" ? COLORS.state : COLORS.edit));
    for (const t of f) (t.labels ?? []).forEach((l) => chips.set(l, COLORS[t.mode]));
    let cy = top + 10;
    for (const [lab, col] of chips) {
      if (cy > bottom - 18) break;
      chip(g, x + 6, cy, sideW - 12, lab, col);
      cy += 20;
    }
    const n = heat.editN.get(k);
    out.push({ x, y: top, w: sideW, h: hh, label: `${label}${arch.tied_embeddings ? " (tied: one matrix)" : ""} · ${arch.vocab.toLocaleString()} tokens × ${arch.d_model}${n ? ` · ${n.toLocaleString()} weights edited` : ""}${chips.size ? ` · rows: ${[...chips.keys()].slice(0, 12).join(", ")}` : ""}` });
  };
  side(pad, "embed", "Embedding");
  side(W - pad - sideW, "head", arch.tied_embeddings ? "Unembed (tied)" : "Unembed");

  for (let L = 0; L < n; L++) {
    const x = colX(L);
    const full = arch.layer_types[L] === "full";
    // attention
    const ka = key("attn", L);
    base(g, x, attnY, bw, attnH, heat.edit.get(ka) ?? 0, heat.everRead.get(ka) ?? 0, full);
    if (colW > 14) {
      g.fillStyle = full ? "#334155" : "#a0abbb";
      g.font = `${full ? 600 : 400} ${Math.min(11, colW * 0.4)}px 'Inter Variable', Inter, system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText(full ? "F" : "L", x + bw / 2, attnY + 13);
      g.textAlign = "left";
    }
    // MLP
    const km = key("mlp", L);
    base(g, x, mlpY, bw, mlpH, heat.edit.get(km) ?? 0, heat.everRead.get(km) ?? 0, false);
    const bins = heat.neuronEdit.get(L);
    const binH = mlpH / NEURON_BINS;
    if (bins) {
      g.fillStyle = COLORS.edit;
      bins.forEach((v, b) => {
        g.globalAlpha = 0.5 + 0.5 * v;
        g.fillRect(x + 2, mlpY + b * binH, bw - 4, Math.max(1.5, binH - 0.5));
      });
      g.globalAlpha = 1;
    }
    // residual node
    const kr = key("resid", L);
    const rcount = heat.everRead.get(kr) ?? 0;
    g.fillStyle = rcount ? `rgba(2,132,199,${Math.min(0.9, 0.3 + rcount * 0.1)})` : "#cbd2dc";
    g.beginPath();
    g.arc(x + bw / 2, residY, Math.max(3, Math.min(6, colW * 0.18)), 0, Math.PI * 2);
    g.fill();

    // flashes
    for (const t of flashFor("attn", L)) glow(g, x, attnY, bw, attnH, COLORS[t.mode], t.layer === null ? 0.4 : pulse);
    for (const t of flashFor("mlp", L)) {
      glow(g, x, mlpY, bw, mlpH, COLORS[t.mode], t.layer === null ? 0.4 : pulse);
      if (t.index) {
        g.fillStyle = COLORS[t.mode];
        for (const i of t.index.slice(0, 400)) {
          const b = binOf(i, arch);
          g.globalAlpha = 0.6 + 0.4 * pulse;
          g.fillRect(x + 1, mlpY + b * binH, bw - 2, Math.max(2, binH - 0.5));
        }
        g.globalAlpha = 1;
      }
    }
    for (const t of flashFor("norm", L)) glow(g, x - 2, attnY + attnH + 2, bw + 4, mlpY - attnY - attnH - 4, COLORS[t.mode], pulse);
    for (const t of flashFor("resid", L)) {
      g.strokeStyle = COLORS[t.mode];
      g.lineWidth = 2.5;
      g.shadowColor = COLORS[t.mode];
      g.shadowBlur = 16 * pulse;
      g.beginPath();
      g.arc(x + bw / 2, residY, 8 + 6 * fresh, 0, Math.PI * 2);
      g.stroke();
      g.shadowBlur = 0;
    }
    for (const t of flash.filter((t) => t.mode === "steer" && t.layer === L)) {
      g.fillStyle = COLORS.steer;
      g.beginPath();
      const cx = x + bw / 2;
      g.moveTo(cx - 7, residY - 22);
      g.lineTo(cx + 7, residY - 22);
      g.lineTo(cx, residY - 11);
      g.fill();
      void t;
    }

    // labels + hit boxes
    if (colW > 16 || L % 2 === 0) {
      g.fillStyle = full ? "#334155" : "#a0abbb";
      g.font = "10px 'Inter Variable', Inter, system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText(String(L), x + bw / 2, bottom + 17);
      g.textAlign = "left";
    }
    const en = (k: string) => (heat.editN.get(k) ? ` · ${heat.editN.get(k)!.toLocaleString()} weights edited` : "");
    out.push({ x, y: attnY, w: bw, h: attnH, label: `Layer ${L} ${full ? "full attention" : "linear attention (Gated DeltaNet)"} · ${full ? `${arch.n_heads} heads` : `${arch.linear_heads} heads`}${en(ka)}${reads(heat, ka)}` });
    out.push({ x, y: mlpY, w: bw, h: mlpH, label: `Layer ${L} MLP · ${arch.d_mlp.toLocaleString()} neurons${en(km)}${bins ? ` · ${bins.size} neuron groups edited` : ""}${reads(heat, km)}` });
    out.push({ x, y: residY - 8, w: bw, h: 16, label: `Layer ${L} residual stream (${arch.d_model} dims)${reads(heat, kr)}` });
  }

  // whole-model flash (e.g. a restore of every weight)
  if (flash.some((t) => t.part === "model" && t.mode === "edit")) glow(g, pad - 4, top - 4, W - 2 * pad + 8, hh + 8, COLORS.edit, pulse);
  return out;
}

function reads(heat: Heat, k: string) {
  const r = heat.everRead.get(k);
  return r ? ` · inspected ${r}×` : "";
}

function base(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, edit: number, read: number, strong: boolean) {
  rr(g, x, y, w, h, 5);
  g.fillStyle = "#eef1f5";
  g.fill();
  if (read) {
    g.fillStyle = `rgba(2,132,199,${Math.min(0.2, 0.04 + read * 0.03)})`;
    g.fill();
  }
  if (edit) {
    g.fillStyle = `rgba(234,88,12,${0.14 + 0.46 * edit})`;
    g.fill();
  }
  g.strokeStyle = strong ? "#94a3b8" : "#d5d9e0";
  g.lineWidth = strong ? 1.5 : 1;
  g.stroke();
}

function glow(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string, a: number) {
  g.save();
  g.shadowColor = color;
  g.shadowBlur = 14 * a;
  g.strokeStyle = color;
  g.globalAlpha = 0.5 + 0.5 * a;
  g.lineWidth = 2.5;
  rr(g, x - 1.5, y - 1.5, w + 3, h + 3, 6);
  g.stroke();
  g.restore();
}

function chip(g: CanvasRenderingContext2D, x: number, y: number, w: number, label: string, color: string) {
  rr(g, x, y, w, 16, 8);
  g.fillStyle = color + "1f";
  g.fill();
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.stroke();
  g.fillStyle = "#0f172a";
  g.font = "500 10.5px \"JetBrains Mono Variable\", ui-monospace, Consolas, monospace";
  const text = JSON.stringify(label).slice(1, -1);
  g.fillText(text.length > 12 ? text.slice(0, 11) + "…" : text, x + 6, y + 11.5, w - 10);
}

function legend(g: CanvasRenderingContext2D, right: number, y: number) {
  const items: [string, string][] = [["read", COLORS.read], ["steer", COLORS.steer], ["edit", COLORS.edit], ["forward pass", COLORS.run]];
  g.font = "11px 'Inter Variable', Inter, system-ui, sans-serif";
  let x = right;
  for (const [label, col] of items.reverse()) {
    const w = g.measureText(label).width;
    x -= w;
    g.fillStyle = "#64748b";
    g.fillText(label, x, y + 4);
    x -= 14;
    g.fillStyle = col;
    g.beginPath();
    g.arc(x + 5, y, 4, 0, Math.PI * 2);
    g.fill();
    x -= 12;
  }
}
