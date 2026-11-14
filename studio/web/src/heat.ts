// What the network map shows at a given moment: lasting edit heat (everything edited so far) plus the
// current step's touches (flash). Rebuilt from step 0 on seek, which is cheap (hundreds of steps).
import type { Arch, Step, Touch } from "./types";

export const NEURON_BINS = 56;

export type Heat = {
  edit: Map<string, number>; // "mlp:12" -> 0..1 intensity of lasting edits
  editN: Map<string, number>; // "mlp:12" -> weights changed (when known)
  neuronEdit: Map<number, Map<number, number>>; // layer -> bin -> intensity
  tokens: Map<number, { label: string; mode: string }>; // embedding / head rows edited so far
  everRead: Map<string, number>; // how often each part was looked at (faint trail)
};

export const key = (part: string, layer: number | null) => `${part}:${layer ?? "*"}`;

export function emptyHeat(): Heat {
  return { edit: new Map(), editN: new Map(), neuronEdit: new Map(), tokens: new Map(), everRead: new Map() };
}

function intensity(n?: number): number {
  if (!n) return 0.65;
  return Math.min(1, 0.35 + Math.log10(n + 1) / 7);
}

export function binOf(index: number, arch: Arch): number {
  return Math.min(NEURON_BINS - 1, Math.floor((index / Math.max(1, arch.d_mlp)) * NEURON_BINS));
}

function applyTouch(h: Heat, t: Touch, arch: Arch) {
  const k = key(t.part, t.layer);
  if (t.mode === "read" || t.mode === "run" || t.mode === "steer") {
    if (t.part !== "model") h.everRead.set(k, (h.everRead.get(k) ?? 0) + 1);
    return;
  }
  // edit / state: lasting
  if (t.part === "model") return; // a whole-model restore: handled by the caller resetting heat
  h.edit.set(k, Math.max(h.edit.get(k) ?? 0, intensity(t.n)));
  if (t.n) h.editN.set(k, t.n);
  if (t.part === "mlp" && t.layer !== null && t.index) {
    const bins = h.neuronEdit.get(t.layer) ?? new Map<number, number>();
    for (const i of t.index) bins.set(binOf(i, arch), 1);
    h.neuronEdit.set(t.layer, bins);
  }
  if ((t.part === "embed" || t.part === "head") && t.index) {
    t.index.forEach((id, j) => h.tokens.set(id, { label: t.labels?.[j] ?? `#${id}`, mode: t.mode }));
  }
}

export function applyStep(h: Heat, s: Step, arch: Arch) {
  const ts = s.touches ?? [];
  // an edit_report ("state") is the full current edit: it replaces earlier lasting heat
  if (ts.some((t) => t.mode === "state") && !ts.some((t) => t.mode === "edit")) {
    const fresh = emptyHeat();
    fresh.everRead = h.everRead;
    // keep token chips only if this state still covers the embedding and names no rows of its own
    const emb = ts.find((t) => t.mode === "state" && (t.part === "embed" || t.part === "head"));
    if (emb && !emb.index) fresh.tokens = h.tokens;
    Object.assign(h, fresh);
  }
  if (ts.some((t) => t.part === "model" && t.mode === "edit")) {
    h.edit.clear();
    h.editN.clear();
    h.neuronEdit.clear();
    h.tokens.clear();
  }
  for (const t of ts) applyTouch(h, t, arch);
}

export function heatUpTo(steps: Step[], upto: number, arch: Arch): Heat {
  const h = emptyHeat();
  for (let i = 0; i <= upto && i < steps.length; i++) applyStep(h, steps[i], arch);
  return h;
}

export const COLORS = {
  read: "#0284c7",
  edit: "#ea580c",
  state: "#d97706",
  steer: "#7c3aed",
  run: "#16a34a",
  error: "#dc2626",
};
