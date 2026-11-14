// Plain-language captions for steps and touches (what the map's headline says).
import type { Step, Touch } from "./types";

export const TOOL_BADGE: Record<string, [string, string]> = {
  python: ["Py", "#3b82f6"],
  generate: ["▶", "#22c55e"],
  edit_report: ["Δ", "#f59e0b"],
  submit: ["✓", "#10b981"],
  logit_lens: ["LL", "#06b6d4"],
  attention: ["At", "#8b5cf6"],
  neuron_logits: ["N", "#ec4899"],
  writers_of: ["W", "#f97316"],
  contrast_direction: ["Dir", "#14b8a6"],
  direction_logits: ["DL", "#0ea5e9"],
  steer_test: ["St", "#a855f7"],
  capture: ["Cap", "#64748b"],
  sae_features: ["SAE", "#eab308"],
  sae_feature_info: ["SAE", "#eab308"],
};

export function badge(tool?: string): [string, string] {
  return TOOL_BADGE[tool ?? ""] ?? [(tool ?? "?").slice(0, 3), "#475569"];
}

export function where(t: Touch): string {
  const L = t.layer === null ? "every layer" : `layer ${t.layer}`;
  const lab = t.labels?.length ? ` (${t.labels.slice(0, 6).map((l) => JSON.stringify(l)).join(", ")}${t.labels.length > 6 ? ", …" : ""})` : "";
  switch (t.part) {
    case "embed":
      return `embedding rows${lab || (t.index ? ` ${t.index.slice(0, 6).join(", ")}` : "")}`;
    case "head":
      return `unembedding${lab}`;
    case "mlp":
      return `${L} MLP${t.index ? ` neuron${t.index.length > 1 ? "s" : ""} ${t.index.slice(0, 5).join(", ")}${t.index.length > 5 ? "…" : ""}` : ""}`;
    case "attn":
      return `${L} attention`;
    case "resid":
      return `${L} residual stream`;
    case "norm":
      return t.layer === null ? "final norm" : `${L} norm`;
    default:
      return "the whole model";
  }
}

export function summarize(ts: Touch[], mode: Touch["mode"]): string {
  const sel = ts.filter((t) => t.mode === mode && t.part !== "model");
  if (!sel.length) return "";
  const parts = sel.slice(0, 3).map(where);
  return parts.join("; ") + (sel.length > 3 ? ` and ${sel.length - 3} more` : "");
}

export function caption(s: Step | undefined): [string, string] {
  if (!s) return ["", ""];
  const ts = s.touches ?? [];
  const n = s.weights_changed;
  switch (s.kind) {
    case "start":
      return [`Task given to ${s.agent ?? "the agent"}`, (s.text ?? "").split("\n")[0].slice(0, 160)];
    case "think":
      return ["Agent is reasoning", ((s.text || "").trim().split("\n")[0] || `calls ${s.calls?.map((c) => c.name).join(", ")}`).slice(0, 170)];
    case "violations":
      return ["Sandbox refused file / system access", `${s.items?.length ?? 0} attempt(s) blocked`];
    case "end": {
      const r = s.result ?? {};
      return [`Run ended: ${r.stop_reason ?? ""}`.slice(0, 120), `${(r.weights_changed as number | undefined)?.toLocaleString() ?? 0} weights changed · ${summarize(ts, "edit")}`];
    }
  }
  const edit = summarize(ts, "edit") || summarize(ts, "state");
  const read = summarize(ts, "read");
  const steer = summarize(ts, "steer");
  const tool = s.tool ?? "";
  const a = (s.args ?? {}) as Record<string, any>;
  if (s.error) return [`${tool} failed`, (s.output ?? "").split("\n").filter(Boolean).slice(-1)[0]?.slice(0, 160) ?? ""];
  if (tool === "python") {
    if (edit) return ["Editing weights", `${edit}${n ? ` · ${n.toLocaleString()} weights changed so far` : ""}`];
    if (read) return ["Reading weights", read];
    return ["Running Python", (a.code ?? "").split("\n")[0].slice(0, 160)];
  }
  if (tool === "generate") return [`Forward pass: generating for ${a.prompts?.length ?? "?"} prompt(s)`, `${s.forward_left ?? "?"} of ${s.budget ?? "?"} passes left`];
  if (tool === "edit_report") return ["Checking the edit", `${n?.toLocaleString() ?? 0} weights changed · ${edit}`];
  if (tool === "submit") return ["Submitted the edit", String(a.explanation ?? "").slice(0, 170)];
  if (tool === "steer_test") return ["Steering preview (no weights changed)", steer];
  if (tool === "logit_lens") return ["Logit lens: what each layer predicts", JSON.stringify(a.prompt ?? "").slice(0, 120)];
  if (tool === "attention") return [`Attention patterns at layer ${a.layer}`, JSON.stringify(a.prompt ?? "").slice(0, 120)];
  if (tool === "neuron_logits") return [`What neuron ${a.layer}.${a.neuron} writes`, "top tokens it pushes up / down"];
  if (tool === "writers_of") return [`Which neurons write direction ${JSON.stringify(a.direction)}`, read];
  if (tool === "contrast_direction") return [`Finding a direction at layer ${a.layer}`, `${a.name}: A minus B`];
  return [tool, read || steer || ""];
}

export function fmtTime(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m${String(s).padStart(2, "0")}s`;
}

export const pct = (x?: number) => (x === undefined || x === null ? "–" : `${Math.round(x * 100)}%`);
