export type Part = "embed" | "attn" | "mlp" | "norm" | "head" | "resid" | "model";
export type Mode = "read" | "edit" | "steer" | "run" | "state";

export type Touch = {
  part: Part;
  layer: number | null;
  mode: Mode;
  index?: number[];
  n?: number;
  labels?: string[];
  guess?: boolean;
  tensor?: string;
};

export type Step = {
  i: number;
  kind: "start" | "think" | "tool" | "violations" | "end";
  turn?: number | null;
  t: number;
  forward_left: number | null;
  budget: number | null;
  tokens: number;
  text?: string;
  calls?: { name: string; args: unknown }[];
  tool?: string;
  args?: Record<string, unknown>;
  output?: string;
  touches?: Touch[];
  weights_changed?: number | null;
  error?: boolean;
  items?: unknown[];
  result?: Record<string, unknown>;
  agent?: string;
  target?: string;
  system?: string;
  tools?: string[];
  // studio-only extras
  lens?: LensTable;
  attention?: AttnGrid;
  seconds?: number;
};

export type Arch = {
  model: string;
  n_layers: number;
  d_model: number;
  d_mlp: number;
  vocab: number;
  layer_types: string[];
  n_heads: number;
  n_kv_heads: number;
  linear_heads: number;
  tied_embeddings: boolean;
};

export type Evals = Record<string, number> | null;

export type RunInfo = {
  id: string;
  exp: string;
  cell: string;
  agent: string | null;
  live: boolean;
  stop_reason: string | null;
  weights_changed: number | null;
  spent_usd: number | null;
  eval: Evals;
  baseline: Evals;
  mtime: number;
};

export type RunData = {
  id: string;
  arch: Arch;
  steps: Step[];
  target: string | null;
  budget: number | null;
  final_diff: Record<string, { n: number; rows?: number[]; cols?: number[] }> | null;
  eval: Evals;
  baseline: Evals;
};

export type LensTable = { layers: number[]; rows: { tok: string; p: number }[][] };
export type AttnGrid = { layer: number; tokens: string[]; heads: number[][][] };
