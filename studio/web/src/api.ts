import type { Arch, RunData, RunInfo } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

// Static build (`npm run build:blog`): no backend; replays come from files written by
// scripts/export_replays.py next to the page. Replay only.
export const STATIC = import.meta.env.VITE_STATIC === "1";
const REPLAYS: string = import.meta.env.VITE_REPLAYS ?? "replays/";

type Index = { arch: Arch; runs: (RunInfo & { file: string })[] };
let index: Promise<Index> | null = null;
const loadIndex = () => (index ??= fetch(`${REPLAYS}index.json`).then((r) => j<Index>(r)));

const live = {
  runs: () => fetch("/api/runs").then((r) => j<RunInfo[]>(r)),
  run: (id: string) => fetch(`/api/run/${id}`).then((r) => j<RunData>(r)),
  arch: (model?: string) => fetch(`/api/arch${model ? `?model=${encodeURIComponent(model)}` : ""}`).then((r) => j<Arch>(r)),
  studioStatus: () => fetch("/api/studio/status").then((r) => j<Record<string, any>>(r)),
  studio: (action: string, body: unknown) =>
    fetch(`/api/studio/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    }).then((r) => j<Record<string, any>>(r)),
};

const fromFiles: typeof live = {
  runs: () => loadIndex().then((x) => x.runs),
  run: (id) => loadIndex().then((x) => {
    const r = x.runs.find((r) => r.id === id);
    if (!r) throw new Error(`no replay ${id}`);
    return fetch(`${REPLAYS}${r.file}`).then((res) => j<RunData>(res));
  }),
  arch: () => loadIndex().then((x) => x.arch),
  studioStatus: () => Promise.resolve({ enabled: false, static: true }),
  studio: () => Promise.reject(new Error("the static replay has no model")),
};

export const api = STATIC ? fromFiles : live;
