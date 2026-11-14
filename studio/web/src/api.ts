import type { Arch, RunData, RunInfo } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
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
