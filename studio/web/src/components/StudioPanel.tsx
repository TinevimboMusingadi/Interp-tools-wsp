// Hands-on mode: the same tools the agent gets, as forms. Each result becomes a step, so the map and
// the detail view show it exactly like a replayed agent action.
import { useState } from "react";
import { api } from "../api";
import type { Arch, Step, Touch } from "../types";

type Props = {
  arch: Arch;
  ws: Record<string, number[]>;
  busy: boolean;
  onResult: (s: Partial<Step>, state?: { touches: Touch[]; n_changed: number; undo: string[] } | null) => void;
  setBusy: (b: boolean) => void;
  undo: string[];
};

const SECTIONS = ["Chat", "Logit lens", "Attention", "Neuron", "Direction", "Edit", "Python"] as const;

export function StudioPanel({ arch, ws, busy, onResult, setBusy, undo }: Props) {
  const [sec, setSec] = useState<(typeof SECTIONS)[number]>("Chat");
  const [prompt, setPrompt] = useState("Darma can eat 20 peanuts in 15 seconds. At this same rate, how many peanuts could she eat in 6 minutes?");
  const [maxTok, setMaxTok] = useState(256);
  const [steerOn, setSteerOn] = useState(false);
  const [dir, setDir] = useState("");
  const [layer, setLayer] = useState(12);
  const [coeff, setCoeff] = useState(4);
  const [method, setMethod] = useState("additive");
  const [layers, setLayers] = useState("");
  const [neuron, setNeuron] = useState(0);
  const [pa, setPa] = useState("Solve: 3+4=\nCompute 9*2\nWhat is 15% of 80?");
  const [pb, setPb] = useState("The weather today\nMy favourite colour\nTell me about cats");
  const [name, setName] = useState("my_dir");
  const [tokens, setTokens] = useState("Wait, Wait, Alternatively, Alternatively");
  const [factor, setFactor] = useState(0.72);
  const [tensor, setTensor] = useState("model.layers.20.mlp.down_proj.weight");
  const [units, setUnits] = useState("0,1,2");
  const [axis, setAxis] = useState("in");
  const [code, setCode] = useState("E = model.hf_model.model.embed_tokens.weight\nprint(E.shape, E.dtype)");
  const dirs = Object.entries(ws).filter(([, s]) => s.length === 1 && s[0] === arch.d_model).map(([k]) => k);

  async function run(action: string, body: unknown, label: Partial<Step>) {
    setBusy(true);
    try {
      const r = await api.studio(action, body);
      if (r.error) onResult({ ...label, output: `error: ${r.error}`, error: true, touches: [] });
      else onResult({ ...label, output: r.output ?? JSON.stringify(r.record ?? r.records ?? r.undone ?? r.saved ?? "", null, 2), touches: r.touches ?? r.state?.touches ?? [], lens: r.lens, attention: r.attention, seconds: r.seconds }, r.state);
    } catch (e) {
      onResult({ ...label, output: String(e), error: true, touches: [] });
    } finally {
      setBusy(false);
    }
  }
  const tool = (name: string, args: Record<string, unknown>) => run("tool", { name, args }, { tool: name, args });
  const parseLayers = () => (layers.trim() ? layers.split(/[ ,]+/).filter(Boolean).map(Number) : null);
  const full = arch.layer_types.map((t, i) => (t === "full" ? i : -1)).filter((i) => i >= 0);

  return (
    <div className="panel-body studio">
      <div className="tabs">
        {SECTIONS.map((s) => <button key={s} className={s === sec ? "on" : ""} onClick={() => setSec(s)}>{s}</button>)}
      </div>
      {busy && <div className="busy">Running on the model… (CPU: a few seconds to a minute)</div>}
      {sec === "Chat" && (
        <div className="form">
          <label>Prompt</label>
          <textarea rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="row"><label>max tokens</label><input type="number" value={maxTok} onChange={(e) => setMaxTok(+e.target.value)} /></div>
          <div className="row"><label><input type="checkbox" checked={steerOn} onChange={(e) => setSteerOn(e.target.checked)} /> steer with a direction</label></div>
          {steerOn && (
            <>
              <div className="row"><label>direction</label>
                <select value={dir} onChange={(e) => setDir(e.target.value)}>
                  <option value="">(pick: build one under Direction)</option>
                  {dirs.map((d) => <option key={d}>{d}</option>)}
                </select></div>
              <div className="row"><label>layer</label><input type="number" min={0} max={arch.n_layers - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} /></div>
              <div className="row"><label>method</label>
                <select value={method} onChange={(e) => setMethod(e.target.value)}><option>additive</option><option>ablate</option></select></div>
              {method === "additive" && <div className="row"><label>strength {coeff}</label><input type="range" min={-16} max={16} step={0.5} value={coeff} onChange={(e) => setCoeff(+e.target.value)} /></div>}
            </>
          )}
          <button className="go" disabled={busy} onClick={() => run("chat", { prompt, max_new_tokens: maxTok, steer: steerOn && dir ? { direction: dir, layer, coeff, method } : null }, { tool: steerOn && dir ? "steer_test" : "generate", args: { prompts: [prompt] } })}>
            {busy ? "running…" : "Send"}
          </button>
        </div>
      )}
      {sec === "Logit lens" && (
        <div className="form">
          <label>Prompt</label>
          <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="row"><label>layers (blank = all)</label><input value={layers} onChange={(e) => setLayers(e.target.value)} placeholder="0 4 8 12 16 20 23" /></div>
          <button className="go" disabled={busy} onClick={() => tool("logit_lens", { prompt, layers: parseLayers(), k: 6 })}>Run lens</button>
        </div>
      )}
      {sec === "Attention" && (
        <div className="form">
          <label>Prompt</label>
          <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="row"><label>layer</label>
            <select value={layer} onChange={(e) => setLayer(+e.target.value)}>
              {full.map((L) => <option key={L} value={L}>{L} (full attention)</option>)}
            </select></div>
          <p className="hint">Only full-attention layers have attention patterns; the others are linear (Gated DeltaNet).</p>
          <button className="go" disabled={busy} onClick={() => tool("attention", { prompt, layer, k: 5 })}>Show attention</button>
        </div>
      )}
      {sec === "Neuron" && (
        <div className="form">
          <div className="row"><label>layer</label><input type="number" min={0} max={arch.n_layers - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} /></div>
          <div className="row"><label>neuron</label><input type="number" min={0} max={arch.d_mlp - 1} value={neuron} onChange={(e) => setNeuron(+e.target.value)} /></div>
          <button className="go" disabled={busy} onClick={() => tool("neuron_logits", { layer, neuron, k: 12 })}>What does it write?</button>
        </div>
      )}
      {sec === "Direction" && (
        <div className="form">
          <label>Prompts A (one per line)</label>
          <textarea rows={3} value={pa} onChange={(e) => setPa(e.target.value)} />
          <label>Prompts B</label>
          <textarea rows={3} value={pb} onChange={(e) => setPb(e.target.value)} />
          <div className="row"><label>layer</label><input type="number" value={layer} onChange={(e) => setLayer(+e.target.value)} /></div>
          <div className="row"><label>name</label><input value={name} onChange={(e) => setName(e.target.value)} /></div>
          <button className="go" disabled={busy} onClick={() => tool("contrast_direction", { prompts_a: lines(pa), prompts_b: lines(pb), layer, name })}>Build direction (A − B)</button>
          <div className="row"><label>direction</label>
            <select value={dir} onChange={(e) => setDir(e.target.value)}><option value="">pick</option>{dirs.map((d) => <option key={d}>{d}</option>)}</select></div>
          <div className="btnrow">
            <button disabled={busy || !dir} onClick={() => tool("direction_logits", { direction: dir, k: 12 })}>Tokens it promotes</button>
            <button disabled={busy || !dir} onClick={() => tool("writers_of", { direction: dir, k: 12 })}>Neurons that write it</button>
          </div>
        </div>
      )}
      {sec === "Edit" && (
        <div className="form">
          <h4>Scale embedding rows by token</h4>
          <p className="hint">The tied embedding is also the output layer: shrinking a row makes the model say that token less. Carbon's p2 edit used 0.72 on "Wait" / "Alternatively".</p>
          <div className="row"><label>tokens</label><input value={tokens} onChange={(e) => setTokens(e.target.value)} /></div>
          <div className="row"><label>factor {factor}</label><input type="range" min={0} max={2} step={0.02} value={factor} onChange={(e) => setFactor(+e.target.value)} /></div>
          <button className="go" disabled={busy} onClick={() => run("edit", { op: "scale", tokens: tokens.split(",").filter((t) => t.trim()) /* ", Wait" keeps its leading space: " Wait" is its own token */, factor, axis: "out" }, { tool: "edit", args: { tokens, factor } })}>Apply</button>
          <h4>Scale units of any tensor</h4>
          <div className="row"><label>tensor</label><input value={tensor} onChange={(e) => setTensor(e.target.value)} /></div>
          <div className="row"><label>units</label><input value={units} onChange={(e) => setUnits(e.target.value)} /></div>
          <div className="row"><label>axis</label><select value={axis} onChange={(e) => setAxis(e.target.value)}><option value="in">in (columns: MLP neurons in down_proj)</option><option value="out">out (rows)</option></select></div>
          <button disabled={busy} onClick={() => run("edit", { op: "scale", tensor, units: units.split(/[ ,]+/).filter(Boolean).map(Number), factor, axis }, { tool: "edit", args: { tensor, units, factor, axis } })}>Apply</button>
          <h4>Bake a direction ablation into the weights</h4>
          <div className="row"><label>direction</label><select value={dir} onChange={(e) => setDir(e.target.value)}><option value="">pick</option>{dirs.map((d) => <option key={d}>{d}</option>)}</select></div>
          <div className="row"><label>layers (blank = all)</label><input value={layers} onChange={(e) => setLayers(e.target.value)} /></div>
          <button disabled={busy || !dir} onClick={() => run("edit", { op: "ablate_direction", direction: dir, layers: parseLayers() }, { tool: "edit", args: { ablate: dir } })}>Remove it from every writer</button>
          <div className="btnrow">
            <button disabled={busy || !undo.length} onClick={() => run("undo", {}, { tool: "undo" })}>Undo{undo.length ? ` (${undo[undo.length - 1]})` : ""}</button>
            <button disabled={busy} onClick={() => run("revert", {}, { tool: "revert" })}>Revert all</button>
            <button disabled={busy} onClick={() => run("save", {}, { tool: "save" })}>Save weights</button>
          </div>
        </div>
      )}
      {sec === "Python" && (
        <div className="form">
          <p className="hint">Same namespace the agent gets: <code>model</code>, <code>torch</code>, <code>ws</code>, <code>run_tool(name, **args)</code>, <code>wsp.edit</code>. Edits here can be undone.</p>
          <textarea className="mono" rows={12} value={code} onChange={(e) => setCode(e.target.value)} />
          <button className="go" disabled={busy} onClick={() => tool("python", { code })}>Run</button>
        </div>
      )}
    </div>
  );
}

const lines = (s: string) => s.split("\n").map((x) => x.trim()).filter(Boolean);
