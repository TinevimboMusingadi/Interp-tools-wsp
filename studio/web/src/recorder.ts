// Video export. "Whole screen": the browser asks which tab/window to capture (pick this tab), so the
// video has all three panels. "Map only": records the network canvas directly, no prompt.
export type Recording = { stop: () => void };

function save(chunks: Blob[], name: string) {
  const blob = new Blob(chunks, { type: "video/webm" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function record(stream: MediaStream, name: string, onStop: () => void): Recording {
  const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    save(chunks, name);
    onStop();
  };
  stream.getVideoTracks()[0]?.addEventListener("ended", () => rec.state !== "inactive" && rec.stop());
  rec.start(500);
  return { stop: () => rec.state !== "inactive" && rec.stop() };
}

export async function recordScreen(name: string, onStop: () => void): Promise<Recording> {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 30 },
    audio: false,
    // @ts-expect-error - Chrome hint to offer the current tab first
    preferCurrentTab: true,
  });
  return record(stream, name, onStop);
}

export function recordMap(name: string, onStop: () => void): Recording {
  const c = document.getElementById("netmap-canvas") as HTMLCanvasElement;
  return record(c.captureStream(30), name, onStop);
}
