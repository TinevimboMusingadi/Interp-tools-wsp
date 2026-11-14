"""Record a replay of one run to a video file, hands-free (for the blog).

    python -m wsp.studio --runs runs            # in another terminal
    python scripts/record_run.py p2_loop_fix/c0_b10_carbon_s0 --speed 4 --out blog/assets/carbon_b10.webm

Needs Playwright (``pip install playwright && playwright install chromium``). Without it, use the
"Record video" button in the page instead: it saves the same video from your own browser.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run id as listed in the Studio, e.g. p2_loop_fix/c0_b10_carbon_s0")
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--speed", type=float, default=4)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--out", default="runs/videos/run.webm")
    ap.add_argument("--max-minutes", type=float, default=15)
    a = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed: pip install playwright && playwright install chromium\n"
              "Or use the 'Record video' button in the Studio page.")
        return 1
    w, h = (int(x) for x in a.size.split("x"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_rec"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": w, "height": h}, record_video_dir=str(tmp),
                                  record_video_size={"width": w, "height": h})
        page = ctx.new_page()
        page.goto(a.url)
        page.wait_for_selector(".runsel")
        page.select_option(".runsel", a.run)
        page.wait_for_function("document.querySelector('.scrub input').max > 1")
        page.select_option(".player select", str(a.speed).rstrip("0").rstrip("."))
        page.click(".pbtn.play")
        t0 = time.time()
        while time.time() - t0 < a.max_minutes * 60:
            done = page.evaluate("(() => { const r = document.querySelector('.scrub input');"
                                 " return Number(r.value) >= Number(r.max); })()")
            if done:
                time.sleep(4)  # hold on the final state
                break
            time.sleep(1)
        video = page.video.path()
        ctx.close()
        browser.close()
    shutil.move(video, out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
