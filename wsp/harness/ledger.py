"""Token and dollar accounting. Every model call goes through a :class:`Budget`, which appends to
``runs/ledger.csv`` and raises :class:`BudgetExceeded` before a run can overspend.

Prices live in ``prices.yaml`` beside this file. They are **placeholders until verified** against the
Vertex AI and Gemini pricing pages; a model with no price entry is refused rather than counted as free.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PRICES_FILE = Path(__file__).with_name("prices.yaml")
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "runs" / "ledger.csv"
FIELDS = ["time", "run_id", "provider", "model", "input_tokens", "output_tokens", "cost_usd",
          "cache_read_tokens", "cache_write_tokens"]


class BudgetExceeded(RuntimeError):
    pass


def load_prices(path: Path = PRICES_FILE) -> dict[str, dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def cost_usd(model: str, input_tokens: int, output_tokens: int, prices: dict | None = None,
             cache_read: int = 0, cache_write: int = 0) -> float:
    """``input_tokens`` are the uncached input tokens; cache reads and writes are billed at their own
    rates (Claude reports them separately; Gemini's implicit cache is not split out, so it counts as
    plain input, the conservative side)."""
    prices = prices if prices is not None else load_prices()
    if model not in prices:
        raise KeyError(f"no price for {model!r}: add it to {PRICES_FILE.name} before running")
    p = prices[model]
    read = p.get("cache_read_per_mtok", p["input_per_mtok"])
    write = p.get("cache_write_per_mtok", p["input_per_mtok"])
    return (input_tokens * p["input_per_mtok"] + output_tokens * p["output_per_mtok"]
            + cache_read * read + cache_write * write) / 1e6


@dataclass
class Budget:
    """Caps for one agent run. ``max_usd`` and ``max_tokens`` are hard stops."""

    run_id: str
    max_usd: float
    max_tokens: int
    ledger_path: Path = DEFAULT_LEDGER
    spent_usd: float = 0.0
    tokens: int = 0
    prices: dict = field(default_factory=load_prices)

    def charge(self, provider: str, model: str, input_tokens: int, output_tokens: int,
               cache_read: int = 0, cache_write: int = 0) -> float:
        c = cost_usd(model, input_tokens, output_tokens, self.prices, cache_read, cache_write)
        self.spent_usd += c
        # the token cap counts every token the model processed, cached or not, so caps mean the same
        # thing across providers
        self.tokens += input_tokens + cache_read + cache_write + output_tokens
        self._append(provider, model, input_tokens, output_tokens, c, cache_read, cache_write)
        return c

    def check(self) -> None:
        if self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"{self.run_id}: ${self.spent_usd:.2f} >= cap ${self.max_usd:.2f}")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"{self.run_id}: {self.tokens:,} tokens >= cap {self.max_tokens:,}")

    def _append(self, provider: str, model: str, i: int, o: int, c: float, cr: int = 0, cw: int = 0) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        _upgrade_header(self.ledger_path)
        new = not self.ledger_path.exists()
        with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), self.run_id, provider, model, i, o, f"{c:.6f}", cr, cw])


def _upgrade_header(path: Path) -> None:
    """Ledgers written before the cache columns existed get them (as 0) so every row has one shape."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == FIELDS:
        return
    width = len(FIELDS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows[1:]:
            w.writerow(r + ["0"] * (width - len(r)))
