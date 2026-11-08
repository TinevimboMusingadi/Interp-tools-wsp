import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from wsp.harness.env import Env  # noqa: E402
from wsp.tools import TOOLSETS  # noqa: E402
from wsp.tools.build import load_library  # noqa: E402
from wsp.tools.sae import load_sae, train_topk_sae  # noqa: E402


@pytest.fixture
def env():
    from tiny import tiny_gpt2

    e = Env(tiny_gpt2(), forward_budget=20, tools=TOOLSETS["c1"]())
    load_library(e, None)
    return e


def test_free_tools_cost_nothing(env):
    out = env.call("neuron_logits", {"layer": 1, "neuron": 3, "k": 3})
    assert "promotes" in out and env.forward_used == 0
    env.namespace["ws"]["d"] = torch.randn(32)
    assert "promotes" in env.call("direction_logits", {"direction": "d"})
    assert "neuron" in env.call("writers_of", {"direction": "d", "k": 3})
    assert env.forward_used == 0


def test_activation_tools_charge_one_pass_each(env):
    assert "shape" in env.call("capture", {"prompts": ["a b c", "d"], "points": ["resid_post.1", "mlp_out.0"]})
    assert "L 0" in env.call("logit_lens", {"prompt": "hello world"})
    assert "head 1" in env.call("attention", {"prompt": "hello world again", "layer": 0})
    out = env.call("contrast_direction", {"prompts_a": ["yes", "sure"], "prompts_b": ["no", "never"],
                                          "layer": 1, "name": "yn"})
    assert "separation" in out and env.namespace["ws"]["yn"].shape == (32,)
    assert "[1]" in env.call("steer_test", {"prompts": ["hi", "yo"], "direction": "yn", "layer": 1,
                                            "max_new_tokens": 3})
    assert "[0]" in env.call("steer_test", {"prompts": ["hi"], "direction": "yn", "layer": 1,
                                            "method": "ablate", "max_new_tokens": 3})
    assert env.forward_used == 6


def test_agent_code_cannot_bypass_budget_with_engine(env):
    out = env.call("python", {"code": "from wsp.tools.interp import _capture\n"
                                      "env = None\nfrom interp_engine import capture\n"
                                      "capture(model, model.to_tokens('x'), ['resid_post.0'])"})
    assert "costs a forward pass" in out


def test_sae_roundtrip_and_tools(env, tmp_path):
    x = torch.randn(500, 32)
    with env.gradients_allowed():
        sae, stats = train_topk_sae(x, "resid_post.1", d_sae=64, k=4, steps=50)
    assert stats["fvu"] < 1.5 and sae.encode(x[:3]).count_nonzero(dim=-1).max() <= 4
    sae.save(tmp_path / "s.npz")
    back = load_sae(tmp_path / "s.npz")
    assert back.point == "resid_post.1" and back.k == 4
    assert torch.allclose(back.encode(x[:2]), sae.encode(x[:2]))
    e2 = Env(env.model, forward_budget=5, tools=TOOLSETS["c1_sae"]())
    load_library(e2, None)
    e2.namespace["saes"] = {"s": back}
    assert "prompt 0" in e2.call("sae_features", {"prompts": ["hello"], "sae": "s", "k": 3})
    assert "stored" in e2.call("sae_feature_info", {"sae": "s", "feature": 2})


def test_gradients_still_blocked_after_harness_training(env):
    with env.gradients_allowed():
        train_topk_sae(torch.randn(50, 32), "resid_post.0", d_sae=8, k=2, steps=2)
    assert "gradients are not allowed" in env.call(
        "python", {"code": "x = torch.ones(1, requires_grad=True)\nx.sum().backward()"})


def test_c2_probe_sae_and_library_persist(tmp_path):
    from tiny import tiny_gpt2

    e = Env(tiny_gpt2(), forward_budget=10, tools=TOOLSETS["c2"]())
    load_library(e, str(tmp_path))
    out = e.call("train_probe", {"prompts": ["yes"] * 6 + ["no"] * 6, "labels": [1] * 6 + [0] * 6,
                                 "layer": 1, "name": "p"})
    assert "held-out acc" in out and "p" in e.namespace["ws"]
    out = e.call("train_sae", {"prompts": ["one two three", "four five"], "layer": 1, "name": "mine",
                               "d_sae": 16, "k": 2})
    assert "SAE 'mine'" in out and (tmp_path / "saes" / "mine.npz").exists()
    code = "def main(text):\n    return run_tool('logit_lens', prompt=text, layers=[1])"
    assert "saved" in e.call("save_tool", {"name": "lens1", "description": "lens at layer 1", "code": code})
    used = e.forward_used
    assert "L 1" in e.call("python", {"code": "print(lib.lens1.main('hi'))"})
    assert e.forward_used == used + 1  # library code still pays for forward passes
    # a fresh run sees the persisted library
    e2 = Env(e.model, forward_budget=1, tools=TOOLSETS["c2"]())
    load_library(e2, str(tmp_path))
    listing = e2.call("list_library", {})
    assert "lens1" in listing and "mine" in listing


def test_unknown_sae_is_reported_not_crash(env):
    e = Env(env.model, forward_budget=2, tools=TOOLSETS["c1_sae"]())
    load_library(e, None)
    out = e.call("sae_features", {"prompts": ["x"], "sae": "nope"})
    assert "KeyError" in out and e.forward_used == 0


def test_two_model_env_routes_calls_and_shares_budget():
    from tiny import tiny_gpt2

    e = Env({"A": tiny_gpt2(), "B": tiny_gpt2()}, forward_budget=3, tools=TOOLSETS["c1"]())
    load_library(e, None)
    assert "model" in e.tools["capture"].spec.parameters["properties"]
    assert "model" not in e.tools["submit"].spec.parameters["properties"]
    e.call("python", {"code": "models['B'].hf_model.lm_head.weight.data[0, 0] += 1"})
    rep = e.call("edit_report", {})
    assert "[A] 0 of" in rep and "[B] 1 of" in rep
    e.call("logit_lens", {"prompt": "hi", "model": "A"})
    e.call("logit_lens", {"prompt": "hi", "model": "B"})
    assert e.forward_used == 2
    assert "unknown model" in e.call("logit_lens", {"prompt": "hi", "model": "C"})
    assert e.status()["per_model_changed"] == {"A": 0, "B": 1}
