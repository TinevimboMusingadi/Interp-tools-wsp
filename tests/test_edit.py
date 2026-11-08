import pytest
import torch

from wsp.edit import (
    feature_edit,
    kl_on_text,
    orthogonalize_in,
    orthogonalize_out,
    rank1,
    scale_units,
    snapshot,
    weight_diff,
)

torch.manual_seed(0)
D_MODEL, D_MLP = 16, 40


def lin():
    return torch.nn.Linear(D_MLP, D_MODEL, bias=False)  # a "down projection": writes d_model


def test_orthogonalize_out_removes_direction_from_writes():
    m, v = lin(), torch.randn(D_MODEL)
    rec = orthogonalize_out(m.weight, v, target="down")
    out = m(torch.randn(50, D_MLP))
    assert torch.allclose(out @ (v / v.norm()), torch.zeros(50), atol=1e-5)
    assert rec.n_changed > 0 and rec.target == "down"


def test_orthogonalize_out_columns_touches_only_those_columns():
    m, v = lin(), torch.randn(D_MODEL)
    before = m.weight.detach().clone()
    rec = orthogonalize_out(m.weight, v, columns=[3, 7])
    untouched = [i for i in range(D_MLP) if i not in (3, 7)]
    assert torch.equal(m.weight[:, untouched], before[:, untouched])
    assert rec.n_changed <= 2 * D_MODEL and rec.params["n_columns"] == 2


def test_orthogonalize_in_makes_module_blind():
    m = torch.nn.Linear(D_MODEL, D_MLP, bias=False)
    v = torch.randn(D_MODEL)
    orthogonalize_in(m.weight, v)
    x = torch.randn(D_MODEL)
    assert torch.allclose(m(x), m(x + 3.0 * v), atol=1e-4)


def test_rank1_writes_u_when_input_along_v():
    m = lin()
    u, v = torch.randn(D_MODEL), torch.zeros(D_MLP)
    v[5] = 1.0
    x = torch.zeros(D_MLP)
    x[5] = 2.0
    base = m(x)
    rank1(m.weight, u, v, scale=0.5)
    assert torch.allclose(m(x) - base, u, atol=1e-5)


def test_scale_units_and_shape_errors():
    m = lin()
    before = m.weight.detach().clone()
    scale_units(m.weight, [0, 1], 2.0)
    assert torch.allclose(m.weight[:, :2], 2 * before[:, :2])
    with pytest.raises(ValueError):
        orthogonalize_out(m.weight, torch.randn(D_MLP))
    with pytest.raises(ValueError):
        orthogonalize_out(m.weight, torch.zeros(D_MODEL))


def test_feature_edit_partial_removal():
    m, v = lin(), torch.randn(D_MODEL)
    vu = v / v.norm()
    x = torch.randn(10, D_MLP)
    before = m(x) @ vu
    feature_edit(m.weight, v, mode="remove", strength=0.5)
    assert torch.allclose(m(x) @ vu, 0.5 * before, atol=1e-5)


def test_weight_diff_and_kl():
    m = lin()
    ref = snapshot(m.named_parameters())
    f_before = lambda t: torch.nn.functional.linear(t, ref["weight"])  # noqa: E731
    scale_units(m.weight, [4], 0.0)
    rep = weight_diff(ref, m.named_parameters())
    assert rep.n_changed == D_MODEL and rep.per_tensor == {"weight": D_MODEL}
    assert "weights changed" in rep.summary()
    kl = kl_on_text(f_before, m, [torch.randn(8, D_MLP)])
    assert kl > 0
    assert kl_on_text(m, m, [torch.randn(8, D_MLP)]) == pytest.approx(0.0, abs=1e-6)
