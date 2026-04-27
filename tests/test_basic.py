"""Synthetic-profile validation tests."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from iconremap import meteo, vertical


def test_constant_profile():
    """No shift, constant T → output equals input."""
    nlev, ncell = 5, 3
    z_src = np.tile(np.linspace(15000, 100, nlev)[:, None], (1, ncell))[None, ...]
    z_tgt = z_src.copy()
    t = np.full((1, nlev, ncell), 290.0, dtype=np.float32)
    out = vertical.interpolate_field(t, z_src, z_tgt, "T")
    assert np.allclose(out, 290.0), f"expected 290 everywhere, got {out}"
    print("✓ constant profile preserved")


def test_lapse_extrapolation_below():
    """LAM surface 100 m above source surface → T should drop by ~0.65 K."""
    nlev = 5
    z_src = np.array([15000, 5000, 2000, 1000, 500.0])[None, :, None]  # (1, 5, 1)
    z_tgt = z_src - 100.0  # shift target downward (= LAM surface lower → easier? wait)

    # Actually if target_z is LOWER than source_z, target is in deeper terrain →
    # for the LOWEST target level (was at 500m source, now 400m target), this is BELOW the source bottom.
    t_src = (290 - meteo.LAPSE_STD * z_src).astype(np.float32)
    out = vertical.interpolate_field(t_src, z_src, z_tgt, "T")
    # Expected T at 400 m via extrapolation from T at 500 m using lapse:
    expected_bottom = (290 - meteo.LAPSE_STD * 400)
    actual_bottom = out[0, -1, 0]
    err = abs(actual_bottom - expected_bottom)
    assert err < 0.1, f"bottom T extrapolation off: expected {expected_bottom:.2f}, got {actual_bottom:.2f}"
    print(f"✓ lapse extrapolation below: err = {err:.4f} K")


def test_pressure_hydrostatic_below():
    """Hydrostatic pressure extrapolation."""
    z_src = np.array([1000, 500.0])[None, :, None]
    z_tgt = np.array([1000, 200.0])[None, :, None]   # bottom 300 m below source bottom
    t_src = np.array([280, 285.0])[None, :, None]
    p_src = np.array([90000, 95000.0])[None, :, None]
    out = vertical.interpolate_field(p_src, z_src, z_tgt, "P", t_src_full=t_src)
    # Expected: P at 200 m via P_500 * exp(g*300/(R*T))
    expected = 95000 * np.exp(meteo.G * 300 / (meteo.R_D * 285.0))
    actual = out[0, -1, 0]
    rel_err = abs(actual - expected) / expected
    assert rel_err < 0.001, f"pressure off by {rel_err*100:.3f}%"
    print(f"✓ hydrostatic pressure below: rel err = {rel_err*100:.3f}%")


def test_qv_zero_floor():
    """QV extrapolation should never go negative."""
    z_src = np.array([1000, 500.0])[None, :, None]
    z_tgt = np.array([1000, 200.0])[None, :, None]
    qv_src = np.array([1e-3, 1e-9])[None, :, None]
    p_src  = np.array([90000, 95000.0])[None, :, None]
    t_src  = np.array([280, 285.0])[None, :, None]
    out = vertical.interpolate_field(qv_src, z_src, z_tgt, "QV",
                                      p_src_full=p_src, t_src_full=t_src)
    assert (out >= 0).all(), "qv went negative"
    print("✓ qv non-negative below surface")


def test_qc_zero_below_surface():
    """QC should be zero below source surface."""
    z_src = np.array([1000, 500.0])[None, :, None]
    z_tgt = np.array([1000, 200.0])[None, :, None]
    qc_src = np.array([0.001, 0.002])[None, :, None]
    out = vertical.interpolate_field(qc_src, z_src, z_tgt, "QC")
    assert out[0, -1, 0] == 0.0, f"expected 0 below surface, got {out[0, -1, 0]}"
    print("✓ qc zero below surface")


def test_shift_hhl():
    """HHL shift produces target HSURF == EXTPAR HSURF."""
    nlev_intf = 5
    ncell = 3
    hhl_src = np.tile(np.linspace(15000, 100, nlev_intf)[:, None], (1, ncell))[None, ...]
    hsurf_src = hhl_src[:, -1, :].mean(axis=0)
    hsurf_tgt = np.array([200.0, 100.0, 50.0])
    hhl_tgt = vertical.shift_hhl_to_target_topo(hhl_src, hsurf_src, hsurf_tgt)
    actual_bottom = hhl_tgt[:, -1, :].mean(axis=0)
    assert np.allclose(actual_bottom, hsurf_tgt), \
        f"shift mismatch: expected {hsurf_tgt}, got {actual_bottom}"
    print("✓ HHL shift gives correct target HSURF")


if __name__ == "__main__":
    test_constant_profile()
    test_lapse_extrapolation_below()
    test_pressure_hydrostatic_below()
    test_qv_zero_floor()
    test_qc_zero_below_surface()
    test_shift_hhl()
    print("\nAll tests passed.")
