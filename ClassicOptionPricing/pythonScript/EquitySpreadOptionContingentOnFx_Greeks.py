"""
Greeks for the Equity Spread Option Contingent on FX
=====================================================
Bump-and-reprice sensitivities.  The same MC seed is used for base and
every bumped run so differences are driven only by the parameter change.

Payoff recap
------------
  Notional * 1{EURUSD_T < 0.99 * EURUSD_0}
           * max(SPX_T/SPX_0 - SX5E_T/SX5E_0, 0)

Bump conventions  (initial spot levels are NEVER changed)
---------------------------------------------------------
  EQUITY DELTA   bump +1% on spx_fwd / sx5e_fwd
  EQUITY VEGA    bump +10 bp on spx_vol / sx5e_vol;
                 vega reported as ΔPV / 10  (i.e. per 1 bp)
  FX DELTA       bump +1% on eurusd_fwd
  FX VEGA        bump +10 bp on eurusd_vol;
                 vega reported as ΔPV / 10  (i.e. per 1 bp)
  CORRELATION    bump +10 bp on correlation (off-diagonal elements);
                 rho01 reported as ΔPV / 10  (i.e. per 1 bp)

Output fields (per Greek)
-------------------------
  dollar_pnl   = PV(bumped) – PV(base)   [$]
  pct_notional = dollar_pnl / Notional   [%]
"""

from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from EquitySpreadOptionContingentOnFx import load_params, price_relative_perf_fx_trigger_mc

# ---------------------------------------------------------------------------
# Load all parameters from YAML  (single source of truth)
# ---------------------------------------------------------------------------
_yaml_path = Path(__file__).parent / "EquitySpreadOptionContingentOnFx" / "params.yaml"
with open(_yaml_path) as _f:
    _cfg = yaml.safe_load(_f)

DEFAULT_PARAMS: Dict = load_params(_yaml_path)

# Bump sizes
_bumps = _cfg["greeks"]["bumps"]
EQUITY_FWD_BUMP_PCT = _bumps["equity_fwd_pct"]
FX_FWD_BUMP_PCT     = _bumps["fx_fwd_pct"]
VOL_BUMP_BPS        = _bumps["vol_bps"]
CORR_BUMP_BPS       = _bumps["corr_bps"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _price(**overrides) -> float:
    params = {**DEFAULT_PARAMS, **overrides}
    return price_relative_perf_fx_trigger_mc(**params)["price"]


def _greek_row(label: str, bump_desc: str, pv_bumped: float,
               base_pv: float, notional: float,
               per_bp_divisor: float = 1.0) -> Dict:
    """
    per_bp_divisor: divide raw ΔPV by this to get a per-unit (per-bp) figure.
    For vega bumped at 10 bp, pass per_bp_divisor=10 so output is per 1 bp.
    """
    raw_dv = pv_bumped - base_pv
    dv     = raw_dv / per_bp_divisor
    return {
        "label":            label,
        "bump":             bump_desc,
        "pv_bumped":        pv_bumped,
        "raw_dollar_pnl":   raw_dv,                    # full bump ΔPV
        "dollar_pnl":       dv,                         # per-unit ΔPV
        # "pct_notional":     dv / notional,
        "pct_base_pv":     dv / base_pv,
        "per_bp_divisor":   per_bp_divisor,
    }


# ---------------------------------------------------------------------------
# 1. Equity Delta  — bump spx_fwd / sx5e_fwd only
# ---------------------------------------------------------------------------
def equity_delta(base_pv: float,
                 bump_pct: float = EQUITY_FWD_BUMP_PCT) -> Dict:
    """
    Bump SPX and/or SX5E forwards by +bump_pct.
    Initial spot levels (spx_0, sx5e_0) are unchanged.
    """
    p    = DEFAULT_PARAMS
    ntl  = p["notional"]
    desc = f"+{bump_pct*100:.1f}% equity fwd"

    pv_spx   = _price(spx_fwd  = p["spx_fwd"]  * (1 + bump_pct))
    pv_sx5e  = _price(sx5e_fwd = p["sx5e_fwd"] * (1 + bump_pct))
    pv_joint = _price(spx_fwd  = p["spx_fwd"]  * (1 + bump_pct),
                      sx5e_fwd = p["sx5e_fwd"] * (1 + bump_pct))
    pv_joint_beta_adj = _price(spx_fwd  = p["spx_fwd"]  * (1 +       bump_pct),
                               sx5e_fwd = p["sx5e_fwd"] * (1 + 0.4 * bump_pct))

    return {
        "spx_delta":   _greek_row("SPX Delta  (+1% SPX fwd)",           desc, pv_spx,   base_pv, ntl),
        "sx5e_delta":  _greek_row("SX5E Delta (+1% SX5E fwd)",          desc, pv_sx5e,  base_pv, ntl),
        "joint_delta": _greek_row("Joint Equity Delta (+1% both fwds)", desc, pv_joint, base_pv, ntl),
        "joint_delta_beta_adj": _greek_row("Joint Equity Delta (+1% SPX and +0.4% SX5E)", desc, pv_joint_beta_adj, base_pv, ntl),
    }


# ---------------------------------------------------------------------------
# 2. Equity Vega  — bump spx_vol / sx5e_vol by VOL_BUMP_BPS, report per 1 bp
# ---------------------------------------------------------------------------
def equity_vega(base_pv: float,
                bump_bps: float = VOL_BUMP_BPS) -> Dict:
    """
    Bump SPX and/or SX5E implied vol by +bump_bps bp, then divide ΔPV by
    bump_bps so the result is expressed as vega per 1 bp.
    """
    bump = bump_bps * 1e-4
    p    = DEFAULT_PARAMS
    ntl  = p["notional"]
    desc = f"+{bump_bps} bp vol bump  (ΔPV ÷ {bump_bps} = per 1 bp)"

    pv_spx   = _price(spx_vol  = p["spx_vol"]  + bump)
    pv_sx5e  = _price(sx5e_vol = p["sx5e_vol"] + bump)
    pv_joint = _price(spx_vol  = p["spx_vol"]  + bump,
                      sx5e_vol = p["sx5e_vol"] + bump)

    return {
        "spx_vega":   _greek_row("SPX Vega  (per 1bp)",           desc, pv_spx,   base_pv, ntl, bump_bps),
        "sx5e_vega":  _greek_row("SX5E Vega (per 1bp)",           desc, pv_sx5e,  base_pv, ntl, bump_bps),
        "joint_vega": _greek_row("Joint Equity Vega (per 1bp)",   desc, pv_joint, base_pv, ntl, bump_bps),
    }


# ---------------------------------------------------------------------------
# 3. FX Delta  — bump eurusd_fwd only
# ---------------------------------------------------------------------------
def fx_delta(base_pv: float,
             bump_pct: float = FX_FWD_BUMP_PCT) -> Dict:
    """
    Bump EURUSD forward by +bump_pct.
    eurusd_0 (trigger anchor) is unchanged.
    A higher forward means the terminal EURUSD distribution shifts up,
    making it harder to breach the fixed trigger => PV falls.
    """
    p    = DEFAULT_PARAMS
    ntl  = p["notional"]
    desc = f"+{bump_pct*100:.1f}% EURUSD fwd (trigger fixed)"

    pv_up = _price(eurusd_fwd = p["eurusd_fwd"] * (1 + bump_pct))

    return _greek_row(
        f"FX Delta (+{bump_pct*100:.1f}% EURUSD fwd)",
        desc, pv_up, base_pv, ntl,
    )


# ---------------------------------------------------------------------------
# 4. FX Vega  — bump eurusd_vol by VOL_BUMP_BPS, report per 1 bp
# ---------------------------------------------------------------------------
def fx_vega(base_pv: float,
            bump_bps: float = VOL_BUMP_BPS) -> Dict:
    """
    Bump EURUSD implied vol by +bump_bps bp, then divide ΔPV by bump_bps
    so the result is expressed as vega per 1 bp.
    """
    bump = bump_bps * 1e-4
    p    = DEFAULT_PARAMS
    ntl  = p["notional"]
    desc = f"+{bump_bps} bp vol bump  (ΔPV ÷ {bump_bps} = per 1 bp)"

    pv_up = _price(eurusd_vol = p["eurusd_vol"] + bump)

    return _greek_row(
        f"FX Vega (per 1 bp EURUSD vol)",
        desc, pv_up, base_pv, ntl, bump_bps,
    )


# ---------------------------------------------------------------------------
# 5. Correlation Rho01 — bump correlation pairs by CORR_BUMP_BPS, report per 1 bp
# ---------------------------------------------------------------------------
def correlation_rho01(base_pv: float,
                      bump_bps: float = CORR_BUMP_BPS) -> Dict:
    """
    Bump each correlation pair by +bump_bps bp (0.001 in absolute terms),
    then divide ΔPV by bump_bps so the result is per 1 bp correlation shift.

    Three pairs:
      - SPX vs SX5E
      - SPX vs EURUSD
      - SX5E vs EURUSD

    Also compute joint bump (all three pairs up by bump_bps).
    """
    bump = bump_bps * 1e-4
    p    = DEFAULT_PARAMS
    ntl  = p["notional"]
    desc = f"+{bump_bps} bp corr bump  (ΔPV ÷ {bump_bps} = per 1 bp)"

    # Individual bumps
    pv_spx_sx5e = _price(corr_spx_sx5e = p["corr_spx_sx5e"] + bump)
    pv_spx_eur  = _price(corr_spx_eurusd = p["corr_spx_eurusd"] + bump)
    pv_sx5e_eur = _price(corr_sx5e_eurusd = p["corr_sx5e_eurusd"] + bump)

    # Joint bump (all three correlations up)
    pv_joint = _price(
        corr_spx_sx5e    = p["corr_spx_sx5e"]    + bump,
        corr_spx_eurusd  = p["corr_spx_eurusd"]  + bump,
        corr_sx5e_eurusd = p["corr_sx5e_eurusd"] + bump,
    )

    return {
        "spx_sx5e":  _greek_row("Corr(SPX, SX5E) Rho01 (per 1bp)",   desc, pv_spx_sx5e, base_pv, ntl, bump_bps),
        "spx_eur":   _greek_row("Corr(SPX, EURUSD) Rho01 (per 1bp)", desc, pv_spx_eur,  base_pv, ntl, bump_bps),
        "sx5e_eur":  _greek_row("Corr(SX5E, EURUSD) Rho01 (per 1bp)",desc, pv_sx5e_eur, base_pv, ntl, bump_bps),
        "joint":     _greek_row("Joint Corr Rho01 (all +1bp)",        desc, pv_joint,    base_pv, ntl, bump_bps),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def compute_all_greeks(verbose: bool = True) -> Dict:
    """Compute base PV then all Greeks. Returns a nested result dict."""
    base_result = price_relative_perf_fx_trigger_mc(**DEFAULT_PARAMS)
    base_pv     = base_result["price"]
    notional    = DEFAULT_PARAMS["notional"]

    results = {
        "base_pv":      base_pv,
        "base_stderr":  base_result["std_error"],
        "notional":     notional,
        "equity_delta": equity_delta(base_pv),
        "equity_vega":  equity_vega(base_pv),
        "fx_delta":     fx_delta(base_pv),
        "fx_vega":      fx_vega(base_pv),
        "corr_rho01":   correlation_rho01(base_pv),
    }

    if verbose:
        _print_results(results)

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def _print_results(r: Dict) -> None:
    notional    = r["notional"]
    base_pv     = r["base_pv"]
    base_stderr = r["base_stderr"]
    SEP  = "=" * 70
    THIN = "─" * 70

    def _row(g: Dict, is_vega: bool = False) -> None:
        print(f"  {g['label']}")
        print(f"    Bump           : {g['bump']}")
        print(f"    PV bumped  ($) : ${g['pv_bumped']:>13,.2f}")
        if is_vega:
            divisor = int(g["per_bp_divisor"])
            print(f"    Raw ΔPV    ($) : ${g['raw_dollar_pnl']:>+13,.2f}  (full {divisor} bp move)")
        print(f"    ΔPV/bp     ($) : ${g['dollar_pnl']:>+13,.2f}  (per 1 bp)")
        # print(f"    ΔPV/bp (% Ntl) : {g['pct_notional']:>+11.4%}  (per 1 bp)")
        print(f"    ΔPV/bp (% Base PV) : {g['pct_base_pv']:>+11.4%}  (per 1 bp)")
        print()

    def _delta_row(g: Dict) -> None:
        print(f"  {g['label']}")
        print(f"    Bump           : {g['bump']}")
        print(f"    PV bumped  ($) : ${g['pv_bumped']:>13,.2f}")
        print(f"    ΔPV        ($) : ${g['dollar_pnl']:>+13,.2f}")
        # print(f"    ΔPV    (% Ntl) : {g['pct_notional']:>+11.4%}")
        print(f"    ΔPV    (% Base PV) : {g['pct_base_pv']:>+11.4%}")
        print()

    print(SEP)
    print("      Exotic Equity Spread Option Contingent on FX – Greeks")
    print("      (Initial spot levels unchanged in all bumps)")
    print(SEP)
    print(f"  Base PV            : ${base_pv:>13,.2f}")
    print(f"  Base PV / Notional : {base_pv/notional:>11.4%}")
    print(f"  MC Std Error       : ${base_stderr:>13,.2f}")
    print(f"  Notional           : ${notional:>13,.2f}")
    print()

    print(THIN)
    print("  EQUITY DELTA  —  +1% on equity fwd  (spx_0 / sx5e_0 unchanged)")
    print(THIN)
    for key in ("spx_delta", "sx5e_delta", "joint_delta", "joint_delta_beta_adj"):
        _delta_row(r["equity_delta"][key])

    print(THIN)
    print(f"  EQUITY VEGA  --  bump +{VOL_BUMP_BPS} bp, vega = DPV / {VOL_BUMP_BPS}  (per 1 bp)")
    print(THIN)
    for key in ("spx_vega", "sx5e_vega", "joint_vega"):
        _row(r["equity_vega"][key], is_vega=True)

    print(THIN)
    print("  FX DELTA  —  +1% on EURUSD fwd  (eurusd_0 / trigger unchanged)")
    print("  EUR fwd up => distribution shifts away from trigger => PV falls")
    print(THIN)
    _delta_row(r["fx_delta"])

    print(THIN)
    print(f"  FX VEGA  --  bump +{VOL_BUMP_BPS} bp, vega = DPV / {VOL_BUMP_BPS}  (per 1 bp)")
    print("  Higher FX vol => fatter tails => more paths breach trigger => PV rises")
    print(THIN)
    _row(r["fx_vega"], is_vega=True)

    print(THIN)
    print(f"  CORRELATION RHO01  --  bump +{CORR_BUMP_BPS} bp, rho01 = DPV / {CORR_BUMP_BPS}  (per 1 bp)")
    print("  Higher correlation => more co-movement between risk factors")
    print(THIN)
    for key in ("spx_sx5e", "spx_eur", "sx5e_eur", "joint"):
        _row(r["corr_rho01"][key], is_vega=True)

    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compute_all_greeks(verbose=True)
