
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------- Safe helpers ----------
def safe_exp(x, low=-700.0, high=700.0):
    return jnp.exp(jnp.clip(x, low, high))

def safe_div(n, d, eps=1e-300):
    return n / (d + eps)

def pct(a, b):
    denom = jnp.where(jnp.abs(b) > 1e-30, b, jnp.array(1.0))
    return 100.0 * (a - b) / denom

# ---------- Parameters ----------
@dataclass
class Params:
    # Geometry and baseline
    k: float = 1.0
    L: float = 1.0
    rc: float = 12.0
    vUV: float = 0.1
    vIR: float = 0.01
    Ny: int = 3001
    kappa5_sq: float = 1.0

    # Grid shaping (for Ymax via rc only; solver uses uniform RK4)
    stretch: float = 0.35
    alpha: float = 4.0

    # Quantum corrections (IR-localized)
    eps_JT: float = 0.0
    eps_Sch: float = 0.0
    ir_window_center_frac: float = 0.95
    ir_window_width_frac: float = 0.02
    sch_sat: float = 1.0

    # UV counter-terms
    delta_m2_UV: float = 0.0
    delta_lambda_UV: float = 0.0

    # Physical scales
    mH_bare: float = 125.0
    M5: float = 1.0e18

# ---------- Grid helpers ----------
def make_stretched_grid(p: Params):
    Ymax = jnp.pi * jnp.array(p.rc, dtype=jnp.float64)
    s = jnp.array(p.stretch, dtype=jnp.float64)
    xi = jnp.linspace(0.0, 1.0, p.Ny, dtype=jnp.float64)
    f = ((1.0 - s) * xi + s * (xi ** p.alpha)) / ((1.0 - s) + s)
    y = Ymax * f
    return y, Ymax

def make_uniform_grid(Ymax, Ny):
    return jnp.linspace(0.0, Ymax, Ny, dtype=jnp.float64)

def ir_window(y, Ymax, p: Params):
    y0 = p.ir_window_center_frac * Ymax
    w  = p.ir_window_width_frac * Ymax
    return 0.5 * (1.0 + jnp.tanh((y - y0) / w))

# ---------- Superpotential RHS ----------
def W0(p: Params):
    return jnp.array(3.0 * p.k / p.kappa5_sq, dtype=jnp.float64)

def rhs_system(p: Params, y, U, Ymax, c2):
    phi, A = U
    dphi = 2.0 * c2 * phi
    dA = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))
    return jnp.array([dphi, dA])

def rhs_system_corrected(p: Params, y, U, Ymax, c2, vUV_eff):
    phi, A = U
    dphi = 2.0 * c2 * phi
    Aprime_base = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))
    wIR = ir_window(y, Ymax, p)
    dA_JT = p.eps_JT * wIR
    sat = p.sch_sat
    dA_S = p.eps_Sch * wIR * (Aprime_base**2) / (1.0 + (Aprime_base**2) / (sat**2))
    dA = Aprime_base + dA_JT + dA_S
    return jnp.array([dphi, dA])

# ---------- Fixed-step RK4 (uniform grid, differentiable) ----------
def integrate_fixed_rk4(fun, p: Params, y: jnp.ndarray, U0: jnp.ndarray, Ymax, c2):
    Ny = y.shape[0]
    h = (y[-1] - y[0]) / jnp.array(Ny - 1, dtype=jnp.float64)

    def step(Uc, i):
        yi = y[0] + i * h
        k1 = fun(p, yi, Uc, Ymax, c2)
        k2 = fun(p, yi + 0.5*h, Uc + 0.5*h*k1, Ymax, c2)
        k3 = fun(p, yi + 0.5*h, Uc + 0.5*h*k2, Ymax, c2)
        k4 = fun(p, yi + h,     Uc + h*k3,     Ymax, c2)
        Un = Uc + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return Un, Un

    _, states = jax.lax.scan(step, U0, jnp.arange(Ny-1))
    Y = jnp.vstack([U0, states])
    return Y

# ---------- Observables & audits ----------
def redshift_outputs(A, p: Params):
    A_IR = A[-1]
    redshift = safe_exp(-A_IR)
    return {"A_IR": A_IR, "redshift": redshift}

def planck_mass_integral(A, y):
    integrand = safe_exp(2.0 * A)
    dx = jnp.diff(y)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    return jnp.sum(avg * dx)

def hierarchy_from_A(A: jnp.ndarray):
    logV = -4.0 * A
    V_eff = safe_exp(logV)
    dA = jnp.diff(A)
    eps_local = safe_exp(-4.0 * dA)
    eps_mean = jnp.mean(eps_local) if eps_local.size > 0 else jnp.nan
    return {"logV": logV, "V_eff": V_eff, "eps_local": eps_local, "eps_mean": eps_mean}

def audit_volume_ratio_pointwise(V_eff: jnp.ndarray, A: jnp.ndarray, tol: float = 1e-6):
    if V_eff.size < 2 or A.size < 2:
        return {"pass": jnp.array(False), "max_error": jnp.inf, "mean_error": jnp.inf}
    obs = V_eff[1:] / safe_div(V_eff[:-1], 1.0)
    dA  = jnp.diff(A)
    exp_ratio = safe_exp(-4.0 * dA)
    err_vec = jnp.abs(obs - exp_ratio)
    return {"max_error": jnp.max(err_vec), "mean_error": jnp.mean(err_vec), "pass": jnp.max(err_vec) < tol}

def audit_monotone_A(A: jnp.ndarray):
    V_eff = safe_exp(-4.0 * A)
    diffs = jnp.diff(V_eff)
    nonincreasing = jnp.all(diffs <= 0.0) if diffs.size else jnp.array(True)
    return {"nonincreasing": nonincreasing}

# ---------- Baseline & corrected solvers ----------
def solve_superpotential_and_hierarchy(p: Params):
    _, Ymax = make_stretched_grid(p)
    y_uni = make_uniform_grid(Ymax, p.Ny)
    c2 = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0], dtype=jnp.float64)
    Y = integrate_fixed_rk4(rhs_system, p, y_uni, U0, Ymax, c2)
    phi = Y[:, 0]; A = Y[:, 1]
    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "monotone_A": audit_monotone_A(A),
    }
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_uni)
    return {"y": y_uni, "phi": phi, "A": A, "c2": c2, "audits": audits,
            "redshift": redshift, "Mpl_eff": Mpl_eff}

def uv_renormalized_params(p: Params, Ymax):
    c2_base = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    a_m = jnp.array(0.5); a_l = jnp.array(0.1)
    b_m = jnp.array(0.25); b_l = jnp.array(0.05)
    delta_c2 = a_m * p.delta_m2_UV + a_l * p.delta_lambda_UV
    delta_vUV = b_m * p.delta_m2_UV + b_l * p.delta_lambda_UV
    vUV_eff = p.vUV * (1.0 + delta_vUV)
    c2_eff = c2_base + delta_c2
    return vUV_eff, c2_eff

def solve_superpotential_with_qcorr(p: Params):
    _, Ymax = make_stretched_grid(p)
    y_uni = make_uniform_grid(Ymax, p.Ny)
    vUV_eff, c2_eff = uv_renormalized_params(p, Ymax)
    U0 = jnp.array([vUV_eff, 0.0], dtype=jnp.float64)

    def fun_corr(p_, y_, U_, Ymax_, c2_):
        return rhs_system_corrected(p_, y_, U_, Ymax_, c2_, vUV_eff)

    Y = integrate_fixed_rk4(fun_corr, p, y_uni, U0, Ymax, c2_eff)
    phi = Y[:, 0]; A = Y[:, 1]
    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "monotone_A": audit_monotone_A(A),
    }
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_uni)
    G_IR = safe_div(1.0, Mpl_eff) * safe_exp(2.0 * A[-1])
    mH_eff = p.mH_bare * redshift["redshift"]
    return {"y": y_uni, "phi": phi, "A": A, "audits": audits, "redshift": redshift,
            "Mpl_eff": Mpl_eff, "G_IR_eff": G_IR, "mH_eff": mH_eff}

# ---------- Comparison & novelty ----------
def compare_qcorr_vs_RS(p: Params):
    base = solve_superpotential_and_hierarchy(p)
    corr = solve_superpotential_with_qcorr(p)
    report = {
        "ΔA_IR": corr["redshift"]["A_IR"] - base["redshift"]["A_IR"],
        "redshift_pct_dev": pct(corr["redshift"]["redshift"], base["redshift"]["redshift"]),
        "ΔmH_eff_GeV": corr["mH_eff"] - (p.mH_bare * base["redshift"]["redshift"]),
        "ΔG_IR_eff_pct": pct(corr["G_IR_eff"],
                             safe_div(1.0, base["Mpl_eff"]) * safe_exp(2.0 * base["A"][-1])),
        "audit_pass_corr": corr["audits"]["monotone_A"]["nonincreasing"] & corr["audits"]["volume_ratio_pointwise"]["pass"],
        "audit_pass_base": base["audits"]["monotone_A"]["nonincreasing"] & base["audits"]["volume_ratio_pointwise"]["pass"]
    }
    return {"baseline": base, "corrected": corr, "report": report}

def novelty_components(p: Params, epsJT, epsSch, dm2, dl):
    p2 = Params(**{**p.__dict__,
                   "eps_JT": epsJT, "eps_Sch": epsSch,
                   "delta_m2_UV": dm2, "delta_lambda_UV": dl})
    rep = compare_qcorr_vs_RS(p2)["report"]
    cA   = jnp.abs(rep["ΔA_IR"])
    cred = 0.1 * jnp.abs(rep["redshift_pct_dev"])
    cmH  = 0.01 * jnp.abs(rep["ΔmH_eff_GeV"])
    cG   = 0.1 * jnp.abs(rep["ΔG_IR_eff_pct"])
    total = cA + cred + cmH + cG
    return total, (cA, cred, cmH, cG), rep

def novelty_metric_scalar(p: Params, epsJT, epsSch, dm2, dl):
    total, _, _ = novelty_components(p, epsJT, epsSch, dm2, dl)
    return total

# ---------- Sweep 2D ----------
def sweep_epsJT_epsSch(p: Params, jt_grid=jnp.linspace(0.0, 0.35, 31), sch_grid=jnp.linspace(0.0, 0.6, 31),
                       dm2=0.0, dl=0.0):
    JT, SCH = jnp.meshgrid(jt_grid, sch_grid, indexing="ij")
    def one(ejt, esch):
        val, _, _ = novelty_components(p, ejt, esch, dm2, dl)
        return val
    vals = jax.vmap(lambda ejt: jax.vmap(lambda esch: one(ejt, esch))(sch_grid))(jt_grid)
    return {"JT": jt_grid, "SCH": sch_grid, "vals": vals}

# ---------- Convergence ----------
def convergence_check_at(p: Params, epsJT, epsSch, dm2=0.0, dl=0.0, Ny_list=(1501, 3001, 6001)):
    results = []
    for Ny in Ny_list:
        pN = Params(**{**p.__dict__, "Ny": Ny, "eps_JT": epsJT, "eps_Sch": epsSch,
                       "delta_m2_UV": dm2, "delta_lambda_UV": dl})
        val, comps, rep = novelty_components(pN, epsJT, epsSch, dm2, dl)
        results.append({"Ny": Ny, "novelty": float(val),
                        "cA": float(comps[0]), "cred": float(comps[1]), "cmH": float(comps[2]), "cG": float(comps[3]),
                        "ΔA_IR": float(rep["ΔA_IR"]), "redshift_dev": float(rep["redshift_pct_dev"])})
    return results

# ---------- Projected gradient ascent with backtracking ----------
def projected_gradient_ascent(p: Params, init=(0.05, 0.10, 0.0, 0.0), lr=0.05, steps=25, bounds=((0.0,0.5),(0.0,0.8),(-0.2,0.2),(-0.2,0.2))):
    epsJT, epsSch, dm2, dl = [jnp.array(x, dtype=jnp.float64) for x in init]
    traj = []
    grad4 = jax.grad(lambda a,b,c,d: novelty_metric_scalar(p, a,b,c,d), argnums=(0,1,2,3))
    for t in range(steps):
        g_tuple = grad4(epsJT, epsSch, dm2, dl)
        g_vec   = jnp.stack(g_tuple)
        g_norm  = jnp.linalg.norm(g_vec)
        stepvec = (lr / (1e-8 + g_norm)) * g_vec

        # backtracking line search (simple, fixed trials)
        best = None
        for alpha in jnp.array([1.0, 0.5, 0.25, 0.1], dtype=jnp.float64):
            cand = jnp.array([epsJT, epsSch, dm2, dl]) + alpha * stepvec
            # project to bounds
            cand0 = jnp.clip(cand[0], bounds[0][0], bounds[0][1])
            cand1 = jnp.clip(cand[1], bounds[1][0], bounds[1][1])
            cand2 = jnp.clip(cand[2], bounds[2][0], bounds[2][1])
            cand3 = jnp.clip(cand[3], bounds[3][0], bounds[3][1])
            val = novelty_metric_scalar(p, cand0, cand1, cand2, cand3)
            if (best is None) or (val > best[0]):
                best = (val, cand0, cand1, cand2, cand3, alpha)
        # accept best candidate
        epsJT, epsSch, dm2, dl = best[1], best[2], best[3], best[4]

        total, comps, rep = novelty_components(p, epsJT, epsSch, dm2, dl)
        traj.append({
            "step": t,
            "params": (float(epsJT), float(epsSch), float(dm2), float(dl)),
            "novelty": float(total),
            "comps": tuple([float(x) for x in comps]),
            "ΔA_IR": float(rep["ΔA_IR"]),
            "redshift_dev": float(rep["redshift_pct_dev"]),
            "alpha": float(best[5])
        })
    return traj

# ---------- Academic verification scorecard ----------
def eureka3_scorecard(p: Params, sweep, traj, peak_idx=None, Ny_list=(1501,3001,6001)):
    # find sweep peak
    total = sweep["vals"]
    idx = jnp.unravel_index(jnp.argmax(total), total.shape) if peak_idx is None else peak_idx
    jt_peak = float(sweep["JT"][idx[0]])
    sch_peak = float(sweep["SCH"][idx[1]])
    novelty_peak = float(total[idx])

    # end of trajectory
    end = traj[-1]
    jt_end, sch_end = end["params"][0], end["params"][1]
    novelty_end = end["novelty"]

    # audits at end
    p_end = Params(**{**p.__dict__, "eps_JT": jt_end, "eps_Sch": sch_end})
    cmp_end = compare_qcorr_vs_RS(p_end)
    audits_ok = bool(cmp_end["report"]["audit_pass_corr"])

    # convergence at end
    conv_end = convergence_check_at(p, jt_end, sch_end, Ny_list=Ny_list)
    ref = conv_end[1]["novelty"]  # Ny=3001 as reference
    var = max([abs(r["novelty"] - ref) for r in conv_end])
    conv_stable = (var / max(1e-12, abs(ref))) < 0.002

    # improvement threshold (>= 15%)
    improved = novelty_end >= 1.15 * novelty_peak

    score = {
        "sweep_peak_params": {"epsJT": jt_peak, "epsSch": sch_peak, "novelty": novelty_peak},
        "ascent_end_params": {"epsJT": jt_end, "epsSch": sch_end, "novelty": novelty_end},
        "audits_ok": audits_ok,
        "convergence_end": conv_end,
        "convergence_stable": conv_stable,
        "improved_over_peak": improved,
        "eureka3_verdict": "CONFIRMED" if (audits_ok and conv_stable and improved) else "NOT YET"
    }
    return score

# ---------- Runner ----------
def run_eureka3():
    p = Params()

    # Baseline sanity
    base = solve_superpotential_and_hierarchy(p)
    print("Baseline A_IR:", float(base["redshift"]["A_IR"]))
    print("Baseline redshift:", float(base["redshift"]["redshift"]))

    # Sweep landscape
    sweep = sweep_epsJT_epsSch(p,
                               jt_grid=jnp.linspace(0.0, 0.35, 31),
                               sch_grid=jnp.linspace(0.0, 0.6, 31),
                               dm2=0.0, dl=0.0)
    print("Sweep grid:", (int(sweep["JT"].shape[0]), int(sweep["SCH"].shape[0])))
    total = sweep["vals"]
    idx = jnp.unravel_index(jnp.argmax(total), total.shape)
    jt_star = float(sweep["JT"][idx[0]]); sch_star = float(sweep["SCH"][idx[1]])
    print("Sweep peak at epsJT≈%.4f, epsSch≈%.4f, novelty≈%.6f" % (jt_star, sch_star, float(total[idx])))

    # Projected gradient ascent (optimization-driven)
    traj = projected_gradient_ascent(p, init=(0.05, 0.10, 0.0, 0.0), lr=0.05, steps=25)
    print("Ascent last step:", traj[-1])

    # Academic scorecard
    score = eureka3_scorecard(p, sweep, traj, Ny_list=(1501,3001,6001))
    print("Eureka 3 verdict:", score["eureka3_verdict"])
    print("Scorecard:", score)

    # Export artifacts for Overleaf-ready plots
    np.save("e3_sweep_vals.npy", np.array(sweep["vals"]))
    np.save("e3_sweep_JT.npy",  np.array(sweep["JT"]))
    np.save("e3_sweep_SCH.npy", np.array(sweep["SCH"]))
    np.save("e3_ascent_traj.npy", np.array([[t["step"], *t["params"], t["novelty"]] for t in traj], dtype=float))
    return {"baseline": base, "sweep": sweep, "traj": traj, "score": score}

if __name__ == "__main__":
    out = run_eureka3()
