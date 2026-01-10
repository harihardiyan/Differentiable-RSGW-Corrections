
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

# ---------- Parameters ----------
@dataclass
class Params:
    k: float = 1.0
    L: float = 1.0
    rc: float = 12.0
    vUV: float = 0.1
    vIR: float = 0.01
    Ny: int = 3001
    stretch: float = 0.35
    alpha: float = 4.0
    rtol: float = 1e-12
    atol: float = 1e-14
    max_substeps: int = 800
    clip: float = 1e12
    kappa5_sq: float = 1.0

    # Quantum corrections
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

# ---------- Fixed-step RK4 (diferensiabel, lax.scan, uniform grid) ----------
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

# ---------- Analytic checks ----------
def analytic_phi(y, vUV, c2):
    return vUV * jnp.exp(2.0 * c2 * y)

def analytic_A(y, p: Params, vUV, c2):
    term = (vUV**2 / 12.0) * (jnp.exp(4.0 * c2 * y) - 1.0)
    return p.k * y + term

def error_metrics(phi_num, A_num, y, p: Params, vUV, c2):
    phi_ref = analytic_phi(y, vUV, c2)
    A_ref   = analytic_A(y, p, vUV, c2)
    phi_err = jnp.max(jnp.abs(phi_num - phi_ref))
    A_err   = jnp.max(jnp.abs(A_num - A_ref))
    return {"phi_max_error": phi_err, "A_max_error": A_err}

# ---------- Redshift dan Planck mass ----------
def redshift_outputs(A, p: Params):
    A_IR = A[-1]
    redshift = safe_exp(-A_IR)
    M_UV = jnp.array(1.0e19)  # GeV
    M_IR = M_UV * redshift
    return {"A_IR": A_IR, "redshift": redshift, "M_IR": M_IR}

def planck_mass_integral(A, y):
    integrand = safe_exp(2.0 * A)
    dx = jnp.diff(y)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    val = jnp.sum(avg * dx)
    return val

# ---------- Baseline solver (RK4 uniform grid) ----------
def solve_superpotential_and_hierarchy(p: Params):
    _, Ymax = make_stretched_grid(p)  # gunakan rc untuk Ymax
    y_uni = make_uniform_grid(Ymax, p.Ny)
    c2 = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0], dtype=jnp.float64)

    Y = integrate_fixed_rk4(rhs_system, p, y_uni, U0, Ymax, c2)
    phi = Y[:, 0]; A = Y[:, 1]

    hdict   = hierarchy_from_A(A)
    audits  = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "local_consistency": audit_local_consistency(hdict["eps_local"]),
        "monotone_A": audit_monotone_A(A),
    }
    errors   = error_metrics(phi, A, y_uni, p, p.vUV, c2)
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_uni)
    return {
        "y": y_uni, "phi": phi, "A": A,
        "c2": c2,
        "hierarchy": hdict,
        "audits": audits,
        "errors": errors,
        "redshift": redshift,
        "Mpl_eff": Mpl_eff
    }

# ---------- UV renormalized params ----------
def uv_renormalized_params(p: Params, Ymax):
    c2_base = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    a_m = jnp.array(0.5); a_l = jnp.array(0.1)
    b_m = jnp.array(0.25); b_l = jnp.array(0.05)
    delta_c2 = a_m * p.delta_m2_UV + a_l * p.delta_lambda_UV
    delta_vUV = b_m * p.delta_m2_UV + b_l * p.delta_lambda_UV
    vUV_eff = p.vUV * (1.0 + delta_vUV)
    c2_eff = c2_base + delta_c2
    return vUV_eff, c2_eff

# ---------- Quantum-corrected solver (RK4 uniform grid) ----------
def solve_superpotential_with_qcorr(p: Params):
    _, Ymax = make_stretched_grid(p)
    y_uni = make_uniform_grid(Ymax, p.Ny)

    vUV_eff, c2_eff = uv_renormalized_params(p, Ymax)
    U0 = jnp.array([vUV_eff, 0.0], dtype=jnp.float64)

    def fun_corr(p_, y_, U_, Ymax_, c2_):
        return rhs_system_corrected(p_, y_, U_, Ymax_, c2_, vUV_eff)

    Y = integrate_fixed_rk4(fun_corr, p, y_uni, U0, Ymax, c2_eff)
    phi = Y[:, 0]; A = Y[:, 1]

    hdict   = hierarchy_from_A(A)
    audits  = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "local_consistency": audit_local_consistency(hdict["eps_local"]),
        "monotone_A": audit_monotone_A(A),
    }
    errors   = error_metrics(phi, A, y_uni, p, vUV_eff, c2_eff)
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_uni)

    mH_eff = p.mH_bare * redshift["redshift"]
    G_IR = safe_div(1.0, Mpl_eff) * safe_exp(2.0 * A[-1])
    obs = {
        "mH_eff": mH_eff,
        "G_IR_eff": G_IR,
        "A_IR": redshift["A_IR"],
        "redshift": redshift["redshift"],
        "Mpl_eff": Mpl_eff,
        "vUV_eff": vUV_eff,
        "c2_eff": c2_eff
    }
    return {
        "y": y_uni, "phi": phi, "A": A,
        "params_eff": {"vUV_eff": vUV_eff, "c2_eff": c2_eff},
        "hierarchy": hdict,
        "audits": audits,
        "errors": errors,
        "redshift": redshift,
        "Mpl_eff": Mpl_eff,
        "observables": obs
    }

# ---------- Analytics & audits ----------
def hierarchy_from_A(A: jnp.ndarray):
    logV = -4.0 * A
    V_eff = safe_exp(logV)
    R_eff = jnp.power(V_eff, 0.25)
    A_eff = jnp.power(R_eff, 3.0)
    dA = jnp.diff(A)
    eps_local = safe_exp(-4.0 * dA)
    eps_mean = jnp.mean(eps_local) if eps_local.size > 0 else jnp.nan
    return {
        "logV": logV,
        "V_eff": V_eff,
        "R_eff": R_eff,
        "A_eff": A_eff,
        "eps_local": eps_local,
        "eps_mean": eps_mean
    }

def audit_volume_ratio_pointwise(V_eff: jnp.ndarray, A: jnp.ndarray, tol: float = 1e-6):
    if V_eff.size < 2 or A.size < 2:
        return {"pass": jnp.array(False), "max_error": jnp.inf, "mean_error": jnp.inf}
    obs = V_eff[1:] / safe_div(V_eff[:-1], 1.0)
    dA  = jnp.diff(A)
    exp_ratio = safe_exp(-4.0 * dA)
    err_vec = jnp.abs(obs - exp_ratio)
    max_err = jnp.max(err_vec)
    mean_err = jnp.mean(err_vec)
    return {"max_error": max_err, "mean_error": mean_err, "pass": max_err < tol}

def audit_local_consistency(eps_local: jnp.ndarray, tol_frac: float = 0.05):
    if eps_local.size == 0:
        return {"pass": jnp.array(True), "rel_std": jnp.array(0.0)}
    mu = jnp.mean(eps_local)
    sigma = jnp.std(eps_local)
    rel = safe_div(sigma, jnp.abs(mu))
    return {"pass": rel < tol_frac, "rel_std": rel}

def audit_monotone_A(A: jnp.ndarray):
    V_eff = safe_exp(-4.0 * A)
    diffs = jnp.diff(V_eff)
    nonincreasing = jnp.all(diffs <= 0.0) if diffs.size else jnp.array(True)
    return {"nonincreasing": nonincreasing}

# ---------- Novelty comparison metrics ----------
def compare_qcorr_vs_RS(p: Params):
    base = solve_superpotential_and_hierarchy(p)
    corr = solve_superpotential_with_qcorr(p)

    def pct(a, b):
        denom = jnp.where(jnp.abs(b) > 1e-30, b, jnp.array(1.0))
        return 100.0 * (a - b) / denom

    Omega_IR_corr = safe_exp(-4.0 * corr["A"][-1])
    Omega_IR_base = safe_exp(-4.0 * base["A"][-1])

    report = {
        "ΔA_IR": corr["redshift"]["A_IR"] - base["redshift"]["A_IR"],
        "redshift_pct_dev": pct(corr["redshift"]["redshift"], base["redshift"]["redshift"]),
        "ΔmH_eff_GeV": corr["observables"]["mH_eff"] - (p.mH_bare * base["redshift"]["redshift"]),
        "ΔG_IR_eff_pct": pct(corr["observables"]["G_IR_eff"],
                             safe_div(1.0, base["Mpl_eff"]) * safe_exp(2.0 * base["A"][-1])),
        "Ω_IR_corr": Omega_IR_corr,
        "Ω_IR_base": Omega_IR_base,
        "audit_pass_corr": corr["audits"]["monotone_A"]["nonincreasing"] & corr["audits"]["volume_ratio_pointwise"]["pass"],
        "audit_pass_base": base["audits"]["monotone_A"]["nonincreasing"] & base["audits"]["volume_ratio_pointwise"]["pass"]
    }
    return {"baseline": base, "corrected": corr, "report": report}

# ---------- Batched parameter sweep ----------
def metrics_vector(p: Params):
    comp = compare_qcorr_vs_RS(p)["report"]
    return jnp.array([
        comp["ΔA_IR"],
        comp["redshift_pct_dev"],
        comp["ΔmH_eff_GeV"],
        comp["ΔG_IR_eff_pct"]
    ], dtype=jnp.float64)

def batch_compare(p: Params, epsJT_arr, epsSch_arr, dm2_arr, dl_arr):
    def one(theta):
        epsJT, epsSch, dm2, dl = theta
        p2 = Params(**{**p.__dict__,
                       "eps_JT": epsJT,
                       "eps_Sch": epsSch,
                       "delta_m2_UV": dm2,
                       "delta_lambda_UV": dl})
        return metrics_vector(p2)
    thetas = jnp.stack([epsJT_arr, epsSch_arr, dm2_arr, dl_arr], axis=-1)
    return jax.vmap(one)(thetas)

# ---------- Novelty metric dan gradients (AD menembus solver RK4) ----------
def novelty_metric_scalar(p: Params, epsJT, epsSch, dm2, dl):
    p2 = Params(**{**p.__dict__,
                   "eps_JT": epsJT,
                   "eps_Sch": epsSch,
                   "delta_m2_UV": dm2,
                   "delta_lambda_UV": dl})
    comp = compare_qcorr_vs_RS(p2)["report"]
    return (jnp.abs(comp["ΔA_IR"]) +
            0.1 * jnp.abs(comp["redshift_pct_dev"]) +
            0.01 * jnp.abs(comp["ΔmH_eff_GeV"]) +
            0.1 * jnp.abs(comp["ΔG_IR_eff_pct"]))

grad_novelty = jax.grad(lambda epsJT, epsSch, dm2, dl, p: novelty_metric_scalar(p, epsJT, epsSch, dm2, dl),
                        argnums=(0,1,2,3))

# ---------- Dekomposisi novelty & sweep ----------
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

def sweep_epsJT_epsSch(p: Params, jt_grid=jnp.linspace(0.0, 0.35, 31), sch_grid=jnp.linspace(0.0, 0.6, 31),
                       dm2=0.0, dl=0.0):
    JT, SCH = jnp.meshgrid(jt_grid, sch_grid, indexing="ij")
    def one(ejt, esch):
        total, comps, rep = novelty_components(p, ejt, esch, dm2, dl)
        return jnp.array([total, *comps], dtype=jnp.float64)
    vals = jax.vmap(lambda ejt: jax.vmap(lambda esch: one(ejt, esch))(SCH[0]))(JT[:,0])
    return {"JT": jt_grid, "SCH": sch_grid, "vals": vals}

# ---------- Uji konvergensi ----------
def convergence_check(p: Params, Ny_list=(1501, 3001, 6001), epsJT=0.1, epsSch=0.2, dm2=0.02, dl=0.01):
    out = []
    for Ny in Ny_list:
        pN = Params(**{**p.__dict__, "Ny": Ny})
        total, comps, rep = novelty_components(pN, epsJT, epsSch, dm2, dl)
        out.append({"Ny": Ny, "novelty": float(total),
                    "cA": float(comps[0]), "cred": float(comps[1]), "cmH": float(comps[2]), "cG": float(comps[3]),
                    "ΔA_IR": float(rep["ΔA_IR"]), "redshift_dev": float(rep["redshift_pct_dev"])})
    return out

# ---------- Gradient ascent (patched indexing) ----------
def gradient_ascent_novelty(p: Params, init=(0.05, 0.1, 0.0, 0.0), lr=0.05, steps=25, clip_bounds=True):
    epsJT, epsSch, dm2, dl = [jnp.array(x, dtype=jnp.float64) for x in init]
    traj = []
    grad4 = jax.grad(lambda a,b,c,d: novelty_metric_scalar(p, a,b,c,d), argnums=(0,1,2,3))

    for t in range(steps):
        g_tuple = grad4(epsJT, epsSch, dm2, dl)
        g_vec   = jnp.stack(g_tuple)
        g_norm  = jnp.linalg.norm(g_vec)
        stepvec = (lr / (1e-8 + g_norm)) * g_vec

        epsJT = epsJT + stepvec[0]
        epsSch = epsSch + stepvec[1]
        dm2   = dm2   + stepvec[2]
        dl    = dl    + stepvec[3]

        if clip_bounds:
            epsJT = jnp.clip(epsJT, 0.0, 0.5)
            epsSch = jnp.clip(epsSch, 0.0, 0.8)
            dm2 = jnp.clip(dm2, -0.2, 0.2)
            dl  = jnp.clip(dl,  -0.2, 0.2)

        total, comps, rep = novelty_components(p, epsJT, epsSch, dm2, dl)
        traj.append({
            "step": t,
            "params": (float(epsJT), float(epsSch), float(dm2), float(dl)),
            "novelty": float(total),
            "comps": tuple([float(x) for x in comps]),
            "ΔA_IR": float(rep["ΔA_IR"]),
            "redshift_dev": float(rep["redshift_pct_dev"])
        })
    return traj

# ---------- Runner utama ----------
def run_novelty_demo():
    p = Params()
    base = solve_superpotential_and_hierarchy(p)
    print("Baseline A_IR:", float(base["redshift"]["A_IR"]))
    print("Baseline redshift:", float(base["redshift"]["redshift"]))

    p_corr = Params(**{**p.__dict__, "eps_JT": 0.1, "eps_Sch": 0.2, "delta_m2_UV": 0.05, "delta_lambda_UV": 0.02})
    cmp = compare_qcorr_vs_RS(p_corr)
    rep = cmp["report"]
    print("ΔA_IR:", float(rep["ΔA_IR"]))
    print("redshift % dev:", float(rep["redshift_pct_dev"]))
    print("ΔmH_eff (GeV):", float(rep["ΔmH_eff_GeV"]))
    print("ΔG_IR_eff (%):", float(rep["ΔG_IR_eff_pct"]))

    epsJT_arr  = jnp.linspace(0.0, 0.2, 6)
    epsSch_arr = jnp.linspace(0.0, 0.3, 6)
    dm2_arr    = jnp.linspace(0.0, 0.06, 6)
    dl_arr     = jnp.linspace(0.0, 0.03, 6)
    metrics = batch_compare(p, epsJT_arr, epsSch_arr, dm2_arr, dl_arr)
    print("Batched metrics shape:", tuple(metrics.shape))

    g = grad_novelty(0.1, 0.2, 0.05, 0.02, p)
    print("Gradients (dNovelty/d[epsJT, epsSch, dm2, dl]):", tuple([float(x) for x in g]))
    return {"baseline": base, "comparison": cmp, "batched": metrics, "grad": g}

# ---------- Runner discovery ----------
def run_discovery():
    p = Params()
    # 1) Sweep 2D
    sw = sweep_epsJT_epsSch(p,
                            jt_grid=jnp.linspace(0.0, 0.35, 31),
                            sch_grid=jnp.linspace(0.0, 0.6, 31),
                            dm2=0.0, dl=0.0)
    print("Sweep grid:", (int(sw["JT"].shape[0]), int(sw["SCH"].shape[0])))
    total = sw["vals"][:,:,0]
    idx = jnp.unravel_index(jnp.argmax(total), total.shape)
    jt_star = float(sw["JT"][idx[0]]); sch_star = float(sw["SCH"][idx[1]])
    print("Max novelty (grid) at epsJT≈%.4f, epsSch≈%.4f, value=%.6f" % (jt_star, sch_star, float(total[idx])))

    # 2) Convergence Ny
    conv = convergence_check(p, Ny_list=(1501, 3001, 6001), epsJT=jt_star, epsSch=sch_star, dm2=0.0, dl=0.0)
    print("Convergence samples:", conv)

    # 3) Gradient ascent
    traj = gradient_ascent_novelty(p, init=(0.05, 0.1, 0.0, 0.0), lr=0.05, steps=25, clip_bounds=True)
    print("Ascent last step:", traj[-1])

    # Export lightweight arrays for plotting (NumPy)
    np.save("sweep_vals.npy", np.array(sw["vals"]))
    np.save("sweep_JT.npy",  np.array(sw["JT"]))
    np.save("sweep_SCH.npy", np.array(sw["SCH"]))
    np.save("ascent_traj.npy", np.array([[t["step"], *t["params"], t["novelty"]] for t in traj], dtype=float))
    return {"sweep": sw, "convergence": conv, "ascent": traj}

if __name__ == "__main__":
    out = run_novelty_demo()
    disc = run_discovery()
