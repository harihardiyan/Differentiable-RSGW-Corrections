

# Differentiable-RSGW-Corrections

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-blue.svg)](https://github.com/google/jax)
[![Physics: RS--GW](https://img.shields.io/badge/Physics-Randall--Sundrum--GW-red.svg)](#)

A high-performance, fully differentiable physics solver implemented in **JAX** for exploring localized quantum corrections in the **Randall–Sundrum–Goldberger–Wise (RS-GW)** warped extra-dimensional model.

## Overview

This repository provides a framework for analyzing the stability and hierarchy of warped geometries through the lens of **Differentiable Programming**. By leveraging automatic differentiation (AD), the solver allows for gradient-based optimization and sensitivity analysis of infrared (IR) and ultraviolet (UV) quantum corrections, bypassing the limitations of traditional non-differentiable numerical integrators.

### Abstract
> We present a fully differentiable fixed-step Runge–Kutta 4 (RK4) integrator applied to the RS-GW model. Our work incorporates localized quantum corrections in both the IR and UV regimes. By propagating gradients seamlessly through the physics solver, we enable direct sensitivity analysis of radion stabilization and effective hierarchy formation. Our findings demonstrate that **IR-localized quantum corrections** (parametrized by $\epsilon_{JT}$ and $\epsilon_{Sch}$) overwhelmingly dominate the deviations in the effective warp factor and Higgs mass shift, while UV counterterms yield only marginal contributions.

---

## Key Features

- **Differentiable Physics Solver**: Built on JAX's `lax.scan`, enabling end-to-end differentiability through the RK4 integration steps.
- **Quantum Correction Modules**: Includes modules for Jackiw-Teitelboim (JT) and Schwarzian-like corrections near the IR brane.
- **Gradient-Based Optimization**: Implements gradient ascent to maximize a composite "Novelty Metric," identifying parameter regions with the highest physical impact.
- **Numerical Rigor**: Features safe-clipping helpers, high-precision (`float64`) support, and comprehensive audit logs for monotonicity and consistency checks.

---

## Mathematical Framework

### 1. Baseline Evolution
The system evolves the scalar field $\phi$ (Goldberger-Wise) and the warp factor $A$ across the extra dimension $y$:
$$ \frac{d\phi}{dy} = 2 c_2 \phi $$
$$ \frac{dA}{dy} = \frac{\kappa_5^2}{3} \left( W_0 + c_2 \phi^2 \right) $$

### 2. Quantum Corrections
We introduce localized deviations in the IR regime ($y \to Y_{max}$):
- **JT-like Correction**: $\Delta (A')_{JT} = \epsilon_{JT} \, w_{IR}(y)$
- **Schwarzian Correction**: $\Delta (A')_{Sch} = \epsilon_{Sch} \, w_{IR}(y) \frac{(A'_{base})^2}{1 + (A'_{base})^2 / \sigma^2}$

### 3. The Novelty Metric
To identify the most "physically interesting" parameter space, we optimize a composite metric $N$:
$$ N = | \Delta A_{IR} | + 0.1 \left| \frac{\Delta \Omega_{IR}}{\Omega_{IR}} \right| + 0.01 | \Delta m_H^{\text{eff}} | + 0.1 | \Delta G_{IR}^{\text{eff}} | $$

---

## Quick Start

### Prerequisites
- Python 3.9+
- JAX & JAXlib
- NumPy

### Installation
```bash
git clone https://github.com/harihardiyan/Differentiable-RSGW-Corrections.git
cd Differentiable-RSGW-Corrections
pip install -r requirements.txt
```

### Usage
Run the main discovery script to perform a 2D parameter sweep and initiate gradient-based optimization:
```bash
Differentiable-RSGW-Corrections.py
```

---

## Results & Discovery

Our analysis reveals that the sensitivity of the hierarchy is highly anisotropic in the parameter space. The gradients $\partial N/\partial \epsilon_{JT}$ are consistently larger than those of the UV counterterms, suggesting that the hierarchy problem's resolution is significantly more sensitive to the IR dynamics of the extra dimension.

- **Baseline Redshift**: $\sim 10^{-17}$
- **Peak Novelty Observed**: $N \approx 17.6$ (after gradient ascent optimization).
- **Dominance**: IR corrections contribute $>95\%$ to the total variance in the effective Higgs mass shift.

---

## Contributing
This project is an academic exploration into the intersection of high-energy physics and differentiable programming. Contributions, discussions, and feedback are warmly welcomed. Please open an issue or submit a pull request for any improvements or bug fixes.

## Author
**Hari Hardiyan**
- *Research focus*: Warped Extra Dimensions, Differentiable Physics, and Quantum Gravity.

## Citation
If you find this work or code useful for your research, please consider citing it:

```bibtex
@software{hardiyan2026diffRSGW,
  author = {Hari Hardiyan},
  title = {Differentiable Quantum Corrections in Randall-Sundrum-Goldberger-Wise Models},
  year = {2026},
  url = {https://github.com/harihardiyan/Differentiable-RSGW-Corrections},
  note = {Physics solver using JAX for AD-based sensitivity analysis}
}
```

