# Optical conductivity of the extended Hubbard chain: DMRG + TDVP

`HDMRG_TDVP_production.jl` and `HDMRG_TDVP_Analysis.jl`. Units $\hbar = e = a = |t| = 1$.

## Workflow

1. `HDMRG_TDVP_production.jl` computes the DMRG ground state $|0\rangle$, applies the current operator, evolves $J|0\rangle$ with TDVP and saves the current–current correlator to `data/tdvp_<tag>.jld2`.
2. `HDMRG_TDVP_Analysis.jl` reads every `data/tdvp_*.jld2`, Fourier transforms the correlator with a window of width $\eta$ into $\sigma(\omega)$, runs the [checks](#checks) and writes figures and a summary table to `analysis/`.

The production script stores only correlators. Every choice about broadening is made in the analysis, so one production run can be analysed at any $\eta$.

## Model and operators

Open chain of $L$ sites,

```math
H = -\sum_{l,\sigma}\bigl(c^\dagger_{l\sigma}c_{l+1,\sigma} + \text{h.c.}\bigr)
    + U\sum_l \bigl(n_{l\uparrow}-\tfrac12\bigr)\bigl(n_{l\downarrow}-\tfrac12\bigr)
    + V\sum_l \bigl(n_l-1\bigr)\bigl(n_{l+1}-1\bigr).
```

The particle–hole symmetric form shifts the energy relative to the usual $`U n_\uparrow n_\downarrow`$ convention; at half filling the $U$ term is shifted by $-U/4$ per site.

Polarization and current:

```math
P = \sum_l \Bigl(l-\tfrac{L+1}{2}\Bigr) n_l, \qquad
J = i[H,P], \qquad
\tilde J \equiv -iJ = t\sum_{l,\sigma}\bigl(c^\dagger_{l\sigma}c_{l+1,\sigma} - \text{h.c.}\bigr),\quad t=-1.
```

$\tilde J$ is a real antisymmetric matrix. Only the kinetic term $T$ contributes to $[H,P]$, and $[P,\tilde J] = -T$. The origin of $P$ drops out of every correlator.

## Kubo formula

Following Takayoshi and Giamarchi [1], with $|0\rangle$ the ground state of energy $E_0$:

```math
\sigma(\omega) = \frac{1}{N\omega}\,\mathrm{Re}\int_0^\infty d\tau\, e^{i\omega\tau} C_{JJ}(\tau),
\qquad C_{JJ}(\tau) = \langle 0| J\, e^{-i(H-E_0)\tau} J |0\rangle,
```

```math
\sigma(\omega) = \frac{1}{N}\,\mathrm{Re}\int_0^\infty d\tau\, e^{i\omega\tau} C_{PJ}(\tau),
\qquad C_{PJ}(\tau) = 2\,\mathrm{Im}\,\langle 0| P\, e^{-i(H-E_0)\tau} J |0\rangle.
```

Both give the same $\sigma(\omega)$. The $J$–$J$ form is the default. The $P$–$J$ form has no $1/\omega$ factor, so it behaves better at small $\omega$, but it costs one extra MPO contraction per step and gives the correlator only on $`[0, t_{\max}]`$ (no doubling). It is switched on with `PJ = 1`.

**Sum rule.** Integrating over frequency,

```math
\int_0^\infty \sigma(\omega)\, d\omega = -\frac{\pi \langle T\rangle}{2N},
```

with $\langle T\rangle$ the kinetic energy of $|0\rangle$, measured in the production run.

## Time evolution

**Initial state.** $`|\tilde\varphi(0)\rangle = \tilde J|0\rangle/\sqrt{C_0}`$, with $`C_0 = C_{JJ}(0) = \langle J^2\rangle`$.

**Time doubling.** $H$, $\tilde J$ and $|0\rangle$ are real, so $`|\tilde\varphi(0)\rangle`$ is a real vector and $`e^{-i(H-E_0)\tau}`$ is a symmetric matrix. Splitting $\tau = 2t$,

```math
C_{JJ}(2t) = C_0 \sum_s \tilde\varphi_s(t)\,\tilde\varphi_s(t),
\qquad |\tilde\varphi(t)\rangle = e^{-i(H-E_0)t}|\tilde\varphi(0)\rangle,
```

a bilinear (not sesquilinear) contraction, computed as `inner(conj_mps(phi), phi)`. Evolving to $`t_{\max}`$ gives $`C_{JJ}`$ on $`[0, 2t_{\max}]`$: the same spectral resolution for half the evolution time, and the evolution time is what makes the bond dimension, and the cost, grow.

**Integrator.** TDVP [2, 3] with $H - E_0$ as the MPO, which removes the global phase.

- Two-site TDVP while the bond dimension is below the cap $\chi$, so the MPS can grow; truncation set by `cutoff`.
- One-site TDVP once $\chi$ is reached: fixed bond dimension, no truncation, cheaper by roughly a factor $d = 4$ per bond.
- Local exponentials by Lanczos (`ishermitian=true`), stopping as soon as `tol = 1e-8` is met (`eager=true`), at most 15 Krylov vectors.
- `normalize=false`: the norm is not forced, so $`\|\tilde\varphi(t)\|`$ is a check.

**Ground state.** DMRG in the sector $`(N, S_z)`$, started from a random MPS (link dimension 10) built on a Néel state at half filling, or on evenly spread electrons otherwise. Sweep schedule (`get_dmrg_results` in `src/analysis/hubbard_dmrg.jl`), at most 30 sweeps:

- sweeps 1–5: bond dimension 32, 64, 128, 256, 512, capped at $`\chi_{\rm gs}`$; noise $10^{-4}$;
- next sweeps: noise $`10^{-5}\cdot 2^{-n}`$, down to $10^{-8}$;
- last 15 sweeps: no noise;
- `cutoff = 1e-12`; stop when the energy changes by less than $10^{-7}$ between sweeps.

## Production script

```bash
julia -t 4 HDMRG_TDVP_production.jl L U V [N_total Sz dt tmax chi cutoff chi_gs PJ]
julia -t 4 HDMRG_TDVP_production.jl 64 4.0 0.0 64 0 0.05 20 400 1e-9 400
```

| argument | meaning | default |
|---|---|---|
| `L U V` | sites, on-site and nearest-neighbour interaction | required |
| `N_total` | electrons | `L` |
| `Sz` | $`(N_\uparrow - N_\downarrow)/2`$ | 0 |
| `dt` | TDVP step | 0.05 |
| `tmax` | evolution time (correlator reaches $`2t_{\max}`$) | 20 |
| `chi` | TDVP bond-dimension cap | 400 |
| `cutoff` | two-site TDVP truncation | 1e-9 |
| `chi_gs` | DMRG bond-dimension cap | `chi` |
| `PJ` | 1: also measure $`C_{PJ}`$ | 0 |

Threads go to block-sparse contractions (BLAS on one thread). The ground state is cached in `states/gs_<tag>.jld2` and the evolution is checkpointed every 100 steps; a killed run resumes from the last checkpoint, and both files are deleted at the end. A run whose data file already exists is skipped.

**Output:** `data/tdvp_<tag>.jld2`, with `tag = L<L>_U<U>_V<V>_N<N>_Sz<Sz>_dt<dt>_T<tmax>_chi<chi>`, and the log `logs/<tag>.log`.

| key | content |
|---|---|
| `L U V N_total Sz_tot dt tmax χ cutoff χ_gs PJ` | run parameters |
| `E_dmrg`, `Ekin`, `C0` | $E_0$, $\langle T\rangle$, $\langle J^2\rangle$ |
| `times`, `times_JJ` | $t_k$ and $2t_k$ |
| `C_JJ` | $`C_{JJ}(2t_k)`$ |
| `C_PJ` | $`C_{PJ}(t_k)`$ (zeros unless `PJ = 1`) |
| `chi_t` | bond dimension of $`\tilde\varphi(t_k)`$ |
| `norm_end`, `t_tdvp` | $`\Vert\tilde\varphi(t_{\max})\Vert`$, TDVP wall time (s) |

## Analysis script

```bash
julia HDMRG_TDVP_Analysis.jl [datadir eta omegaplot window]
julia HDMRG_TDVP_Analysis.jl data auto 10
```

`datadir` holds the `tdvp_*.jld2` files (default `data`). `omegaplot` is only the upper $\omega$ of the plots; the spectrum is always computed up to $\omega = 10 + U + 4|V|$. `window` is `gauss` (default) or `lorentz`.

**Window.** The correlator is known only up to $`\tau_{\max} = 2t_{\max}`$, so before the transform (trapezoid rule on the saved grid) it is multiplied by

```math
W(\tau) = e^{-\eta^2\tau^2/2}\ \ (\texttt{gauss}) \qquad\text{or}\qquad W(\tau) = e^{-\eta\tau}\ \ (\texttt{lorentz}).
```

so that

```math
\sigma_{JJ}(\omega) = \frac{1}{N\omega}\,\mathrm{Re}\int_0^{2t_{\max}} d\tau\, e^{i\omega\tau}\, W(\tau)\, C_{JJ}(\tau),
\qquad
\sigma_{PJ}(\omega) = \frac{1}{N}\,\mathrm{Re}\int_0^{t_{\max}} d\tau\, e^{i\omega\tau}\, W(\tau)\, C_{PJ}(\tau),
```

the second only if $`C_{PJ}`$ was measured. $\sigma(\omega)$ then comes out convolved with a Gaussian of standard deviation $\eta$ (or a Lorentzian of half-width $\eta$): $\eta$ is the frequency resolution.

**Choice of $\eta$.** `eta` is a number, used as given, or `auto` (default):

```math
\eta = \max\Bigl(\eta_{\rm close},\ \frac{2\pi v}{L}\Bigr), \qquad
\eta_{\rm close} = \frac{\sqrt{2\ln 10^3}}{\tau_{\max}} \approx \frac{3.72}{\tau_{\max}}
\ \ \Bigl(\texttt{lorentz}:\ \frac{\ln 10^3}{\tau_{\max}}\Bigr), \qquad v = 2.
```

1. $`\eta_{\rm close}`$: the window has fallen to $10^{-3}$ at $`\tau_{\max}`$. A smaller $\eta$ cuts the correlator off abruptly and $\sigma(\omega)$ rings.
2. $2\pi v/L$: the level spacing of the finite chain, with $v = 2$ the largest charge velocity. A smaller $\eta$ resolves individual finite-size levels instead of the continuum.

Both depend only on $`t_{\max}`$ and $L$, never on $\chi$. Runs of one system that differ only in $\chi$ get the same $\eta$, so convergence in $\chi$ is judged at a fixed resolution; $\eta$ is never raised to make a low-$\chi$ run agree with a higher one.

**Output** in `analysis/`, in the style of `figstyle.jl`:

- `tdvp_<tag>.{pdf,svg,png}`: (a) $`C_{JJ}(\tau)/C_{JJ}(0)`$, real and imaginary parts; (b) $\sigma(\omega)$ with its peaks marked.
- `sigma_<U|chi|Uchi>_L.._V.._N.._Sz..`: $\sigma(\omega)$ of all runs sharing $`(L, V, N, S_z)`$, labelled by the parameter that varies.
- `tdvp_summary.csv`: parameters, $\eta$, $E_0$, $\langle T\rangle$, $C_0$, sum rules and target, final $\chi$, the correlator time at which $\chi$ was reached, peak positions (local maxima above 2% of the maximum).

## Checks

**Inside each production run** (log lines marked `[Verification]`):

- the ground state has the requested quantum numbers (flux);
- $`C_{PJ}(0) = -\langle T\rangle`$ when `PJ = 1`, which tests $P$ and $\tilde J$;
- $`\vert C_{JJ}^{\rm doubled}(t_{\max}) - C_{JJ}^{\rm direct}(t_{\max})\vert / C_0`$, the direct one computed as $`\langle\tilde\varphi(0)|\tilde\varphi(t_{\max})\rangle`$. Without truncation the two agree exactly, so the difference measures the truncation error of the evolution;
- $`\Vert\tilde\varphi(t_{\max})\Vert = 1`$ up to truncation.

**In the analysis:**

- *Sum rule* against the run's own $\langle T\rangle$.
- *Ground-state energies.* Bethe ansatz, $U=4$, infinite chain: $e_0 = -0.57373$ per site in the $`U n_\uparrow n_\downarrow`$ convention, i.e. $-1.57373$ here, and $\langle T\rangle/L = -0.9747$. Open chains differ by boundary corrections of order $1/L$.
- *Bond dimension.* Compare the $\chi$ runs of one system in the `sigma_chi_*` overlay at the same (`auto`) $\eta$. Signs of truncation: $`\vert C_{JJ}\vert/C_0`$ growing at late $\tau$, a large doubled-vs-direct difference, and features that move or split with $\chi$.

## References

1. S. Takayoshi and T. Giamarchi, Eur. Phys. J. D **76**, 213 (2022).
2. J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, and F. Verstraete, Phys. Rev. B **94**, 165116 (2016).
3. S. Paeckel *et al.*, Ann. Phys. **411**, 167998 (2019).
