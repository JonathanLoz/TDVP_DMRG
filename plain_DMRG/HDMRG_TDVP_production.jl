using Logging
using MKL
using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2

include("../src/backend/mps.jl")
include("../src/hamiltonians/generalized_hubbard.jl")
include("../src/analysis/hubbard_dmrg.jl")

# ─── Command line ────────────────────────────────────────────────────────────

const L       = parse(Int, ARGS[1])
const U       = parse(Float64, ARGS[2])
const V       = parse(Float64, ARGS[3])
const N_total = length(ARGS) >= 4  ? parse(Int,     ARGS[4])  : L
const Sz_tot  = length(ARGS) >= 5  ? parse(Int,     ARGS[5])  : 0
const dt      = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : 0.05
const tmax    = length(ARGS) >= 7  ? parse(Float64, ARGS[7])  : 20.0
const χ       = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 400
const cutoff  = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 1e-9
const χ_gs    = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : χ
const PJ      = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) == 1 : false

if Threads.nthreads() > 1
    BLAS.set_num_threads(1)
    NDTensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse(true)
else
    BLAS.set_num_threads(1)
    NDTensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse(false)
end

@info "Running: L=$L  U=$U  V=$V  N=$N_total  Sz=$Sz_tot  dt=$dt  tmax=$tmax  χ=$χ  cutoff=$cutoff  χ_gs=$χ_gs  PJ=$PJ"

const CKPT_EVERY = 100    # TDVP steps between checkpoints
const LOG_EVERY  = 20     # TDVP steps between log lines

# Krylov exponentials of tdvp: Lanczos (H − E0 is Hermitian) with early exit, instead
# of KrylovKit's default Arnoldi with a fixed 30-vector build per bond. tol is per unit
# time; keep the default verbosity so a failed convergence is reported.
const UPDATER_KWARGS = (; ishermitian=true, eager=true, tol=1e-8, krylovdim=15)

# ─── Types ────────────────────────────────────────────────────────────────

struct ModelParams
    Nx::Int
    t::Float64
    U::Float64
    V::Float64
    μ::Float64
end

struct DMRGConfig
    χ::Int
    energy_tol::Float64
    maxsweeps::Int
end
DMRGConfig(χ; energy_tol=1e-7, maxsweeps=30) = DMRGConfig(χ, energy_tol, maxsweeps)

# ─── Operators ────────────────────────────────────────────────────────────────

function hopping_opsum(p::ModelParams, L)
    Nx, Ny = p.Nx, L
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    ampo = OpSum()
    for b in lattice
        ampo += p.t, "Cdagup", b.s1, "Cup", b.s2
        ampo += p.t, "Cdagup", b.s2, "Cup", b.s1
        ampo += p.t, "Cdagdn", b.s1, "Cdn", b.s2
        ampo += p.t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    return ampo
end

function hubbard_opsum(p::ModelParams, L)
    Nx, Ny = p.Nx, L
    N = Nx * Ny
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    ampo = hopping_opsum(p, L)
    # U(n↑ − ½)(n↓ − ½) = U n↑n↓ − (U/2) n + U/4
    for i in 1:N
        ampo += p.U, "Nupdn", i
        ampo += -p.U / 2, "Ntot", i
        ampo += p.U / 4, "Id", i
    end
    # V(nᵢ − 1)(nⱼ − 1) = V nᵢnⱼ − V nᵢ − V nⱼ + V
    for b in lattice
        ampo += p.V, "Ntot", b.s1, "Ntot", b.s2
        ampo += -p.V, "Ntot", b.s1
        ampo += -p.V, "Ntot", b.s2
        ampo += p.V, "Id", b.s1
    end
    for i in 1:N
        ampo += -p.μ, "Ntot", i
    end
    return ampo
end

# J̃ = −iJ = t Σ_σ (c†_{i,σ} c_{i+1,σ} − h.c.): real, antisymmetric; t = p.t.
function current_opsum(p::ModelParams, L)
    lattice = square_lattice(p.Nx, L; yperiodic=false)
    ampo = OpSum()
    for b in lattice
        ampo +=  p.t, "Cdagup", b.s1, "Cup", b.s2
        ampo += -p.t, "Cdagup", b.s2, "Cup", b.s1
        ampo +=  p.t, "Cdagdn", b.s1, "Cdn", b.s2
        ampo += -p.t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    return ampo
end

# P = Σ_l (l − (L+1)/2) n_l; the origin drops out of the correlators.
function polarization_opsum(L)
    ampo = OpSum()
    for l in 1:L
        ampo += (l - (L + 1) / 2), "Ntot", l
    end
    return ampo
end

function build_H(ampo, sites; splitblocks=true)
    return MPO(ampo, sites; splitblocks=splitblocks)
end

function setup_model(p::ModelParams, L)
    N = p.Nx * L
    sites = siteinds("Electron", N; conserve_qns=true)
    ampo = hubbard_opsum(p, L)
    H = build_H(ampo, sites)
    return H, sites, ampo
end

# ─── MPS helpers ──────────────────────────────────────────────────────────────

function extract_sites(psi::MPS)
    return [siteind(psi, i) for i in 1:length(psi)]
end

conj_mps(psi::MPS) = MPS([conj(psi[j]) for j in 1:length(psi)])

# ─── Ground state ─────────────────────────────────────────────────────────────

function init_state(L, N_total, Sz)
    N_up = N_total ÷ 2 + Sz
    N_down = N_total - N_up
    if N_total == L && Sz == 0
        return [isodd(i) ? "Up" : "Dn" for i in 1:L]
    end
    state = fill("Emp", L)
    for i in round.(Int, range(1, L, length=N_up + 2)[2:end-1])
        state[i] = "Up"
    end
    for i in round.(Int, range(1.5, L + 0.5, length=N_down + 2)[2:end-1])
        state[i] = state[i] == "Up" ? "UpDn" : "Dn"
    end
    return state
end

function run_dmrg_sz(H, sites, L, N_total, sz, cfg::DMRGConfig; psi_init=nothing)
    if isnothing(psi_init)
        state = init_state(L, N_total, sz)
        psi_0 = random_mps(sites, state; linkdims=10)
    else
        psi_0 = psi_init
    end
    obs = DMRGObserver(; energy_tol=cfg.energy_tol)
    E, psi = get_dmrg_results(psi_0, H; maxsweeps=cfg.maxsweeps, maxdim=cfg.χ, observer=obs)

    err = isempty(obs.truncerrs) ? 0.0 : maximum(obs.truncerrs)
    return E, psi, err
end

# ─── Run one (L, U, V) point ──────────────────────────────────────────────────

function run_point()
    p = ModelParams(1, -1.0, U, V, 0.0)
    tag = "L$(L)_U$(U)_V$(V)_N$(N_total)_Sz$(Sz_tot)_dt$(dt)_T$(tmax)_chi$(χ)"

    datadir = "data"
    statedir = "states"
    logdir = "logs"
    mkpath(datadir)
    mkpath(statedir)
    mkpath(logdir)

    data_file = joinpath(datadir,  "tdvp_$(tag).jld2")
    gs_file   = joinpath(statedir, "gs_$(tag).jld2")
    ckpt_file = joinpath(statedir, "ckpt_$(tag).jld2")

    if isfile(data_file)
        @info "Data already exists for $tag. Nothing to do."
        return
    end

    H, sites, ampo = setup_model(p, L)

    io = open(joinpath(logdir, "$(tag).log"), "a")
    global_logger(ConsoleLogger(io, Logging.Info))

    times = collect(0.0:dt:tmax)
    nt    = length(times)
    N     = L

    # ── Ground state ─────────────────────────────────────────────────────────
    if isfile(gs_file)
        loaded = load(gs_file)
        psi_gs = loaded["psi_gs"]
        E_dmrg = loaded["E_dmrg"]
        sites  = extract_sites(psi_gs)
        H = build_H(ampo, sites)
        @info "── Ground state loaded from $gs_file (E = $E_dmrg). H rebuilt."
    else
        cfg = DMRGConfig(χ_gs)
        @info "── Ground state (Sz=$Sz_tot, N=$N_total) at χ_gs = $χ_gs ..."
        time_taken = @elapsed begin
            E_dmrg, psi_gs, err_dmrg = run_dmrg_sz(H, sites, L, N_total, Sz_tot, cfg)
        end
        @info "  E = $E_dmrg   maxlinkdim = $(maxlinkdim(psi_gs))   err = $err_dmrg"
        @info "  [Timer] GS completed in $(round(time_taken / 3600, digits=3)) hours."
        @info "  [Memory] after GS: $(round(Sys.maxrss(), digits=2)) "
        @save gs_file psi_gs E_dmrg
    end

    @info "  [Verification] flux(psi_gs) = $(flux(psi_gs))   (target N = $N_total, Sz = $Sz_tot)"

    # ── Operators for the correlators ────────────────────────────────────────
    Jt = build_H(current_opsum(p, L), sites)
    T  = build_H(hopping_opsum(p, L), sites)
    P  = PJ ? build_H(polarization_opsum(L), sites) : nothing
    Ekin = real(inner(psi_gs', T, psi_gs))
    @info "  ⟨T⟩ = $Ekin   (f-sum rule: ∫_0^∞ σ dω = −π⟨T⟩/2N = $(-π * Ekin / (2N)), checked in the analysis)"

    # Evolve with H − E0 (removes the global phase; Krylov sees the bandwidth, not E0).
    ampo_s = hubbard_opsum(p, L)
    ampo_s += -E_dmrg, "Id", 1
    Hs = build_H(ampo_s, sites)

    # ── φ̃(0) = J̃|0⟩, normalised; C0 = ⟨J²⟩ ──────────────────────────────────
    phi0 = apply(Jt, psi_gs; cutoff=1e-12, maxdim=χ)
    C0   = real(inner(phi0, phi0))
    @info "  [Verification] C0 = ⟨J²⟩ = $C0   flux(φ) == flux(ψ0): $(flux(phi0) == flux(psi_gs))   maxlinkdim(φ0) = $(maxlinkdim(phi0))"
    C0 > 1e-10 || error("J|ψ0⟩ vanishes — check the current operator")
    phi0 = (1 / sqrt(C0)) * phi0
    if PJ
        # C_PJ(0) = −⟨T⟩ exactly, from [P, J̃] = −T.
        CPJ0 = 2 * sqrt(C0) * real(inner(psi_gs', P, phi0))
        @info "  [Verification] C_PJ(0) = $CPJ0  vs  −⟨T⟩ = $(-Ekin)   (rel. err $(abs(CPJ0 + Ekin) / abs(Ekin)))"
        abs(CPJ0 + Ekin) / abs(Ekin) > 1e-6 &&
            @warn "  [Verification] C_PJ(0) ≠ −⟨T⟩ — P or J̃ is wrong."
    end

    # ── Time evolution ───────────────────────────────────────────────────────
    C_JJ  = zeros(ComplexF64, nt)   # C_JJ(2 t_k)
    z_PJ  = zeros(ComplexF64, nt)   # ⟨0|P|φ̃(t_k)⟩, C_PJ = 2 Re z_PJ
    chi_t = zeros(Int, nt)
    k_done = 1

    if isfile(ckpt_file)
        ck = load(ckpt_file)
        phi = ck["phi"]; k_done = ck["k_done"]
        C_JJ[1:k_done] = ck["C_JJ"][1:k_done]; z_PJ[1:k_done] = ck["z_PJ"][1:k_done]
        chi_t[1:k_done] = ck["chi_t"][1:k_done]
        siteinds(phi) == sites || @warn "checkpoint site indices differ from the ground state's"
        @info "── Resuming from checkpoint at step $k_done / $nt (t = $(times[k_done]))"
    else
        phi = MPS([complex(phi0[j]) for j in 1:length(phi0)])
    end

    measure!(k) = begin
        C_JJ[k]  = C0 * inner(conj_mps(phi), phi)
        PJ && (z_PJ[k] = sqrt(C0) * inner(psi_gs', P, phi))
        chi_t[k] = maxlinkdim(phi)
    end
    k_done == 1 && measure!(1)

    @info "── TDVP: $(nt - k_done) steps of dt = $dt, χ ≤ $χ, cutoff = $cutoff (2-site until χ is reached, then 1-site); Krylov $(UPDATER_KWARGS)"
    t_start = time(); k_start = k_done
    for k in (k_done + 1):nt
        nsite = maxlinkdim(phi) >= χ ? 1 : 2

        phi = tdvp(Hs, -im * dt, phi;
                   nsite=nsite, maxdim=χ, cutoff=cutoff, normalize=false, outputlevel=0,
                   updater_kwargs=UPDATER_KWARGS)
        measure!(k)

        if k % LOG_EVERY == 0 || k == nt
            el  = time() - t_start
            eta_s = el / (k - k_start) * (nt - k)
            @info "  t = $(round(times[k], digits=3))   ‖φ‖ = $(round(norm(phi), digits=8))   " *
                  "χ = $(chi_t[k]) ($(nsite)-site)   |C_JJ(2t)|/C0 = $(round(abs(C_JJ[k]) / C0, digits=5))   " *
                  "elapsed $(round(el/60, digits=1)) min, remaining ≈ $(round(eta_s/60, digits=1)) min"
            flush(io)
        end
        if k % CKPT_EVERY == 0 && k < nt
            k_done = k
            @save ckpt_file phi k_done C_JJ z_PJ chi_t
        end
    end
    t_tdvp = time() - t_start
    @info "  [Timer] TDVP completed in $(round(t_tdvp/3600, digits=3)) hours."
    @info "  [Memory] after TDVP: $(round(Sys.maxrss(), digits=2)) "

    # Doubled C_JJ(tmax) against the direct ⟨φ̃(0)|φ̃(tmax)⟩: equal up to truncation.
    if isodd(nt)
        j   = (nt + 1) ÷ 2
        dev = abs(C_JJ[j] - C0 * inner(phi0, phi)) / C0
        @info "  [Verification] |C_JJ(tmax) doubled − direct| / C0 = $(round(dev, sigdigits=3))"
    end
    norm_end = norm(phi)
    @info "  [Verification] ‖φ(tmax)‖ = $norm_end (1 up to truncation)   final χ = $(chi_t[end])"

    times_JJ = 2 .* times
    C_PJ     = 2 .* real.(z_PJ)

    @save data_file L U V N_total Sz_tot dt tmax χ cutoff χ_gs PJ E_dmrg Ekin C0 times times_JJ C_JJ C_PJ chi_t norm_end t_tdvp
    @info "  Saved → $data_file"

    rm(ckpt_file, force=true)
    rm(gs_file, force=true)
    @info "All MPS checkpoints deleted for $tag."
    @info "  [Memory] after production run: $(round(Sys.maxrss(), digits=2))"
    close(io)
end

run_point()
