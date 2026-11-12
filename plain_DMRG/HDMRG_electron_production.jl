using Logging
using MKL
using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2

include("../src/analysis/hubbard_dmrg.jl")

# ─── Parse command line ──────────────────────────────────────────────────────

const L = parse(Int, ARGS[1])
const U = parse(Float64, ARGS[2])
const V = parse(Float64, ARGS[3])
const χ_vals = length(ARGS) >= 4 ? sort(parse.(Int, split(ARGS[4], ","))) : [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]
const GPU = length(ARGS) >= 5 && ARGS[5] == "gpu"

GPU && @eval using CUDA
to_gpu(x) = isdefined(NDTensors, :cu) ? NDTensors.cu(x) : CUDA.cu(x)   # NDTensors.cu keeps Float64

if Threads.nthreads() > 1 && !GPU
    BLAS.set_num_threads(1)
    NDTensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse(true)
else
    BLAS.set_num_threads(1)
    NDTensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse(false)
end

@info "Running: L=$L  U=$U  V=$V  χ_vals=$χ_vals  GPU=$GPU"

# ─── Types ───────────────────────────────────────────────────────────────────

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

# ─── Hamiltonian OpSum (particle-hole symmetric form) ────────────────────────

function hubbard_opsum(p::ModelParams, L)
    Nx, Ny = p.Nx, L
    N = Nx * Ny
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    ampo = OpSum()
    # Hopping
    for b in lattice
        ampo += p.t, "Cdagup", b.s1, "Cup", b.s2
        ampo += p.t, "Cdagup", b.s2, "Cup", b.s1
        ampo += p.t, "Cdagdn", b.s1, "Cdn", b.s2
        ampo += p.t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    # U term: U(n↑ - ½)(n↓ - ½) = U·Nupdn - U/2·Ntot + U/4
    for i in 1:N
        ampo += p.U, "Nupdn", i
        ampo += -p.U / 2, "Ntot", i
        ampo += p.U / 4, "Id", i
    end
    # V term: V(nᵢ - 1)(nⱼ - 1) = V·nᵢnⱼ - V·nᵢ - V·nⱼ + V
    for b in lattice
        ampo += p.V, "Ntot", b.s1, "Ntot", b.s2
        ampo += -p.V, "Ntot", b.s1
        ampo += -p.V, "Ntot", b.s2
        ampo += p.V, "Id", b.s1
    end
    # Chemical potential
    for i in 1:N
        ampo += -p.μ, "Ntot", i
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

# ─── Helper: extract sites from a loaded MPS ────────────────────────────────

function extract_sites(psi::MPS)
    return [siteind(psi, i) for i in 1:length(psi)]
end

# ─── State initialisation ───────────────────────────────────────────────────

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
    if GPU
        psi_0, H = to_gpu(psi_0), to_gpu(H)
        NDTensors.iscu(psi_0[1]) && NDTensors.iscu(H[1]) || error("state or Hamiltonian is not on the GPU")
        eltype(psi_0[1]) == Float64 || error("GPU tensors are $(eltype(psi_0[1])), not Float64")
    end
    E, psi = get_dmrg_results(psi_0, H; maxsweeps=cfg.maxsweeps, maxdim=cfg.χ, observer=obs)
    GPU && (psi = NDTensors.cpu(psi))

    err = isempty(obs.truncerrs) ? 0.0 : maximum(obs.truncerrs)
    return E, psi, err
end

# ─── Correlations ────────────────────────────────────────────────────────────

function compute_correlation_data(psi)
    n_total = real.(expect(psi, "Ntot"))
    Sz = real.(expect(psi, "Sz"))
    SzSz = real.(correlation_matrix(psi, "Sz", "Sz"))
    NN_tot = real.(correlation_matrix(psi, "Ntot", "Ntot"))
    return (; n_total, Sz, SzSz, NN_tot)
end

function calculate_spin_luttinger(corr)
    Lc = length(corr.Sz)
    q = 2π / Lc
    Sq = sum(cos(q * (i - j)) * (corr.SzSz[i, j] - corr.Sz[i] * corr.Sz[j])
             for i in 1:Lc, j in 1:Lc)
    return 4π * Sq / (Lc * q)
end

function calculate_charge_luttinger(corr)
    Lc = length(corr.n_total)
    q = 2π / Lc
    Nq = sum(cos(q * (k - l)) * (corr.NN_tot[k, l] - corr.n_total[k] * corr.n_total[l])
             for k in 1:Lc, l in 1:Lc)
    return Nq / 2
end

# ─── BOW ─────────────────────────────────────────────────────────────────────

function bond_operator(sites, i)
    ampo = OpSum()
    ampo += 0.5, "Cdagup", i, "Cup", i + 1
    ampo += 0.5, "Cdagup", i + 1, "Cup", i
    ampo += 0.5, "Cdagdn", i, "Cdn", i + 1
    ampo += 0.5, "Cdagdn", i + 1, "Cdn", i
    return MPO(ampo, sites)
end

function BOW_order(psi, sites, L)
    return abs(real(inner(psi', bond_operator(sites, L ÷ 2) - bond_operator(sites, (L ÷ 2) + 1), psi)))
end

# ─── Run single (L, U, V) with chi sweep ─────────────────────────────────────

function run_point()
    p = ModelParams(1, -1.0, U, V, 0.0)
    tag = "L$(L)_U$(U)_V$(V)" * (GPU ? "_gpu" : "")

    datadir = "data"
    statedir = "states"
    logdir = "logs"
    mkpath(datadir)
    mkpath(statedir)
    mkpath(logdir)

    data_file(χ) = joinpath(datadir, "dmrg_$(tag)_chi$(χ).jld2")
    state_file(χ) = joinpath(statedir, "mps_$(tag)_chi$(χ).jld2")

    if all(χ -> isfile(data_file(χ)), χ_vals)
        @info "All χ values already completed for $tag. Nothing to do."
        return
    end

    H, sites, ampo = setup_model(p, L)

    io = open(joinpath(logdir, "$(tag).log"), "a")
    global_logger(ConsoleLogger(io, Logging.Info))
    GPU && @info "  GPU: $(CUDA.name(CUDA.device()))"

    psi_gs = random_mps(sites, init_state(L, L, 0); linkdims=10)
    #psi_sz1 = random_mps(sites, init_state(L, L, 1); linkdims=10)
    #psi_p = random_mps(sites, init_state(L, L + 2, 0); linkdims=10)
    #psi_m = random_mps(sites, init_state(L, L - 2, 0); linkdims=10)

    for χ in χ_vals

        if isfile(data_file(χ))
            @info "── Data for χ = $χ already exists. Loading MPS checkpoint..."
            if isfile(state_file(χ))
                loaded = load(state_file(χ))
                psi_gs = loaded["psi_gs"]
                psi_sz1 = loaded["psi_sz1"]
                psi_p = loaded["psi_p"]
                psi_m = loaded["psi_m"]

                sites = extract_sites(psi_gs)
                H = build_H(ampo, sites)
                @info "  MPS checkpoint loaded & H rebuilt for χ = $χ. Skipping."
            else
                @info "  No MPS file found for χ = $χ (deleted?). Skipping with current states."
            end
            continue
        end

        cfg = DMRGConfig(χ)

        @info "── χ = $χ ──────────────────────────────────"

        time_taken = @elapsed begin
            @info "  Ground state (Sz=0, N=$L) ..."
            E_dmrg, psi_gs, err_dmrg = run_dmrg_sz(H, sites, L, L, 0, cfg; psi_init=psi_gs)
            @info "  [Memory] after GS: $(round(Sys.maxrss(), digits=2)) "

            @info "  Spin triplet (Sz=1, N=$L) ..."
           # E_Sz1, psi_sz1, err_Sz1 = run_dmrg_sz(H, sites, L, L, 1, cfg; psi_init=psi_sz1)
            @info "  [Memory] after Sz1: $(round(Sys.maxrss(), digits=2)) "

            @info "  N+2 sector ..."
            #E_p, psi_p, err_p = run_dmrg_sz(H, sites, L, L + 2, 0, cfg; psi_init=psi_p)
            @info "  [Memory] after N+2: $(round(Sys.maxrss(), digits=2)) "

            @info "  N-2 sector ..."
            #E_m, psi_m, err_m = run_dmrg_sz(H, sites, L, L - 2, 0, cfg; psi_init=psi_m)
            @info "  [Memory] after N-2: $(round(Sys.maxrss(), digits=2)) "
        end

        hours = round(time_taken / 3600, digits=3)
        @info "  [Timer] All 4 sectors for χ = $χ completed in $hours hours."

        #corr = compute_correlation_data(psi_gs)

        #charge_gap = (E_p + E_m - 2E_dmrg) / 2
        #spin_gap = E_Sz1 - E_dmrg
        #bow = BOW_order(psi_gs, sites, L)
        #K_σ = calculate_spin_luttinger(corr)
        #K_ρ = calculate_charge_luttinger(corr)

        #@info "  Results:" #charge_gap spin_gap bow K_σ K_ρ maxlinkdim = maxlinkdim(psi_gs) err_dmrg err_Sz1 err_p err_m

        @save state_file(χ) psi_gs #psi_sz1 psi_p psi_m
        t_dmrg, maxrss, nthreads = time_taken, Sys.maxrss(), Threads.nthreads()   # seconds for this χ step, bytes
        @save data_file(χ) L U V χ E_dmrg t_dmrg maxrss nthreads GPU #E_Sz1 E_p E_m err_dmrg err_Sz1 err_p err_m charge_gap spin_gap bow K_σ K_ρ corr
        @info "  Saved → $(data_file(χ)) + $(state_file(χ))"
        idx = findfirst(==(χ), χ_vals)
        if idx > 1
            prev = state_file(χ_vals[idx-1])
            rm(prev, force=true)
            @info "  Deleted previous checkpoint: $prev"
        end
    end
    for χ in χ_vals
        rm(state_file(χ), force=true)
    end
    @info "All MPS checkpoints deleted for $tag."
    @info "  [Memory] after production run: $(round(Sys.maxrss(), digits=2))"
    close(io)
end

run_point()