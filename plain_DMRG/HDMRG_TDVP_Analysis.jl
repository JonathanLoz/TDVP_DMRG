

using JLD2, Printf, CairoMakie
include(joinpath(@__DIR__, "figstyle.jl"));
import .FigStyle as FS

# ─── Command line ────────────────────────────────────────────────────────────

datadir = length(ARGS) >= 1 ? ARGS[1] : "data"
ηarg    = length(ARGS) >= 2 ? ARGS[2] : "auto"
const WCLOSE = 1e-3       # window value required at the end of the correlator
const VMAX   = 2.0        # velocity setting the finite-size level spacing 2πv/L
function ηof(L, τend)
    ηarg == "auto" || return parse(Float64, ηarg)
    ηclose = window == :gauss ? sqrt(2log(1 / WCLOSE)) / τend : log(1 / WCLOSE) / τend
    return max(ηclose, 2π * VMAX / L)
end
ωplot   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : Inf
window  = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :gauss
window in (:gauss, :lorentz) || error("window must be gauss or lorentz")
const NW = 2000                                     # ω grid points of every spectrum
outdir  = pwd()
adir    = (d = joinpath(outdir, "analysis"); mkpath(d); d)
nogrid = FS.theme()
r6(x) = round(x, digits=6)

# ─── Spectrum ────────────────────────────────────────────────────────────────

# σ(ω) from a correlator on a uniform τ grid (trapezoid); kernel = :JJ or :PJ (header).
function spectrum(τ, C, N, ω, η; kernel)
    nt = length(τ)
    h  = τ[2] - τ[1]
    w  = fill(h, nt); w[1] /= 2; w[end] /= 2
    Wt = window == :gauss ? (@. exp(-(η * τ)^2 / 2)) : (@. exp(-η * τ))
    σ  = similar(ω)
    for (m, ωm) in enumerate(ω)
        if kernel == :JJ
            s = zero(ComplexF64)
            @inbounds for k in 1:nt
                s += w[k] * Wt[k] * C[k] * cis(ωm * τ[k])
            end
            σ[m] = real(s) / (N * ωm)
        else
            s = 0.0
            @inbounds for k in 1:nt
                s += w[k] * Wt[k] * real(C[k]) * cos(ωm * τ[k])
            end
            σ[m] = s / N
        end
    end
    return σ
end

trapz(x, y) = sum((x[k+1] - x[k]) * (y[k+1] + y[k]) / 2 for k in 1:length(x)-1)

# ─── Load ────────────────────────────────────────────────────────────────────

# One run: parameters, correlators, and σ(ω) computed here. Reads the first data layout
# too (doubled correlator stored as C_JJ2 on 2·times).
load_point(f) = jldopen(f, "r") do d
    tag = replace(basename(f), r"^tdvp_" => "", r"\.jld2$" => "")
    L = d["L"]; U = d["U"]; V = d["V"]; N = d["N_total"]; Ekin = d["Ekin"]
    times = d["times"]
    tJJ   = haskey(d, "times_JJ") ? d["times_JJ"] : 2 .* times
    CJJ   = haskey(d, "C_JJ2") ? d["C_JJ2"] : d["C_JJ"]
    η     = round(ηof(L, tJJ[end]), sigdigits=3)
    CPJ   = d["C_PJ"]
    havePJ = any(!iszero, CPJ)
    W_JJ = window == :gauss ? exp(-(η * tJJ[end])^2 / 2) : exp(-η * tJJ[end])
    W_JJ > 1e-3 && @warn "$tag: window not closed at 2 tmax (W = $(round(W_JJ, sigdigits=2))): σ_JJ will ring; raise η or tmax."
    if havePJ
        W_PJ = window == :gauss ? exp(-(η * times[end])^2 / 2) : exp(-η * times[end])
        W_PJ > 1e-3 && @warn "$tag: window not closed at tmax (W = $(round(W_PJ, sigdigits=2))): σ_PJ will ring."
    end
    ωmax = 8.0 + U + 4 * abs(V) + 2.0
    ω    = collect(range(ωmax / NW, ωmax; length=NW))
    σJJ  = spectrum(tJJ, CJJ, N, ω, η; kernel=:JJ)
    σPJ  = havePJ ? spectrum(times, CPJ, N, ω, η; kernel=:PJ) : zeros(0)
    (; tag, L, U, V, N, Sz=d["Sz_tot"], dt=d["dt"], tmax=d["tmax"], χ=d["χ"], η,
        E=d["E_dmrg"], Ekin, C0=d["C0"], tJJ, CJJ, chi_t=d["chi_t"], ω, σJJ, σPJ,
        fJJ=trapz(ω, σJJ), fPJ=havePJ ? trapz(ω, σPJ) : NaN, ftarget=-π * Ekin / (2N))
end

files = filter(f -> startswith(basename(f), "tdvp_") && endswith(f, ".jld2"),
               readdir(datadir, join=true))
isempty(files) && error("No tdvp_*.jld2 in $datadir")
recs = [load_point(f) for f in files]
sort!(recs, by=r -> (r.L, r.V, r.N, r.Sz, r.U, r.χ))
@info "Loaded $(length(recs)) run(s) from $datadir"

# ─── Peaks ───────────────────────────────────────────────────────────────────

# Local maxima of σ above frac of its maximum: the excitation energies
function peaks(ω, σ; frac=0.02)
    thr = frac * maximum(σ)
    [(ω[i], σ[i]) for i in 2:length(σ)-1 if σ[i] > thr && σ[i] >= σ[i-1] && σ[i] > σ[i+1]]
end

# ─── Per-run figure and summary ──────────────────────────────────────────────

csv = open(joinpath(adir, "tdvp_summary.csv"), "w")
println(csv, "tag,L,U,V,N,Sz,dt,tmax,chi,eta,E,Ekin,C0,fsum_JJ,fsum_PJ,fsum_target,chi_final,tau_chi_cap,peaks_omega")

for r in recs
    havePJ = !isempty(r.σPJ)
    ωmax   = min(ωplot, r.ω[end])
    m      = r.ω .<= ωmax
    pk     = peaks(r.ω[m], r.σJJ[m])
    icap   = findfirst(==(r.χ), r.chi_t)
    tcap   = icap === nothing ? NaN : r.tJJ[icap]              # in correlator time 2t

    println("\n── $(r.tag)")
    @printf("   E = %.10f   ⟨T⟩ = %.8f   C0 = ⟨J²⟩ = %.8f\n", r.E, r.Ekin, r.C0)
    @printf("   f-sum: ∫σ_JJ = %.6f", r.fJJ); havePJ && @printf("   ∫σ_PJ = %.6f", r.fPJ)
    @printf("   target −π⟨T⟩/2N = %.6f\n", r.ftarget)
    @printf("   χ(t): final %d of cap %d%s\n", r.chi_t[end], r.χ,
            isnan(tcap) ? " (cap never reached)" : @sprintf(" (cap reached at correlator time %.2f)", tcap))
    println("   peaks of σ_JJ(ω):")
    for (n, (ωp, sp)) in enumerate(pk)
        @printf("     n = %d   ω = %.4f   σ = %.4f\n", n, ωp, sp)
    end
    println(csv, join([r.tag, r.L, r.U, r.V, r.N, r.Sz, r.dt, r.tmax, r.χ, r.η,
                       r6(r.E), r6(r.Ekin), r6(r.C0), r6(r.fJJ), havePJ ? r6(r.fPJ) : "",
                       r6(r.ftarget), r.chi_t[end], tcap,
                       join([@sprintf("%.4f", p[1]) for p in pk], ";")], ","))

    with_theme(nogrid) do
        fig = Figure(size=FS.figsize(:double, aspect=0.42))
        title = L"L=%$(r.L),\ U=%$(r.U),\ V=%$(r.V),\ N=%$(r.N)"

        # (a) correlator
        ax = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"C_{JJ}(\tau)/C_{JJ}(0)", title=title,
                  xticks=LinearTicks(FS.NTICKS), yticks=LinearTicks(FS.NTICKS))
        c = r.CJJ ./ r.C0
        lines!(ax, r.tJJ, real.(c), color=FS.CAT(3)[1], label="Re")
        lines!(ax, r.tJJ, imag.(c), color=FS.CAT(3)[2], linestyle=FS.dash(2), label="Im")
        hlines!(ax, [0.0], color=:gray, linestyle=:dot, linewidth=FS.P().lwthin)
        xlims!(ax, 0, r.tJJ[end])
        ylims!(ax, -1.15, 1.55)                     # |C/C0| ≤ 1: headroom for the legend
        FS.legend!(fig, ax; position=:rt, maxrows=1)

        # (b) optical conductivity
        bx = Axis(fig[1, 2], xlabel=L"\omega", ylabel=L"\sigma(\omega)",
                  title=L"\eta=%$(r.η)\ (\mathrm{%$(window)}),\ \tau_{\max}=%$(r.tJJ[end]),\ \chi\le%$(r.χ)",
                  xticks=LinearTicks(FS.NTICKS), yticks=LinearTicks(FS.NTICKS))
        lines!(bx, r.ω[m], r.σJJ[m], color=FS.CAT(3)[1], label=L"J\!-\!J")
        havePJ && lines!(bx, r.ω[m], r.σPJ[m], color=FS.CAT(3)[3], linestyle=FS.dash(2),
                         label=L"P\!-\!J")
        isempty(pk) || scatter!(bx, first.(pk), last.(pk), color=:black, marker=:vline,
                                markersize=1.6 * FS.P().ms)
        xlims!(bx, 0, ωmax)
        ylims!(bx, 0, 1.15 * maximum(r.σJJ[m]))
        FS.legend!(fig, bx; position=:rt)

        Label(fig[1, 1, TopLeft()], "(a)", font=:bold, padding=(0, 6, 4, 0))
        Label(fig[1, 2, TopLeft()], "(b)", font=:bold, padding=(0, 6, 4, 0))
        FS.savefig(fig, joinpath(adir, "tdvp_$(r.tag).png"))
    end
end
close(csv)
@info "Summary → $(joinpath(adir, "tdvp_summary.csv"))"

# ─── σ(ω) overlays: U at fixed (L, V, N, Sz) with the largest χ of each U; χ at fixed U ─────

function overlay(rs, key, fixed, fname)
    val(r) = key == :chi ? r.χ : r.U                     # the ordered quantity for the colours
    mixχ   = key == :U && length(unique(r.χ for r in rs)) > 1
    lab(r) = key == :chi ? L"\chi=%$(r.χ)" : mixχ ? L"U=%$(r.U),\ \chi=%$(r.χ)" : L"U=%$(r.U)"
    lo, hi = extrema(val.(rs))
    cat = FS.categorical(length(rs))
    with_theme(nogrid) do
        fig = Figure(size=FS.figsize(:single, cb=!cat))
        ηtxt = join(string.(unique(r.η for r in rs)), ", ")
        ax = Axis(fig[1, 1], xlabel=L"\omega", ylabel=L"\sigma(\omega)",
                  title=L"%$(fixed),\ \eta=%$(ηtxt)",
                  xticks=LinearTicks(FS.NTICKS), yticks=LinearTicks(FS.NTICKS))
        ωmax = min(ωplot, maximum(r.ω[end] for r in rs))
        for (i, r) in enumerate(sort(rs, by=val))
            st = FS.series_style(i, length(rs); value=val(r), lo=lo, hi=hi)
            m = r.ω .<= ωmax
            lines!(ax, r.ω[m], r.σJJ[m], color=st.color, linestyle=st.linestyle, label=lab(r))
        end
        xlims!(ax, 0, ωmax)
        if cat
            FS.legend!(fig, ax; position=:rt)
        else
            Colorbar(fig[1, 2], limits=(lo, hi), colormap=FS.SEQ, label=key == :chi ? L"\chi" : L"U")
        end
        FS.savefig(fig, joinpath(adir, fname))
    end
end

byU = Dict{Any,Vector{eltype(recs)}}()
for r in recs; push!(get!(byU, (r.L, r6(r.V), r.N, r.Sz), eltype(recs)[]), r); end
for ((L_, V_, N_, Sz_), rs) in byU
    best = [argmax(r -> r.χ, filter(r -> r.U == U, rs)) for U in unique(r.U for r in rs)]
    length(best) > 1 || continue
    χs = unique(r.χ for r in best)
    fixed = "L=$(L_),\\ V=$(V_),\\ N=$(N_)" * (length(χs) == 1 ? ",\\ \\chi\\le$(χs[1])" : "")
    overlay(best, :U, fixed, "sigma_U_L$(L_)_V$(V_)_N$(N_)_Sz$(Sz_).png")
end

byχ = Dict{Any,Vector{eltype(recs)}}()
for r in recs; push!(get!(byχ, (r.L, r.U, r6(r.V), r.N, r.Sz), eltype(recs)[]), r); end
for ((L_, U_, V_, N_, Sz_), rs) in byχ
    length(unique(r.χ for r in rs)) > 1 || continue
    overlay(rs, :chi, "L=$(L_),\\ U=$(U_),\\ V=$(V_),\\ N=$(N_)",
            "sigma_chi_L$(L_)_U$(U_)_V$(V_)_N$(N_)_Sz$(Sz_).png")
end
