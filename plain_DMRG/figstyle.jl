module FigStyle

using CairoMakie

const MM = 72 / 25.4
mm(x) = x * MM

const PROFILES = Dict(
    :screen => (; wsingle = 440.0, wdouble = 640.0, aspect = 0.6875,
                  base = 12.0, lab = 11.0, tick = 10.0,
                  lw = 1.6, lwref = 1.0, lwthin = 0.8, ms = 7.0,
                  cbwidth = 12.0, pad = 6.0, px_per_unit = 4.0),
    :print  => (; wsingle = 85 * MM, wdouble = 170 * MM, aspect = 0.62,
                  base = 9.0, lab = 10.0, tick = 9.0,
                  lw = 0.9, lwref = 0.6, lwthin = 0.45, ms = 3.2,
                  cbwidth = 6.0, pad = 3.0, px_per_unit = 600 / 72),
)

const _PROFILE = Ref{Symbol}(:print)
profile!(p::Symbol) = (_PROFILE[] = p)
profile() = _PROFILE[]
P() = PROFILES[_PROFILE[]]

function figsize(cols::Symbol = :double; aspect = nothing, cb::Bool = false)
    p = P()
    w = cols === :single ? p.wsingle : p.wdouble
    h = w * (aspect === nothing ? p.aspect : aspect)
    cb && (w += 4.2 * p.cbwidth)
    (w, h)
end

function theme()
    p = P()
    merge(theme_latexfonts(), Theme(
        fontsize = p.base,
        figure_padding = p.pad,
        Axis = (; xlabelsize = p.lab, ylabelsize = p.lab, titlesize = p.lab,
                  xticklabelsize = p.tick, yticklabelsize = p.tick,
                  xgridvisible = false, ygridvisible = false,
                  xminorgridvisible = false, yminorgridvisible = false,
                  xminorticksvisible = true, yminorticksvisible = true,
                  xminorticks = IntervalsBetween(2), yminorticks = IntervalsBetween(2),
                  xticksize = 0.5 * p.tick, yticksize = 0.5 * p.tick,
                  xminorticksize = 0.3 * p.tick, yminorticksize = 0.3 * p.tick,
                  spinewidth = p.lwref, xtickwidth = p.lwref, ytickwidth = p.lwref),
        Lines        = (; linewidth = p.lw),
        Scatter      = (; markersize = p.ms),
        ScatterLines = (; linewidth = p.lw, markersize = p.ms),
        Errorbars    = (; linewidth = p.lwthin, whiskerwidth = 0.8 * p.ms),
        Band         = (; alpha = 0.18),
        Colorbar     = (; labelsize = p.lab, ticklabelsize = p.tick,
                          size = p.cbwidth, spinewidth = p.lwref),
        Legend       = (; labelsize = p.tick, framevisible = false,
                          patchsize = (Float32(1.2 * p.lab), Float32(0.8 * p.tick)),
                          padding = (2f0, 2f0, 2f0, 2f0), rowgap = 0),
    ))
end

legend_kw() = (; framevisible = false, labelsize = P().tick,
                 patchsize = (Float32(1.2 * P().lab), Float32(0.8 * P().tick)),
                 padding = (2f0, 2f0, 2f0, 2f0), rowgap = 0)

const LEGEND_MAXROWS = 3
legend_inset() = Float32(1.2 * P().lab + 6)

function n_legend_entries(ax)
    try
        return length(CairoMakie.Makie.get_labeled_plots(ax; merge = false, unique = false))
    catch
    end
    n = 0
    for pl in ax.scene.plots
        lb = get(pl.attributes, :label, nothing)
        lb === nothing && continue
        v = lb[]
        (v === nothing || v == "") || (n += 1)
    end
    n
end

function legend!(fig, ax; position = :rt, maxrows = LEGEND_MAXROWS, margin = nothing, kw...)
    n = n_legend_entries(ax)
    n == 0 && return nothing
    m = margin === nothing ? ntuple(_ -> legend_inset(), 4) : margin
    axislegend(ax; position = position, nbanks = max(1, cld(n, maxrows)), colgap = 6,
               margin = m, legend_kw()..., kw...)
    nothing
end

const NTICKS = 6
const SEQ = :cividis
const MASK = RGBAf(0.82, 0.82, 0.82, 1.0)
const INTERPOLATE_DEFAULT = true
const OVERFLOW  = RGBAf(0.60, 0.08, 0.14, 1.0)
const UNDERFLOW = RGBAf(0.72, 0.89, 0.80, 1.0)

function clipped_range(lo, hi; clip_lo = nothing, clip_hi = nothing)
    l = clip_lo === nothing ? float(lo) : max(float(lo), float(clip_lo))
    h = clip_hi === nothing ? float(hi) : min(float(hi), float(clip_hi))
    h <= l && (h = l * 10)
    kw = NamedTuple()
    (clip_hi !== nothing && float(hi) > h) && (kw = merge(kw, (; highclip = OVERFLOW)))
    (clip_lo !== nothing && float(lo) < l) && (kw = merge(kw, (; lowclip  = UNDERFLOW)))
    ((l, h), kw)
end

function _ticklabel(v)
    r = round(float(v), sigdigits = 2)
    (r >= 1 && isinteger(r)) ? string(Int(r)) : string(r)
end

function log_ticks(lo, hi)
    lo <= 0 && (lo = hi / 1e6)
    e0 = floor(Int, log10(lo)); e1 = ceil(Int, log10(hi))
    pos = [m * 10.0^e for e in e0:e1 for m in (1, 2, 5) if lo <= m * 10.0^e <= hi]
    for v in (float(lo), float(hi))
        any(t -> abs(log10(t) - log10(v)) < 0.05, pos) || push!(pos, v)
    end
    sort!(pos)
    (pos, [_ticklabel(v) for v in pos])
end

function log_minorticks(lo, hi)
    lo <= 0 && (lo = hi / 1e6)
    e0 = floor(Int, log10(lo)); e1 = ceil(Int, log10(hi))
    [m * 10.0^e for e in e0:e1 for m in 2:9 if lo <= m * 10.0^e <= hi]
end

logcb_kw(lo, hi; log::Bool = true) =
    log ? (; ticks = log_ticks(lo, hi), minorticks = log_minorticks(lo, hi),
             minorticksvisible = true, minortickwidth = P().lwref,
             minorticksize = 0.3 * P().tick) : NamedTuple()

CAT(n::Integer) = cgrad(:tab10, max(n, 2), categorical = true)

const MARKERS = [:circle, :rect, :utriangle, :diamond, :dtriangle, :pentagon, :star5, :hexagon]
const DASHES  = [:solid, :dash, :dot, :dashdot, :dashdotdot]
marker(i::Integer) = MARKERS[mod1(i, length(MARKERS))]
dash(i::Integer)   = DASHES[mod1(i, length(DASHES))]

const CATMAX = 8
categorical(n::Integer) = n <= CATMAX

function series_style(i::Integer, n::Integer; value = nothing, lo = 0.0, hi = 1.0)
    cat = categorical(n)
    col = if cat
        CAT(n)[i]
    elseif value === nothing || !(hi > lo)
        cgrad(SEQ)[0.5]
    else
        cgrad(SEQ)[clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)]
    end
    (color = col, marker = marker(i), linestyle = dash(i), categorical = cat)
end

function linfit_se(x, y)
    n = length(x)
    n == 0 && return (a = NaN, b = NaN, sa = NaN, sb = NaN, s = NaN, n = 0)
    n == 1 && return (a = float(y[1]), b = 0.0, sa = NaN, sb = NaN, s = NaN, n = 1)
    xb = sum(x) / n; yb = sum(y) / n
    Sxx = sum((xi - xb)^2 for xi in x)
    Sxy = sum((x[i] - xb) * (y[i] - yb) for i in 1:n)
    b = Sxx == 0 ? 0.0 : Sxy / Sxx
    a = yb - b * xb
    (n < 3 || Sxx == 0) && return (a = a, b = b, sa = NaN, sb = NaN, s = NaN, n = n)
    s2 = sum((y[i] - a - b * x[i])^2 for i in 1:n) / (n - 2)
    (a = a, b = b, sa = sqrt(max(s2 * (1 / n + xb^2 / Sxx), 0.0)),
     sb = sqrt(max(s2 / Sxx, 0.0)), s = sqrt(max(s2, 0.0)), n = n)
end

function quadsum(ses...)
    v = [float(s) for s in ses if isfinite(s)]
    length(v) == length(ses) ? sqrt(sum(abs2, v)) : NaN
end

function errorbars_if!(ax, x, y, se; kw...)
    ok = [i for i in eachindex(x) if isfinite(se[i]) && se[i] > 0]
    isempty(ok) || errorbars!(ax, collect(x)[ok], collect(y)[ok], collect(se)[ok]; kw...)
    [i for i in eachindex(x) if !(i in ok)]
end

function savefig(fig, path::AbstractString)
    base = replace(String(path), r"\.(png|pdf|svg)$" => "")
    dir, name = dirname(base), basename(base)
    for ext in ("pdf", "svg")
        mkpath(joinpath(dir, ext))
        CairoMakie.save(joinpath(dir, ext, name * "." * ext), fig; pt_per_unit = 1)
    end
    mkpath(joinpath(dir, "png"))
    CairoMakie.save(joinpath(dir, "png", name * ".png"), fig; px_per_unit = P().px_per_unit)
    nothing
end

end
