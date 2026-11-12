using Printf
using ITensors
using JLD2
#include("../hamiltonians/hubbard.jl")
#include("../backend/utils.jl")


const N_NOISY = 15    # noisy sweeps at the start of every stage; the rest run clean

# Seeds (bond dim ≤ 16: product or random MPS): bond-dimension ramp, noise 1e-4 for 5 sweeps, then
# halving to 1e-8. Continuations: target χ, noise 1e-5 halving to 1e-8. Noise-free after N_NOISY sweeps.
function get_dmrg_results(psi_0, H; maxdim=1024, maxsweeps=10,
    energy_tol=1e-7, observer=nothing)

    seed = maxlinkdim(psi_0) <= 16
    nr = seed ? min(5, maxsweeps) : 0
    ramp = [i <= nr ? min(2^(i + 4), maxdim) : maxdim for i in 1:maxsweeps]
    noise = zeros(Float64, maxsweeps)
    for i in 1:min(N_NOISY, maxsweeps)
        noise[i] = i <= nr ? 1e-4 : max(1e-5 * 0.5^(i - nr - 1), 1e-8)
    end

    obs = isnothing(observer) ? DMRGObserver(; energy_tol=energy_tol) : observer

    return dmrg(H, psi_0;
        nsweeps=maxsweeps,
        noise=noise,
        maxdim=ramp,
        cutoff=1e-12,
        observer=obs,
    )
end


function get_hubbard_dmrg(data_folder, maxdim_list, Nx, Ny; t=-1, u=2)
    dmrg_file = @sprintf("%shubbard_%d_%d.jld2", data_folder, Nx, Ny)
    if !isfile(dmrg_file)
        H_dmrg, sites_dmrg = FermiHubbard_2D(Nx, Ny; t=t, u=u, conserve=true)
        states_dmrg = get_product_state(Nx, Ny)
        psi0_dmrg = randomMPS(sites_dmrg, states_dmrg, 16)
        E_list, psi_list = [], []
        for χ in maxdim_list
            println("maxdim = ", χ)
            E_dmrg, psi_dmrg = get_dmrg_results(psi0_dmrg, H_dmrg; maxsweeps=50, maxdim=χ)
            psi_dmrg = removeqns(psi_dmrg)
            append!(E_list, E_dmrg)
            append!(psi_list, [psi_dmrg])
        end

        save_object(dmrg_file, (E_list, psi_list, H_dmrg, sites_dmrg, states_dmrg))
    else
        (E_list, psi_list, H_dmrg, sites_dmrg, states_dmrg) = load_object(dmrg_file)
    end

    return (E_list, psi_list, H_dmrg, sites_dmrg, states_dmrg)
end
