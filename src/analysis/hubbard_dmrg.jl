using Printf
using ITensors
using JLD2
#include("../hamiltonians/hubbard.jl")
#include("../backend/utils.jl")


const NOISE_FLOOR = 1e-8    #held at across the bulk of the sweeps
const N_CLEAN     = 15      # final noise-free sweeps so the observer sees clean convergence

function get_dmrg_results(psi_0, H; maxdim=1024, maxsweeps=10,
    energy_tol=1e-7, observer=nothing)
    
    start_dim = maxlinkdim(psi_0)
    fresh = start_dim <= 32    
    
    if fresh
        ramp = [min(2^(i + 4), maxdim) for i in 1:min(5, maxsweeps)]
        append!(ramp, fill(maxdim, max(0, maxsweeps - length(ramp))))
        ramp_len = min(5, maxsweeps)
        noise = zeros(Float64, maxsweeps)
        for i in 1:maxsweeps
            if i <= ramp_len
                noise[i] = 1e-4
            elseif i > maxsweeps - N_CLEAN
                noise[i] = 0.0
            else
                noise[i] = max(1e-5 * 0.5^(i - ramp_len - 1), NOISE_FLOOR)  
            end
        end
    else
        ramp = fill(maxdim, maxsweeps)

        # Warm start: held near NOISE_FLOOR for many sweeps
        # (soft-mode relaxation), off only for the final N_CLEAN clean-up sweeps.
        noise = zeros(Float64, maxsweeps)
        for i in 1:maxsweeps
            if i > maxsweeps - N_CLEAN
                noise[i] = 0.0
            else
                noise[i] = max(1e-5 * 0.5^(i - 1), NOISE_FLOOR)  # 1e-5 → floor, held
            end
        end
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
