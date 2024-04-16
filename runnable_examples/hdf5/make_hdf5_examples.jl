include("$(@__DIR__)/../../src/BlockSparseGPUTests.jl")
using .BlockSparseGPUTests
using .BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge
using ITensors, HDF5

# hdf5/dense_small
function get_model(model_size::String)
    if model_size == "small"
        model = Model{TwoDHubbSmall}()
    elseif model_size == "medium"
        model = Model{TwoDHubbMed}()
    elseif model_size == "large"
        model = Model{TwoDHubbLarge}()
    else
        error("Model $(model_size) is not currently supported keyword")
    end
    return model 
end

function make_all_tensor_networks(ψ, H, site)    
    EL11,EL12 = make_EL1_network(ψ, H, site)
    EL21, EL22 = make_EL2_network(EL11, EL12, H, site)

    ## Components of the two site tensor
    S11 = ψ[site]
    S12 = ψ[site + 1]

    ## Contract the twosite with the environment tensor
    S21, S22 = make_S2_network(EL21, ψ, S11, S12, site)

    S31, S32 = make_S3_network(ψ, H, S22, site)

    return (EL11, EL12), (EL21, EL22), (S11, S12), (S21, S22), (S31, S32)
end

function make_EL1_network(ψ, H, site)
    ## Create the LHS environment and grab the site -2 environment
    L = BlockSparseGPUTests.left_environment_tensor(ψ, H; N=site)[site - 2]
    ## Grab the next MPS from ψ*
    EL12 = dag(ψ[site - 1])'

    return L, EL12
end

function make_EL2_network(EL11, EL12, H, site)
    EL21 = EL11 * EL12
    EL22 = H[site-1]
    return EL21, EL22
end

function make_S2_network(EL21, ψ, S11, S12, site)
    S21 = EL21 * ψ[site-1]
    S22 = S11 * S12
    return S21, S22
end

function make_S3_network(ψ, H, S22, site)
    S31 = S22 * H[site] * H[site + 1]
    S32 = BlockSparseGPUTests.right_environment_tensor(ψ, H; N=site)[site + 2]
    return S31, S32
end

function make_and_write_hdf5_tensor_networks(foldername; blocksparse = false, model_size::String = "small", site = nothing)
    ψ, H = construct_psi_h("two_d_hubbard"; conserve_qns = blocksparse, model = get_model(model_size))

    h5open("$(foldername)/psi.h5", "w") do fid
        fid["psi"] = ψ 
    end
    h5open("$(foldername)/H.h5", "w") do fid
        fid["H"] = H
    end

    site = isnothing(site) ? div(length(ψ), 2) : site
    TNs = make_all_tensor_networks(ψ, H, site)

    EL1 = TNs[1]
    h5open("$(foldername)/EL1.h5", "w") do fid
        fid["T1"] = EL1[1]
        fid["T2"] = EL1[2]
    end
    
    EL2 = TNs[2]
    h5open("$(foldername)/EL2.h5", "w") do fid
        fid["T1"] = EL2[1]
        fid["T2"] = EL2[2]
    end

    S1 = TNs[3]
    h5open("$(foldername)/S1.h5", "w") do fid
        fid["T1"] = S1[1]
        fid["T2"] = S1[2]
    end
    
    S2 = TNs[4]
    h5open("$(foldername)/S2.h5", "w") do fid
        fid["T1"] = S2[1]
        fid["T2"] = S2[2]
    end

    S3 = TNs[5]
    h5open("$(foldername)/S3.h5", "w") do fid
        fid["T1"] = S3[1]
        fid["T2"] = S3[2]
    end
    return nothing
end

make_and_write_hdf5_tensor_networks("$(@__DIR__)/small/dense"; blocksparse = false, model_size="small")
make_and_write_hdf5_tensor_networks("$(@__DIR__)/small/sparse"; blocksparse = true, model_size="small")
