include("$(@__DIR__)/../../src/BlockSparseGPUTests.jl")
using .BlockSparseGPUTests
using .BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge
using ITensors, ITensorMPS, JLD2

struct long_medium end
function BlockSparseGPUTests.default_vals(::BlockSparseGPUTests.Model{<:long_medium})
  conserve_qns = false
  yperiodic = true
  Nx = 60
  Ny = 4
  N = Nx * Ny

  U = 4.0
  t = 1.0

  nsweeps = 5
  maxdim = [100, 200, 200, 400, 400, 800, 1600, 5000]
  cutoff = [1e-10]
  noise = [1E-6, 1E-7, 1E-8, 0.0]
  return conserve_qns, yperiodic, Nx, Ny, U, t, nsweeps, maxdim, cutoff, noise
end

# hdf5/dense_small
function get_model(model_size::String)
  if model_size == "small"
    model = Model{TwoDHubbSmall}()
  elseif model_size == "medium"
    model = Model{TwoDHubbMed}()
  elseif model_size == "large"
    model = Model{TwoDHubbLarge}()
  elseif model_size == "long_medium"
    model = Model{long_medium}()
  else
    error("Model $(model_size) is not currently supported keyword")
  end
  return model
end

function make_and_write_tensor_networks(
  foldername; blocksparse=false, model_size::String="small", site=nothing
)
  for i in [8]
    ψ, h = construct_psi_h(
      "two_d_hubbard"; conserve_qns=blocksparse, model=get_model(model_size), nsweeps=i
    )

    psi = BlockSparseGPUTests.remove_data_from_ITensor.(ψ)
    H = BlockSparseGPUTests.remove_data_from_ITensor.(h)

    jldsave("$(foldername)/psi_sweep_$(i).jld"; psi)
    # jldsave("$(foldername)/H.jld"; H)
  end

  # site = isnothing(site) ? div(length(ψ), 2) : site
  # TNs = make_all_tensor_networks(ψ, h, site)
  # names = ["E1", "E2", "S1", "S2", "S3"]

  # for i in 1:length(names)
  #   TN = TNs[i]
  #   TN = BlockSparseGPUTests.remove_data_from_ITensor.(TN)
  #   jldopen("$(foldername)/$(names[i]).jld", "w") do fid
  #     fid["T1"] = TN[1]
  #     fid["T2"] = TN[2]
  #   end
  # end

  return nothing
end

make_and_write_tensor_networks(
  "$(@__DIR__)/long_medium/sparse/scan_60_4"; blocksparse=true, model_size="long_medium"
)

function make_and_write_2d_momentum_hubbard(
  foldername;
  conserve_ky::Bool=false,
  conserve_sz::Bool=false,
  conserve_nf::Bool=false,
  conserve_nfparity::Bool=false,
  model_size::String="small",
  site=nothing,
  write_only_psi::Bool=false,
)
  for i in [5]
    ψ, h = construct_psi_h(
      "two_d_hubbard";
      conserve_ky,
      conserve_sz,
      conserve_nf,
      conserve_nfparity,
      model=get_model(model_size),
      nsweeps=i,
    )

    if write_only_psi
      psi = BlockSparseGPUTests.remove_data_from_ITensor.(ψ)
      jldsave("$(foldername)/psi_sweep_$(i).jld"; psi)
    else
      site = isnothing(site) ? div(length(ψ), 2) : site
      TNs = make_all_tensor_networks(ψ, h, site)
      names = ["E1", "E2", "S1", "S2", "S3"]

      for i in 1:length(names)
        TN = TNs[i]
        TN = BlockSparseGPUTests.remove_data_from_ITensor.(TN)
        jldopen("$(foldername)/$(names[i]).jld", "w") do fid
          fid["T1"] = TN[1]
          fid["T2"] = TN[2]
        end
      end
    end
  end

  return nothing
end

make_and_write_2d_momentum_hubbard(
  "$(@__DIR__)/2d_momentum_hubbard/medium/dense"; model_size="medium"
)

ψ, h = construct_psi_h(
  "two_d_hubbard";
  conserve_ky,
  conserve_sz,
  conserve_nf,
  conserve_nfparity,
  model=get_model(model_size),
  nsweeps=i,
)
