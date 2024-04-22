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

function make_and_write_hdf5_tensor_networks(
  foldername; blocksparse=false, model_size::String="small", site=nothing
)
  ψ, H = construct_psi_h(
    "two_d_hubbard"; conserve_qns=blocksparse, model=get_model(model_size)
  )

  h5open("$(foldername)/psi.h5", "w") do fid
    fid["psi"] = ψ
  end
  h5open("$(foldername)/H.h5", "w") do fid
    fid["H"] = H
  end

  site = isnothing(site) ? div(length(ψ), 2) : site
  TNs = make_all_tensor_networks(ψ, H, site)
  names = ["EL1", "EL2", "S1", "S2", "S3"]

  for i in 1:length(names)
    TN = TNs[i]
    h5open("$(foldername)/$(names[i]).h5", "w") do fid
      fid["T1"] = TN[1]
      fid["T2"] = TN[2]
    end
  end


  return nothing
end

make_and_write_hdf5_tensor_networks(
  "$(@__DIR__)/small/dense"; blocksparse=false, model_size="small"
)
make_and_write_hdf5_tensor_networks(
  "$(@__DIR__)/small/sparse"; blocksparse=true, model_size="small"
)

# make_and_write_hdf5_tensor_networks(
#   "$(@__DIR__)/medium/dense"; blocksparse=false, model_size="medium"
# )
# make_and_write_hdf5_tensor_networks(
#   "$(@__DIR__)/medium/sparse"; blocksparse=true, model_size="medium"
# )
