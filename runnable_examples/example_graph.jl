include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using HDF5, ITensors, TimerOutputs, NDTensors
using .BlockSparseGPUTests
using Plots

## Converts a block index into a tuple of their block sizes
## eg (QN() => 2, QN() => 3) -> (2,3)
function block_extents(ind::Index)
  return ntuple(i -> blockdim(ind, i), nblocks(ind))
end

## So here I want to take a specific tensor and calculate
## the size of all blocks in the tensor. 
function get_tensor_block_sizes(A::ITensor)
  is = inds(A)
  ## count the number of blocks in each index of A
  blockextents = [block_extents(i) for i in is]

  ## for each index of A compute the product of the size of each block
  hold = blockextents[1]
  for i in 2:length(blockextents)
    be = blockextents[i]
    l = length(hold) * length(be)
    hold_new = reshape([i * j for i in hold, j in be], l)
    hold = hold_new
  end

  return hold
end

## Here I want to 3d histogram plot block sizes of i, j and k in a tensor contraction
## So given a set of indices for i, compute the block sizes of each
function historgram_3d_contraction_blocks(indsI::Tuple, indsJ::Tuple, indsK::Tuple)
end

##
function histogram_tensor_block_size(
  A::ITensor; label::String, nonzero::Bool=false, color::Symbol
)
  ## convert tuple to vector for histogram
  if !nonzero
    AllBlockDimensions = (get_tensor_block_sizes(A))
    histogram(
      AllBlockDimensions;
      label=label,
      normalize=:true,
      color=color,
      bins=100,
      fillalpha=0.35,
    )
  else
    NZBlockDimensions = ([blockdim(tensor(A), i) for i in nzblocks(A)])
    histogram(
      NZBlockDimensions; label=label, normalize=:true, color=color, bins=100, fillalpha=0.35
    )
  end
end

function plot_block_sizes(prefix::String, size::String)
  timer = TimerOutput()
  foldername = "$prefix/$size/sparse"
  tensor_networks = ["EL1", "EL2", "S1", "S2", "S3"]
  for filename in tensor_networks
    fid = h5open("$(foldername)/$(filename).h5")
    T1 = read(fid, "T1", ITensor)
    T2 = read(fid, "T2", ITensor)
    close(fid)

    t = histogram_tensor_block_size(
      T1; label="$(filename) T1 All", color=:black, nonzero=false
    )
    t = histogram!(; title="T1 $(filename) $(size) all blocks")
    savefig("$(@__DIR__)/plots/$(size)/all_blocks/$(filename)_block_dimensions_T1.pdf")

    t = histogram_tensor_block_size(
      T1; label="$(filename) T1 NZ", color=:green, nonzero=true
    )
    t = histogram!(; title="T1 $(filename) $(size) nonzero blocks")
    savefig("$(@__DIR__)/plots/$(size)/nonzero/$(filename)_block_dimensions_T1.pdf")

    t = histogram_tensor_block_size(
      T2; label="$(filename) T2 All", color=:black, nonzero=false
    )
    t = histogram!(; title="T2 $(filename) $(size) all blocks")
    savefig("$(@__DIR__)/plots/$(size)/all_blocks/$(filename)_block_dimensions_T2.pdf")

    t = histogram_tensor_block_size(
      T2; label="$(filename) T2 NZ", color=:green, nonzero=true
    )
    t = histogram!(; title="T2 $(filename) $(size) nonzero blocks")
    savefig("$(@__DIR__)/plots/$(size)/nonzero/$(filename)_block_dimensions_T2.pdf")
  end
end

plot_block_sizes("$(@__DIR__)/hdf5", "medium")
