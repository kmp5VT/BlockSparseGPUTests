include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using HDF5, ITensors, TimerOutputs
using .BlockSparseGPUTests
using Plots

## Converts a block index into a tuple of their block sizes
## eg (QN() => 2, QN() => 3) -> (2,3)
function block_extents(ind::Index)
  ispace = ind.space
  ntuple(i -> dim(ispace[i]), length(ispace))
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
    hold_new = Tuple(reshape([i * j for i in hold, j in be], l))
    hold = hold_new
  end

  return hold
end

## Here I want to 3d histogram plot block sizes of i, j and k in a tensor contraction
## So given a set of indices for i, compute the block sizes of each
function historgram_3d_contraction_blocks(indsI::Tuple, indsJ::Tuple, indsK::Tuple)

end

##
function histogram_tensor_block_size(A::ITensor)
  ## convert tuple to vector for histogram
  AllBlockDimensions = [i for i in get_tensor_block_sizes(A)]
  histogram(AllBlockDimensions, label="All Block", color=:gray, bins=100)

  NZBlockDimensions = [length(i) for i in nzblocks(A)]

  histogram!(NZBlockDimensions, label="Nonzero Block", color=:green)
end

function plot_block_sizes(prefix::String; title::String)
  timer = TimerOutput()
  foldername = "$prefix/$size/sparse"
  tensor_networks = ["EL1", "EL2", "S1", "S2", "S3"]
  for filename in tensor_networks
    fid = h5open("$(foldername)/$(filename).h5")
    T1 = read(fid, "T1", ITensor)
    T2 = read(fid, "T2", ITensor)
    close(fid)

    histogram_tensor_block_size(T1)
  end
end
fid = h5open("runnable_examples/hdf5/small/sparse/S1.h5")
T1 = read(fid, "T1", ITensor)
T2 = read(fid, "T2", ITensor)
close(fid)

v = histogram_tensor_block_size(T2)
histogram!(title="Small T1")

