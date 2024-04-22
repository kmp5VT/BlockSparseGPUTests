using Plots

## So here I want to take a specific tensor and calculate
## the size of all blocks in the tensor. 
function get_tensor_block_sizes(A::ITensor)
  ## count the number of blocks in each index of A
  nblocks = Vector{Int}([nblocks(i) for i in inds(A)])

  ## for each index of A compute the product of the size of each block
  
end

## Here I want to 3d histogram plot block sizes of i, j and k in a tensor contraction
## So given a set of indices for i, compute the block sizes of each
function historgram_3d_contraction_blocks(indsI::Tuple, indsJ::Tuple, indsK::Tuple)

end