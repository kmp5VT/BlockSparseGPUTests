using Plots

## So here I want to take a specific tensor and calculate
## the size of all blocks in the tensor. 
function get_tensor_block_sizes(A::ITensor)
  nblocks = Vector{Int}([nblocks(i) for i in inds(A)])
  
end