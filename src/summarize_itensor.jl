using ITensors: ITensor, Index, QNBlocks
using NDTensors: BlockSparseTensor, Tensor

"""
    summarize_itensor(t::ITensor)
    summarize_itensor(t::ITensor; printlevel::Int)

Optional argument:
  - 'outputlevel::Int' - Option to increase the information level. Level 0 shows indices only. 
    Level 1 also shows non-zero block locations and sizes.
  
A simplified printing function for block sparse ITensors. Does not show data. Instead prints
information about the tensor indices and the sizes and locations of the non-zero blocks.
"""
function summarize_itensor(t::ITensor; outputlevel::Int=0)
  r = order(t)
  println("Order-$(r) Tensor")
  for i in 1:r
    print("\tIndex $(i):\n\t\t")
    summarize_index(ind(t, i); outputlevel)
  end
  if outputlevel >= 2
    printblocks(tensor(t))
  end
end

function summarize_index(i::Index; outputlevel::Int=0)
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|tags=$(tags(i)))")
end

function summarize_index(i::Index{<:QNBlocks}; outputlevel::Int=0)
  dir = (
    if i.dir == 1
      "Out"
    elseif i.dir == -1
      "In"
    else
      "Neither"
    end
  )
  blockdims = [blockdim(i,j) for j in 1:nblocks(i)]
  println("(dim=$(dim(i))|id=$(id(i)%1000)|tags=$(tags(i))|dir=$(dir))")
  if outputlevel >= 1 
    println("\t\t\tblockdims:$(blockdims)")
  end
end

printblocks(t::Tensor) = println("All data in a single block with $(dim(t)) elements")

function printblocks(t::BlockSparseTensor)
  for i in nzblocks(t)
    println("\t  length of block $(i) is $(length(i))")
  end
end
