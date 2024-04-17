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
    print("\tIndex $(i):    ")
    summarize_index(ind(t, i))
  end
  if outputlevel >= 1
    printblocks(tensor(t))
  end
end

function summarize_index(i::Index)
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|tags=$(tags(i)))")
end

function summarize_index(i::Index{<:QNBlocks})
  dir = (
    if i.dir == 1
      "Out"
    elseif i.dir == -1
      "In"
    else
      "Neither"
    end
  )
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|tags=$(tags(i))|dir=$(dir))")
end

printblocks(t::Tensor) = println("All data in a single block with $(dim(t)) elements")

function printblocks(t::BlockSparseTensor)
  for i in nzblocks(t)
    println("\t  length of block $(i) is $(length(i))")
  end
end
