using ITensors: ITensor, Index, QNBlocks
using NDTensors: BlockSparseTensor, Tensor
"""
```julia
easyprint(t::ITensor)
easyprint(t::ITensor; printlevel::Int)
easyprint(i::Index)
```

Optional Argument:

  - 'printlevel::Int' - Option to increase the print level, level 0 shows indices level 1 shows block indices with block size; 0
  
  A simplified print version for BlockSparse ITensors. Does not show data simply prints
  The indices of each mode in the tensor and then prints each non-zero block and the dimension of
  each mode in the block
"""
function easyprint(t::ITensor; printlevel::Int=0)
  r = order(t)
  println("Order-$(r) Tensor")
  for i in 1:r
    print("Index $(i):\t")
    easyprint(ind(t,i))
  end
  if printlevel == 1
    printblocks(tensor(t))
  end
end

function easyprint(i::Index)
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|tags=$(tags(i)))")
end

function easyprint(i::Index{<:QNBlocks})
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

printblocks(t::Tensor) = println("All data in a single block with $(dim(t)) elements" )

function printblocks(t::BlockSparseTensor)
  for i in nzblocks(t)
    println("blockdim($(i)) = $(length(i))")
  end
end