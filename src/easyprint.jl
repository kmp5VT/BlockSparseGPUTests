using ITensors: ITensor, Index
"""
```julia
easyprint(t::ITensor)
easyprint(i::Index)
```

  A simplified print version for BlockSparse ITensors. Does not show data simply prints
  The indices of each mode in the tensor and then prints each non-zero block and the dimension of
  each mode in the block
"""
function easyprint(t::ITensor)
  println("dimensions of indices")
  for i in inds(t)
    easyprint(i)
  end
  ten = tensor(t)
  if ten isa BlockSparseTensor
    println("\nDimensions of each block")
    for i in nzblocks(t)
      println("blockdim($(i)) = $(blockdims(ten,i))")
    end
  end
end

function easyprint(i::Index)
  dir = (
    if i.dir == 1
      "Out"
    elseif i.dir == -1
      "In"
    else
      "Neither"
    end
  )
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|$(tags(i)))")
end
