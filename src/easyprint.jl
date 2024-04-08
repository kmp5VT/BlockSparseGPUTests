using ITensors: ITensor, Index
## another idea is to write 
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
  dir = (if i.dir == 1
    "Out"
  elseif i.dir == -1
    "In"
  else
    "Neither"
  end)
  return println("(dim=$(dim(i))|id=$(id(i)%1000)|$(tags(i)))")
end
