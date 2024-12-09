using ITensorMPS: ITensorMPS, AbstractObserver, linkind
using ITensors: ITensors

mutable struct BondDimObserver <: AbstractObserver
  ind::Int64
  maxdims::Vector{Int64}
end

function ITensorMPS.measure!(o::BondDimObserver; bond, psi, kwargs...)
  o.ind += 1
  if bond == length(psi) ÷ 2 && o.ind % 2 == 0
    i = linkind(psi, bond)
    bdims = [blockdim(i, j) for j in 1:nblocks(i)]
    push!(o.maxdims, maximum(bdims))
  end
end