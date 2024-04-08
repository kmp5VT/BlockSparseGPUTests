using NDTensors, ITensors
using TimerOutputs

include("1d_heisenberg_conserve_spin.jl")
## Model is a string with option of 
 # one_d_heisenberg
 # two_d_hubbard
function construct_psi_h(model::String; N=5, conserve_qns=nothing, conserve_pn=nothing, nsweeps=nothing, cutoff = nothing, maxdim=nothing, noise=nothing)
  if model=="one_d_heisenberg"
    (E, ψ, H) = compute_1d_heisenberg(N; conserve_qns=conserve_qns, nsweeps=nsweeps, maxdim = maxdim, cutoff=cutoff, noise=noise)
  else
    error("$(model) is not a valid model name. Please refer to documentation.")
  end

  return ψ, H
end

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
  dir = (i.dir == 1 ? "Out" : i.dir == -1 ? "In" : "Neither")
  println("(dim=$(dim(i))|id=$(id(i)%1000)|$(tags(i)))")
end

function representative_contract_timing(ψ, H; N = nothing, nrepeat=10, twosite=nothing, LHS=nothing)
  timer = TimerOutput()
  if isnothing(twosite)
    N = (isnothing(N) ? Int(length(ψ) / 2) : N)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    orthogonalize!(ψ, N)
    println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    twosite = ψ[N] * ψ[N + 1]
    println("Dimensions of two-site tensor")
  else
    println("Using the provided two-site tensor")
  end
  easyprint(twosite)
  println()

  ## contract to form LHS
  if isnothing(LHS)
    println("Forming the LHS tensor to contract with two-site tensor")
    @timeit timer "Construct LHS" begin
      LHS = dag(ψ'[1]) * H[1] * (ψ[1])
      for i in 2:(N-1)
        LHS = (LHS) * (dag(ψ'[i]) * H[i] * ψ[i])
      end
    end
  else
    println("Using the provided LHS tensor")
  end

  println("Dimension of LHS tensor")
  easyprint(LHS)
  println()

  println("Starting the timer for contracting LHS with two-site tensor")

  P = nothing
  for i in 1:nrepeat
    @timeit timer "LHS with two-site"  P = LHS * twosite
  end

  println("Timing for the contractions:\n$(timer)")
  return twosite, LHS, P;
end

function representative_svd_timing(ψ, H; N = nothing, nrepeat=10, twosite=nothing, LHS=nothing)
  timer = TimerOutput()
  if isnothing(twosite)
    N = (isnothing(N) ? Int(length(ψ) / 2) : N)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    orthogonalize!(ψ, N)
    println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    twosite = ψ[N] * ψ[N + 1]
    println("Dimensions of two-site tensor")
  else
    println("Using the provided two-site tensor")
  end
  easyprint(twosite)
  println()

  println("Starting the timer for contracting LHS with two-site tensor")

  P = nothing
  for i in 1:nrepeat
    @timeit timer "LHS with two-site"  svd
  end

  println("Timing for the contractions:\n$(timer)")
  return twosite, LHS, P;
end