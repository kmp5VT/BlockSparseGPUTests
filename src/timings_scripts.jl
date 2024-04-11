using ITensors, NDTensors, TimerOutputs
using ITensors.ITensorMPS
# using BenchmarkTools: @btime

function representative_contract_timing(
  ψ,
  H;
  N=nothing,
  nrepeat=10,
  twosite=nothing,
  LHS=nothing,
  verbose=false,
  timer=TimerOutput(),
  timer_string::String="LHS with two-site",
)
  if isnothing(twosite)
    N = (isnothing(N) ? Int(length(ψ) / 2) : N)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    if verbose
      println("Orthogonalizing ψ to site $(N)")
    end
    orthogonalize!(ψ, N)
    if verbose
      println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    end
    twosite = ψ[N] * ψ[N + 1]
    if verbose
      println("Dimensions of two-site tensor")
    end
  else
    if verbose
      println("Using the provided two-site tensor")
    end
  end
  if verbose
    easyprint(twosite)
    println()
  end

  ## contract to form LHS
  if isnothing(LHS)
    if verbose
      println("Forming the LHS tensor to contract with two-site tensor")
    end
    LHS = dag(ψ'[1]) * H[1] * (ψ[1])
    for i in 2:(N - 1)
      LHS = ((LHS * dag(ψ'[i])) * H[i]) * (ψ[i])
    end
  else
    if verbose
      println("Using the provided LHS tensor")
    end
  end

  if verbose
    println("Dimension of LHS tensor")
    easyprint(LHS)
    println()
    println("Starting the timer for contracting LHS with two-site tensor")
  end

  ## benchtools maybe to force contraction
  for i in 1:nrepeat
    @timeit timer timer_string begin
      LHS * twosite
    end
  end
  if verbose
    println("Timing for the contractions:\n$(timer)")
  end

  return twosite, LHS
end

function representative_svd_timing(
  ψ,
  H;
  N=nothing,
  nrepeat=10,
  verbose=false,
  twosite=nothing,
  timer=TimerOutput(),
  timer_string="eigsolve",
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  ortho=nothing,
  which_decomp=nothing,
  eigen_perturbation=nothing,
  svd_alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  min_blockdim=nothing,
)
  N = (isnothing(N) ? Int(length(ψ) / 2) : N)
  if isnothing(twosite)
    ## Given a psi grab the middle and middle plus one and contract
    if N == length(ψ)
      error("You cannot choose the last site to run test")
    end

    if verbose
      println("Orthogonalizing ψ to site $(N)")
    end
    orthogonalize!(ψ, N)
    if verbose
      println("Constructing the two-site tensor for site $(N) and $(N + 1)")
    end
    twosite = ψ[N] * ψ[N + 1]
    if verbose
      println("Dimensions of two-site tensor")
    end
  else
    if verbose
      println("Using the provided two-site tensor")
    end
  end
  if verbose
    easyprint(twosite)
    println()
  end

  ## benchtools maybe to force contraction
  indssite = inds(ψ[N])
  ITensors.factorize(
    twosite,
    indssite;
    mindim,
    maxdim,
    cutoff,
    ortho,
    which_decomp,
    eigen_perturbation,
    svd_alg,
    tags=tags(linkind(ψ, N)),
    use_absolute_cutoff,
    use_relative_cutoff,
    min_blockdim,
  )

  for i in 1:nrepeat
    @timeit timer timer_string begin
      ITensors.factorize(
        twosite,
        indssite;
        mindim,
        maxdim,
        cutoff,
        ortho,
        which_decomp,
        eigen_perturbation,
        svd_alg,
        tags=tags(linkind(ψ, N)),
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
      )
    end
  end
  if verbose
    println("Timing for the contractions:\n$(timer)")
  end

  return nothing
end

# indsMb = inds(M[b])
# b = site
# M = MPS
# all else nothing
# L, R, spec = factorize(
#   phi,
#   indsMb;
#   mindim,
#   maxdim,
#   cutoff,
#   ortho,
#   which_decomp,
#   eigen_perturbation,
#   svd_alg,
#   tags=tags(linkind(M, b)),
#   use_absolute_cutoff,
#   use_relative_cutoff,
#   min_blockdim,
# )
