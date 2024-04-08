using ITensors, NDTensors, TimerOutputs

function representative_contract_timing(
  ψ, H; N=nothing, nrepeat=10, twosite=nothing, LHS=nothing, time=true, verbose = false
)
  timer = TimerOutput()
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
    #@timeit timer "Construct LHS" begin
      LHS = dag(ψ'[1]) * H[1] * (ψ[1])
      for i in 2:(N - 1)
        @show i
        temp = dag(ψ'[i]) * H[i] * (ψ[i])
        LHS = LHS * temp
      end
    #end
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

  P = nothing
  if time
    ## benchtools maybe to force contraction
    for i in 1:nrepeat
      @timeit timer "LHS with two-site" P = LHS * twosite
    end
    println("Timing for the contractions:\n$(timer)")
  end

  return twosite, LHS, P
end

function representative_svd_timing(
  ψ, H; N=nothing, nrepeat=10, twosite=nothing, LHS=nothing
)
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
    @timeit timer "LHS with two-site" svd(twosite, (ind(twosite, 1), ind(twosite, 2)))
  end

  println("Timing for the contractions:\n$(timer)")
  return twosite, LHS, P
end
