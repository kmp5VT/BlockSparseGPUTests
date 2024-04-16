using ITensors.ITensorMPS
function left_environment_tensor(ψ, H; N = nothing)
  N = isnothing(N) ? trunc(Int,length(ψ) / 2) : N
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, N)
  ITensorMPS.makeL!(PH, ψ, N)

  return PH.LR
end

function right_environment_tensor(ψ, H; N = nothing)
  N = isnothing(N) ? trunc(Int, length(ψ) / 2) : N
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, N)
  ITensorMPS.makeR!(PH, ψ, N)

  return PH.LR
end

function two_site_tensor(ψ; N = nothing)
   N = isnothing(N) ? trunc(Int, length(ψ) / 2) : N
   return ψ[N] * ψ[N + 1] 
end