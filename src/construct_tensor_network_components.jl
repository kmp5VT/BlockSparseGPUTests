using ITensors.ITensorMPS
function make_left_environment_tensor(ψ, H; N = nothing)
  N = isnothing(N) ? trunc(Int,length(ψ) / 2) : N
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, N)
  ITensorMPS.makeL!(PH, ψ, N)

  return PH.LR[N-1]
end

function make_right_environment_tensor(ψ, H; N = nothing)
  N = isnothing(N) ? trunc(Int, length(ψ) / 2) : N
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, N)
  ITensorMPS.makeR!(PH, ψ, N)

  return PH.LR[N + 2]
end

function make_two_site_tensor(ψ; N = nothing)
   N = isnothing(N) ? trunc(Int, length(ψ) / 2) : N
   return ψ[N] * ψ[N + 1] 
end