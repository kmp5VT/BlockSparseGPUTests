using ITensors.ITensorMPS

function left_environment_tensor(ψ, H; j=length(ψ) ÷ 2)
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, j)
  ITensorMPS.makeL!(PH, ψ, j)
  return PH.LR
end

function right_environment_tensor(ψ, H; j=length(ψ) ÷ 2)
  PH = ProjMPO(H)
  orthogonalize!(PH, ψ, j)
  ITensorMPS.makeR!(PH, ψ, j)
  return PH.LR
end

function two_site_tensor(ψ; j=length(ψ) ÷ 2)
  return ψ[j] * ψ[j + 1]
end
