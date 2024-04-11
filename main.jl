using CUDA, Adapt
using ITensors, NDTensors
using TimerOutputs
include("src/BlockSparseGPUTests.jl")

adapt32(ten::ITensor) = adapt(Float32, ten)
## 1D Heisenberg timing
begin
  timer = TimerOutput()
  @timeit timer "Dense" begin
    LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(;
      N=100,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="cpuF64: LHS * 2site",
      nrepeat_contract=1000,
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      adapt32;
      twosite=twosite,
      LHS=LHS,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="cpuF32: LHS * 2site",
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      NDTensors.cu;
      twosite=twosite,
      LHS=LHS,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="cuF64: LHS * 2site",
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      cu;
      twosite=twosite,
      LHS=LHS,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="cuF32: LHS * 2site",
    )
    println("Printing twosite indices")
    BlockSparseGPUTests.easyprint(twosite)
    println("Printing the indices for LHS")
    BlockSparseGPUTests.easyprint(LHS)
    println()
  end

  @timeit timer "BS" begin
    LHS, twosite = BlockSparseGPUTests.test_one_d_heisenberg(;
      N=100,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="cpuF64: LHS * 2site",
      nrepeat_contract=1000,
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      adapt32;
      twosite=twosite,
      nrepeat_contract=1000,
      LHS=LHS,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="cpuF32: LHS * 2site",
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      NDTensors.cu;
      twosite=twosite,
      nrepeat_contract=1000,
      LHS=LHS,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="cuF64: LHS * 2site",
    )
    BlockSparseGPUTests.test_one_d_heisenberg(
      cu;
      twosite=twosite,
      nrepeat_contract=1000,
      LHS=LHS,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="cuF32: LHS * 2site",
    )
    println("Printing twosite indices")
    BlockSparseGPUTests.easyprint(twosite)
    println("Printing the indices for LHS")
    BlockSparseGPUTests.easyprint(LHS)
    println()
  end
  print_timer(timer; sortby=:name, compact=false, allocations=false)
  return nothing
end

## 2D Hubbard timing
begin
  timer = TimerOutput()
  @timeit timer "Dense" begin
    LHSD, twositeD = BlockSparseGPUTests.test_two_d_hubbard(;
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="CPUF64 LHS * 2site",
      nsweeps=6,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      adapt32;
      twosite=twositeD,
      LHS=LHSD,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="CPUF32 LHS * 2site",
      nsweeps=1,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      NDTensors.cu;
      twosite=twositeD,
      LHS=LHSD,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="CUF64 LHS * 2site",
      nsweeps=1,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      cu;
      twosite=twositeD,
      LHS=LHSD,
      nrepeat_contract=1000,
      conserve_qns=false,
      timer=timer,
      timer_string_contract="CUF32 LHS * 2site",
      nsweeps=1,
    )
    println("Printing the dense twosite indices")
    BlockSparseGPUTests.easyprint(twositeD)
    println("Printing the dense indices for LHS")
    BlockSparseGPUTests.easyprint(LHSD)
    println()
  end

  @timeit timer "BS" begin
    LHSS, twositeS = BlockSparseGPUTests.test_two_d_hubbard(;
      nrepeat_contract=1000,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="CPUF64 LHS * 2site",
      nsweeps=6,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      adapt32;
      twosite=twositeS,
      LHS=LHSS,
      nrepeat_contract=1000,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="CPUF32 LHS * 2site",
      nsweeps=1,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      NDTensors.cu;
      twosite=twositeS,
      LHS=LHSS,
      nrepeat_contract=1000,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="CUF64 LHS * 2site",
      nsweeps=1,
    )
    BlockSparseGPUTests.test_two_d_hubbard(
      cu;
      twosite=twositeS,
      LHS=LHSS,
      nrepeat_contract=1000,
      conserve_qns=true,
      timer=timer,
      timer_string_contract="CUF32 LHS * 2site",
      nsweeps=1,
    )
    println("Printing the BS twosite indices")
    BlockSparseGPUTests.easyprint(twositeS)
    println("Printing the BS indices for LHS")
    BlockSparseGPUTests.easyprint(LHSS)
    println()
  end

  print_timer(timer; sortby=:name, compact=false, allocations=false)
  return nothing
end

## 1D Heisenberg eigsolve timing
begin
  timer = TimerOutput()
  ψ, H = BlockSparseGPUTests.construct_psi_h("one_d_heisenberg"; N=100, conserve_qns=false)
  @timeit timer "Dense" begin
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="qr", timer_string="F64 qr"
    )
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="eigen", timer_string="F64 eigen"
    )
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="svd", timer_string="F64 svd"
    )

    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="F32 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="F32 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="F32 svd",
    )

    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="cu F64 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="cu F64 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="cu F64 svd",
    )

    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="cu F32 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="cu F32 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="cu F32 svd",
    )
  end
  ψ, H = BlockSparseGPUTests.construct_psi_h("one_d_heisenberg"; N=100, conserve_qns=true)

  @timeit timer "BS" begin
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="qr", timer_string="F64 qr"
    )
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="eigen", timer_string="F64 eigen"
    )
    BlockSparseGPUTests.representative_svd_timing(
      ψ, H; N=50, nrepeat=1000, timer=timer, which_decomp="svd", timer_string="F64 svd"
    )

    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="F32 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="F32 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      adapt(Float32, ψ),
      adapt(Float32, H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="F32 svd",
    )

    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="cu F64 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="cu F64 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      NDTensors.cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="cu F64 svd",
    )

    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="qr",
      timer_string="cu F32 qr",
    )
    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="eigen",
      timer_string="cu F32 eigen",
    )
    BlockSparseGPUTests.representative_svd_timing(
      cu(ψ),
      NDTensors.cu(H);
      N=50,
      nrepeat=1000,
      timer=timer,
      which_decomp="svd",
      timer_string="cu F32 svd",
    )
  end
  print_timer(timer; sortby=:name, compact=false, allocations=false)
  return nothing
end
print_timer(timer; sortby=:name, compact=false, allocations=false)
