include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using HDF5, ITensors, TimerOutputs
using .BlockSparseGPUTests

function run_timings(prefix; size="small", type="dense", adaptor=identity)
  timer = TimerOutput()
  foldername = "$prefix/$size/$type"
  tensor_networks = ["EL1", "EL2", "S1", "S2", "S3"]
  for filename in tensor_networks
    fid = h5open("$(foldername)/$(filename).h5")
    T1 = read(fid, "T1", ITensor)
    T2 = read(fid, "T2", ITensor)
    close(fid)

    ## TODO update and use benchmarktools here
    BlockSparseGPUTests.timing_contract(
      adaptor(T1),
      adaptor(T2);
      nrepeat=1000,
      timer=timer,
      timer_string="$size,$type,$filename",
    )
  end

  @show timer
end

run_timings("$(@__DIR__)/hdf5"; size="small", type="dense")
run_timings("$(@__DIR__)/hdf5"; size="small", type="sparse")
