include("../src/BlockSparseGPUTests.jl")
using HDF5, ITensors, TimerOutputs
using .BlockSparseGPUTests

function run_timings(foldername)
    timer = TimerOutput()
    tensor_networks = ["EL1", "EL2", "S1", "S2", "S3"]
    for filename in tensor_networks
        fid = h5open("$(foldername)/$(filename).h5")
        T1 = read(fid, "T1", ITensor)
        T2 = read(fid, "T2", ITensor)
        close(fid)

        BlockSparseGPUTests.timing_contract(T1, T2; nrepeat=1000, timer=timer, timer_string="$(foldername)/$(filename)")
    end

    @show timer
end

run_timings("hdf5/dense_small")
run_timings("hdf5/sparse_small")

