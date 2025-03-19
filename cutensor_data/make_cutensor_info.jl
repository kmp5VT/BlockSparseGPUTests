#include("$(@__DIR__)/../src/BlockSparseGPUTests.jl")
using BlockSparseGPUTests
using BlockSparseGPUTests: Model, TwoDHubbSmall, TwoDHubbMed, TwoDHubbLarge
using ITensors, ITensorMPS, JLD2

function block_extents(ind::Index)
    return ntuple(i -> dim(ind.space[i]), nblocks(ind))
end

path="$(@__DIR__)/../runnable_examples/saved_tensors_jld2/"
folder_names = ["symm_kysznf/","symm_nf/", "symm_nfparity/", "symm_sznf/"]
systems = ["E1","E2", "S1","S2","S3"]

#folder_names = ["symm_nfparity/"]
# systems = ["S3"]

for fname in folder_names
  for sys in systems
    d = jldopen(path*"2d_momentum_hubbard/medium/" * fname * sys * ".jld")
    T1=d["T1"]
    T2=d["T2"]
    T1 = BlockSparseGPUTests.replace_ITensor_data_with_random(T1)
    T2 = BlockSparseGPUTests.replace_ITensor_data_with_random(T2)
    combo = combiner(commoninds(T1, T2))
    T1M = T1 * combo 
    T2M = dag(combo) * T2 
    T3 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1) * BlockSparseGPUTests.replace_ITensor_data_with_ones(T2)
    T3M = T1M * T2M

    labelsC, labelsA, labelsB = ITensors.compute_contraction_labels(Tuple(noncommoninds(T1,T2)), inds(T1), inds(T2))
    min_label = minimum(labelsA)
    labelsA = [labelsA...] .- min_label .+ 1
    labelsB = [labelsB...] .- min_label .+ 1
    labelsC = [labelsC...] .- min_label .+ 1

    data_path = mkdir("$(@__DIR__)/"*fname*sys*"/")
    open(data_path*"A_labels.txt", "w") do file
      for x in labelsA
        println(file, x)
      end
    end
    open(data_path*"B_labels.txt", "w") do file
      for x in labelsB
        println(file, x)
      end
    end
    open(data_path*"C_labels.txt", "w") do file
      for x in labelsC
        println(file, x)
      end
    end

    for is in 1:length(inds(T1))
        idx = ind(T1, is)
        open(data_path*"A_block_extent_index_$(is).txt", "w") do file
            for x in block_extents(idx)
                println(file, x)
            end
        end
    end

    for is in 1:length(inds(T2))
        idx = ind(T2, is)
        open(data_path*"B_block_extent_index_$(is).txt", "w") do file
            for x in block_extents(idx)
                println(file, x)
            end
        end
    end

    nzA = [Int64.(x.data) for x in nzblocks(T1)]
    nzA = [(nzA...)...]
    nzB = [Int64.(x.data) for x in nzblocks(T2)]
    nzB = [(nzB...)...]
    nzC = [Int64.(x.data) for x in nzblocks(T3)]
    nzC = [(nzC...)...]
    open(data_path*"A_nz_blocks.txt", "w") do file
      for x in nzA
        println(file, x)
      end
    end
    open(data_path*"B_nz_blocks.txt", "w") do file
      for x in nzB
        println(file, x)
      end
    end
    open(data_path*"C_nz_blocks.txt", "w") do file
      for x in nzC
        println(file, x)
      end
    end

    open(data_path*"C_data.txt","w") do file
      d3 = NDTensors.data(T3)
      for x in 1:(length(d3)-1)
        print(file, "$(d3[x]),")
      end
      print(file, d3[end])
    end
  end
end

## This is the combined version of the code.
for fname in folder_names
  for sys in systems
    d = jldopen(path*"2d_momentum_hubbard/medium/" * fname * sys * ".jld")
    T1=d["T1"]
    T2=d["T2"]
    T1 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1)
    T2 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T2)
    cominds = commoninds(T1,T2)
    combo_internal = combiner(cominds)
    T1 = T1 * combo_internal 
    T2 = dag(combo_internal) * T2
    T3 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1) * BlockSparseGPUTests.replace_ITensor_data_with_ones(T2)
    

    labelsC, labelsA, labelsB = ITensors.compute_contraction_labels(Tuple(noncommoninds(T1,T2)), inds(T1), inds(T2))
    min_label = minimum(labelsA)
    labelsA = [labelsA...] .- min_label .+ 1
    labelsB = [labelsB...] .- min_label .+ 1
    labelsC = [labelsC...] .- min_label .+ 1

    data_path = mkdir("$(@__DIR__)/block_permuted_internal_only/"*fname*sys*"/")
    open(data_path*"A_labels.txt", "w") do file
      for x in labelsA
        println(file, x)
      end
    end
    open(data_path*"B_labels.txt", "w") do file
      for x in labelsB
        println(file, x)
      end
    end
    open(data_path*"C_labels.txt", "w") do file
      for x in labelsC
        println(file, x)
      end
    end

    for is in 1:length(inds(T1))
        idx = ind(T1, is)
        open(data_path*"A_block_extent_index_$(is).txt", "w") do file
            for x in block_extents(idx)
                println(file, x)
            end
        end
    end

    for is in 1:length(inds(T2))
        idx = ind(T2, is)
        open(data_path*"B_block_extent_index_$(is).txt", "w") do file
            for x in block_extents(idx)
                println(file, x)
            end
        end
    end

    nzA = [Int64.(x.data) for x in nzblocks(T1)]
    nzA = [(nzA...)...]
    nzB = [Int64.(x.data) for x in nzblocks(T2)]
    nzB = [(nzB...)...]
    nzC = [Int64.(x.data) for x in nzblocks(T3)]
    nzC = [(nzC...)...]
    open(data_path*"A_nz_blocks.txt", "w") do file
      for x in nzA
        println(file, x)
      end
    end
    open(data_path*"B_nz_blocks.txt", "w") do file
      for x in nzB
        println(file, x)
      end
    end
    open(data_path*"C_nz_blocks.txt", "w") do file
      for x in nzC
        println(file, x)
      end
    end

    # open(data_path*"C_data.txt","w") do file
    #   d3 = NDTensors.data(T3)
    #   for x in 1:(length(d3)-1)
    #     print(file, "$(d3[x]),")
    #   end
    #   print(file, d3[end])
    # end
  end
end

for fname in folder_names
  for sys in systems
  d = jldopen(path*"2d_momentum_hubbard/medium/" * fname * sys * ".jld")
  T1=d["T1"]
  T2=d["T2"]
  T1 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1)
  T2 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T2)
  cominds = commoninds(T1,T2)
  combo_internal = combiner(cominds)
  combo_external_T1 = combiner(noncommoninds(inds(T1), cominds))
  combo_external_T2 = combiner(noncommoninds(inds(T2), cominds))
  T1 = combo_external_T1 * T1 * combo_internal 
  T2 = dag(combo_internal) * T2 * combo_external_T2
  T3 = BlockSparseGPUTests.replace_ITensor_data_with_ones(T1) * BlockSparseGPUTests.replace_ITensor_data_with_ones(T2)
  

  labelsC, labelsA, labelsB = ITensors.compute_contraction_labels(Tuple(noncommoninds(T1,T2)), inds(T1), inds(T2))
  min_label = minimum(labelsA)
  labelsA = [labelsA...] .- min_label .+ 1
  labelsB = [labelsB...] .- min_label .+ 1
  labelsC = [labelsC...] .- min_label .+ 1

  data_path = mkdir("$(@__DIR__)/block_permuted_matrices/"*fname*sys*"/")
  open(data_path*"A_labels.txt", "w") do file
    for x in labelsA
      println(file, x)
    end
  end
  open(data_path*"B_labels.txt", "w") do file
    for x in labelsB
      println(file, x)
    end
  end
  open(data_path*"C_labels.txt", "w") do file
    for x in labelsC
      println(file, x)
    end
  end

  for is in 1:length(inds(T1))
      idx = ind(T1, is)
      open(data_path*"A_block_extent_index_$(is).txt", "w") do file
          for x in block_extents(idx)
              println(file, x)
          end
      end
  end

  for is in 1:length(inds(T2))
      idx = ind(T2, is)
      open(data_path*"B_block_extent_index_$(is).txt", "w") do file
          for x in block_extents(idx)
              println(file, x)
          end
      end
  end

  nzA = [Int64.(x.data) for x in nzblocks(T1)]
  nzA = [(nzA...)...]
  nzB = [Int64.(x.data) for x in nzblocks(T2)]
  nzB = [(nzB...)...]
  nzC = [Int64.(x.data) for x in nzblocks(T3)]
  nzC = [(nzC...)...]
  open(data_path*"A_nz_blocks.txt", "w") do file
    for x in nzA
      println(file, x)
    end
  end
  open(data_path*"B_nz_blocks.txt", "w") do file
    for x in nzB
      println(file, x)
    end
  end
  open(data_path*"C_nz_blocks.txt", "w") do file
    for x in nzC
      println(file, x)
    end
  end

  # open(data_path*"C_data.txt","w") do file
  #   d3 = NDTensors.data(T3)
  #   for x in 1:(length(d3)-1)
  #     print(file, "$(d3[x]),")
  #   end
  #   print(file, d3[end])
  # end
end
end