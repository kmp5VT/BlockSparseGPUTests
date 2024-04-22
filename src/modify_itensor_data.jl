using FillArrays
function remove_data_from_ITensor(T::ITensor)
  store = storage(T)
  nelems = length(store)
  elt = eltype(T)
  datat = NDTensors.datatype(store)
  lazyzero = NDTensors.UnallocatedArrays.UnallocatedZeros{elt}(Zeros(nelems), datat)
  lazyzerostorage = NDTensors.setdata(store, lazyzero)
  return NDTensors.setstorage(T, lazyzerostorage)
end

function replace_ITensor_data_with_random(T::ITensor)
  store = storage(T)
  nelems = length(store)
  datatype = NDTensors.TypeParameterAccessors.parenttype(similar(ITensors.data(T)))
  randomdata = NDTensors.generic_randn(datatype, nelems)
  randomstorage = NDTensors.setdata(store, randomdata)
  return NDTensors.setstorage(T, randomstorage)
end

function replace_ITensor_data_with_random(T::ITensor, datatype::Type{<:AbstractArray})
  store = storage(T)
  nelems = length(store)
  randomdata = NDTensors.generic_randn(datatype, nelems)
  randomstorage = NDTensors.setdata(store, randomdata)
  return NDTensors.setstorage(T, randomstorage)
end
