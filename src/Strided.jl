module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims

using LinearAlgebra

using TupleTools
using TupleTools: StaticLength

export StridedView, @strided, @unsafe_strided, sreshape

# function __init__()
#     LinearAlgebra.BLAS.set_num_threads(1)
#     Threads.nthreads() == 1 && warn("Strided disables BLAS multithreading, enable Julia threading (`export JULIA_NUM_THREADS = N`) to benefit from multithreaded matrix multiplication and more")
# end

# used to factor the number of threads
function simpleprimefactorization(n::Int)
    factors = Vector{Int}()
    k = 2
    while k <= n
        d, r = divrem(n, k)
        if r == 0
            push!(factors, k)
            n = d
        else
            k += 1
        end
    end
    return factors
end

include("abstractstridedview.jl")
include("stridedview.jl")
include("unsafestridedview.jl")
include("mapreduce.jl")
include("broadcast.jl")
include("macros.jl")

end
