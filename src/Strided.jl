module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims

using TupleTools
using TupleTools: StaticLength

export StridedView, sview

using Compat

@static if VERSION < v"0.7-"
    const adjoint = Base.ctranspose
    const adjoint! = Base.ctranspose!

    import Base.LinAlg: scale!, axpy!
    """
        axpby!(a, X, b, Y)

    Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
    """
    function axpby! end

    const LinearAlgebra = Base.LinAlg
else
    using LinearAlgebra
    import LinearAlgebra: adjoint, adjoint!, axpy!, axpby!, mul!
end

@static if VERSION < v"0.7.0-DEV.3155"
    const popfirst! = shift!
end

function __init__()
    LinearAlgebra.BLAS.set_num_threads(1)
    Threads.nthreads() == 1 && warn("Strided disables BLAS multithreading, enable Julia threading (`export JULIA_NUM_THREADS = N`) to benefit from multithreaded matrix multiplication and more")
end

# for use in combination with treading
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


include("stridedview.jl")
include("sview.jl")
include("map.jl")

end
