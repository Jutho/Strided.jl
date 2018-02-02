module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex

using TupleTools
using TupleTools: StaticLength

export StridedView, StridedIterator, BlockedIterator, splitdims, sview

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

include("stridedview.jl")
include("sview.jl")
include("map.jl")

end
