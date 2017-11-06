module Strided

# ParentIndex: index directly in parent array of StridedView
import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex

using TupleTools
using TupleTools: StaticLength

export StridedView, StridedIterator, BlockedIterator, splitdims, sview

@static if !isdefined(Base, :EqualTo)
    struct EqualTo{T} <: Function
        x::T
        EqualTo(x::T) where {T} = new{T}(x)
    end
    (f::EqualTo)(y) = isequal(f.x, y)
    const equalto = EqualTo
    export equalto
end

@static if !isdefined(Base, :adjoint)
    const adjoint = Base.ctranspose
    const adjoint! = Base.ctranspose!
    export adjoint, adjoint!
else
    import Base: adjoint, adjoint!
end

import Base.LinAlg.axpy!
@static if !isdefined(Base.LinAlg, :axpby!)
    """
        axpby!(a, X, b, Y)

    Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
    """
    function axpby! end
else
    import Base.LinAlg.axpby!
end

export axpby!, axpy!


include("stridedview.jl")
include("sview.jl")
include("map.jl")

end
