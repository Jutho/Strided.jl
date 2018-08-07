module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims

using LinearAlgebra

using TupleTools
using TupleTools: StaticLength

export StridedView, @strided

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

include("stridedview.jl")
include("mapreduce.jl")
include("broadcast.jl")

macro strided(ex)
    _strided(ex)
end

function _strided(ex::Expr)
    if ex.head == :call && ex.args[1] isa Symbol
        if ex.args[1] == :reshape
            return Expr(:call, :sreshape, _strided.(ex.args[2:end])...)
        else
            return Expr(:call, esc(ex.args[1]), _strided.(ex.args[2:end])...)
        end
    elseif ex.head == :(=) && ex.args[1] isa Symbol
        return Expr(:(=), esc(ex.args[1]), Expr(:call, :maybeunstrided, _strided(ex.args[2])))
    else
        return Expr(ex.head, _strided.(ex.args)...)
    end
end
const exclusionlist = Symbol[:(:)]
_strided(ex::Symbol) =  ex in exclusionlist ? esc(ex) : :(maybestrided($(esc(ex))))
_strided(ex) = ex

maybestrided(A::DenseArray) = StridedView(A)
maybestrided(A) = A
maybeunstrided(A::StridedView) = parent(A)
maybeunstrided(A) = A


end
