module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims
const SliceIndex = Union{RangeIndex,Colon}

using LinearAlgebra

using TupleTools
using TupleTools: StaticLength

export StridedView, @strided, @unsafe_strided, sreshape, sview

const _NTHREADS = Ref(1)
_nthreads() = _NTHREADS[]

function set_num_threads(n::Int)
    N = Base.Threads.nthreads()
    if n > N
        n = N
        _set_num_threads_warn(n)
    end
    _NTHREADS[] = n
end
@noinline function _set_num_threads_warn(n)
    @warn "Maximal number of threads limited by number of Julia threads,
            setting number of threads equal to Threads.nthreads() = $n"
end

enable_threads() = set_num_threads(Base.Threads.nthreads())
disable_threads() = set_num_threads(1)

const _use_threaded_mul = Ref(false)
use_threaded_mul() = _use_threaded_mul[]

function disable_threaded_mul()
    _use_threaded_mul[] = false
    return
end

function enable_threaded_mul()
    _use_threaded_mul[] = true
    return
end

function __init__()
    set_num_threads(Base.Threads.nthreads())
end

include("abstractstridedview.jl")
include("stridedview.jl")
include("unsafestridedview.jl")
include("mapreduce.jl")
include("broadcast.jl")
include("macros.jl")

# function _precompile_()
#     ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
#     for T in (Float64, ComplexF64)
#         USMatrix = UnsafeStridedView{T, 2, T, typeof(Base.identity)}
#         SMatrix = StridedView{T, 2, Vector{T}, typeof(Base.identity)}
#
#         precompile(Tuple{typeof(mul!), USMatrix, USMatrix, USMatrix})
#         precompile(Tuple{typeof(mul!), USMatrix, USMatrix, USMatrix, T, T})
#
#         precompile(Tuple{typeof(mul!), SMatrix, SMatrix, SMatrix})
#         precompile(Tuple{typeof(mul!), SMatrix, SMatrix, SMatrix, T, T})
#
#         if T <: Complex
#             CUSMatrix = UnsafeStridedView{T, 2, T, typeof(Base.conj)}
#             CSMatrix = StridedView{T, 2, Vector{T}, typeof(Base.conj)}
#
#             precompile(Tuple{typeof(mul!), USMatrix, USMatrix, CUSMatrix})
#             precompile(Tuple{typeof(mul!), USMatrix, CUSMatrix, USMatrix})
#             precompile(Tuple{typeof(mul!), USMatrix, CUSMatrix, CUSMatrix})
#
#             precompile(Tuple{typeof(mul!), SMatrix, SMatrix, CSMatrix})
#             precompile(Tuple{typeof(mul!), SMatrix, CSMatrix, SMatrix})
#             precompile(Tuple{typeof(mul!), SMatrix, CSMatrix, CSMatrix})
#         end
#
#         for N = 1:4
#             USArray = UnsafeStridedView{T, N, T, typeof(Base.identity)}
#             SArray = StridedView{T, N, Vector{T}, typeof(Base.identity)}
#
#             for ASArray in (USArray, SArray)
#                 precompile(Tuple{typeof(copyto!), ASArray, ASArray})
#                 precompile(Tuple{typeof(mul!), ASArray, T, ASArray})
#                 precompile(Tuple{typeof(mul!), ASArray, ASArray, T})
#                 precompile(Tuple{typeof(rmul!), ASArray, T})
#                 precompile(Tuple{typeof(lmul!), T, ASArray})
#                 precompile(Tuple{typeof(axpy!), T, ASArray, ASArray})
#                 precompile(Tuple{typeof(axpby!), T, ASArray, T, ASArray})
#             end
#
#             if T <: Complex
#                 CUSArray = UnsafeStridedView{T, N, T, typeof(Base.conj)}
#                 CSArray = StridedView{T, N, Vector{T}, typeof(Base.conj)}
#
#                 for (ASArray, CASArray) in ((USArray, CUSArray), (SArray, CSArray))
#                     precompile(Tuple{typeof(copyto!), ASArray, CASArray})
#                     precompile(Tuple{typeof(mul!), ASArray, T, CASArray})
#                     precompile(Tuple{typeof(mul!), ASArray, CASArray, T})
#                     precompile(Tuple{typeof(rmul!), CASArray, T})
#                     precompile(Tuple{typeof(lmul!), T, CASArray})
#                     precompile(Tuple{typeof(axpy!), T, CASArray, ASArray})
#                     precompile(Tuple{typeof(axpby!), T, CASArray, T, ASArray})
#                 end
#             end
#         end
#     end
# end
# _precompile_()

end
