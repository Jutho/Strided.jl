const FN = typeof(identity)
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)
_methodconj(::FN) = conj
_methodconj(::FC) = identity
_methodconj(::FA) = transpose
_methodconj(::FT) = adjoint

abstract type AbstractStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N} end

Base.elsize(::Type{<:AbstractStridedView{T}}) where {T} = Base.isbitstype(T) ? sizeof(T) : (Base.isbitsunion(T) ? Base.bitsunionsize(T) : sizeof(Ptr))

# Converting back to other DenseArray type:
function Base.convert(T::Type{<:DenseArray}, a::AbstractStridedView)
    b = T(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
function Base.convert(::Type{Array}, a::AbstractStridedView{T}) where {T}
    b = Array{T}(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end

# Methods based on map!
Base.copyto!(dst::AbstractStridedView{<:Any,N}, src::AbstractStridedView{<:Any,N}) where {N} =
    map!(identity, dst, src)
Base.conj!(a::AbstractStridedView{<:Real}) = a
Base.conj!(a::AbstractStridedView) = map!(conj, a, a)
LinearAlgebra.adjoint!(dst::AbstractStridedView{<:Any,N}, src::AbstractStridedView{<:Any,N}) where {N} =
    copyto!(dst, adjoint(src))
Base.permutedims!(dst::AbstractStridedView{<:Any,N}, src::AbstractStridedView{<:Any,N}, p) where {N} =
    copyto!(dst, permutedims(src, p))

# linear algebra
LinearAlgebra.rmul!(dst::AbstractStridedView, α::Number) = mul!(dst, dst, α)
LinearAlgebra.lmul!(α::Number, dst::AbstractStridedView) = mul!(dst, α, dst)

LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, α::Number, src::AbstractStridedView{<:Number,N}) where {N} =
    α == 1 ? copyto!(dst, src) : map!(x->α*x, dst, src)
LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, src::AbstractStridedView{<:Number,N}, α::Number) where {N} =
    α == 1 ? copyto!(dst, src) : map!(x->x*α, dst, src)
LinearAlgebra.axpy!(a::Number, X::AbstractStridedView{<:Number,N}, Y::AbstractStridedView{<:Number,N}) where {N} =
    a == 1 ? map!(+, Y, X, Y) : map!((x,y)->(a*x+y), Y, X, Y)
LinearAlgebra.axpby!(a::Number, X::AbstractStridedView{<:Number,N}, b::Number, Y::AbstractStridedView{<:Number,N}) where {N} =
    b == 1 ? axpy!(a, X, Y) : map!((x,y)->(a*x+b*y), Y, X, Y)

function LinearAlgebra.mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2}, B::AbstractStridedView{<:Any,2})
    if C.op == conj
        if stride(C,1) < stride(C,2)
            mul!(conj(C), conj(A), conj(B))
        else
            mul!(C', B', A')
        end
    elseif stride(C,1) > stride(C,2)
        mul!(transpose(C), transpose(B), transpose(A))
    else
        _mul!(C, A, B)
        # EXPERIMENTAL: only use in combination with BLAS.set_num_threads(1)
        # if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(size(C)) < Threads.nthreads()*1024
        #     _mul!(C, A, B)
        # else
        #     mranges, nranges = _computethreadedmulblocks(size(C,1), size(C,2))
        #     @inbounds Threads.@threads for i = 1:Threads.nthreads()
        #         _mul!(C[mranges[i], nranges[i]], A[mranges[i], :], B[ :, nranges[i]])
        #     end
        # end
    end
    return C
end

_mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2}, B::AbstractStridedView{<:Any,2}) = __mul!(C, A, B)
function __mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2}, B::AbstractStridedView{<:Any,2})
    if stride(A,1) < stride(A,2) && stride(B,1) < stride(B,2)
        LinearAlgebra.generic_matmatmul!(C,'N','N',A,B)
    elseif stride(A,1) < stride(A,2)
        LinearAlgebra.generic_matmatmul!(C,'N','T',A,transpose(B))
    elseif stride(B,1) < stride(B,2)
        LinearAlgebra.generic_matmatmul!(C,'T','N',transpose(A),B)
    else
        LinearAlgebra.generic_matmatmul!(C,'T','T',transpose(A),transpose(B))
    end
    return C
end
function _mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2}, B::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if !(any(isequal(1), strides(A)) && any(isequal(1), strides(B)) && any(isequal(1), strides(C)))
        return __mul!(C,A,B)
    end
    if A.op == identity
        if stride(A,1) == 1
            A2 = A
            cA = 'N'
        else
            A2 = transpose(A)
            cA = 'T'
        end
    else
        if stride(A,1) != 1
            A2 = A'
            cA = 'C'
        else
            return LinearAlgebra.generic_matmatmul!(C,'N','N',A,B)
        end
    end
    if B.op == identity
        if stride(B,1) == 1
            B2 = B
            cB = 'N'
        else
            B2 = transpose(B)
            cB = 'T'
        end
    else
        if stride(B,1) != 1
            B2 = B'
            cB = 'C'
        else
            return LinearAlgebra.generic_matmatmul!(C,'N','N',A,B)
        end
    end
    LinearAlgebra.BLAS.gemm!(cA,cB,one(T),A2,B2,zero(T),C)
end

# ParentIndex: index directly into parent array
struct ParentIndex
    i::Int
end

function sreshape end
struct ReshapeException <: Exception
end
Base.show(io::IO, e::ReshapeException) = print(io, "Cannot produce a reshaped StridedView without allocating, try sreshape(copy(array), newsize) or fall back to reshape(array, newsize)")

# Auxiliary routines
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} = (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

_defaultstrides(sz::Tuple{}, s = 1) = ()
_defaultstrides(sz::Dims, s = 1) = (s, _defaultstrides(tail(sz), s*sz[1])...)

@inline function _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
_computeviewsize(::Tuple{}, ::Tuple{}) = ()

@inline function _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Int)
        return _computeviewstrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else
        return (oldstrides[1]*step(I[1]), _computeviewstrides(tail(oldstrides), tail(I))...)
    end
end
_computeviewstrides(::Tuple{}, ::Tuple{}) = ()

@inline function _computeviewoffset(strides::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Colon)
        return _computeviewoffset(tail(strides), tail(I))
    else
        return (first(I[1])-1)*strides[1]+_computeviewoffset(tail(strides), tail(I))
    end
end
_computeviewoffset(::Tuple{}, ::Tuple{}) = 0

_computereshapestrides(newsize::Tuple{}, oldsize::Tuple{}, strides::Tuple{}) = ()
function _computereshapestrides(newsize::Tuple{}, oldsize::Dims{N}, strides::Dims{N}) where {N}
    all(isequal(1), oldsize) || throw(DimensionMismatch())
    return ()
end
function _computereshapestrides(newsize::Dims, oldsize::Tuple{}, strides::Tuple{})
    all(isequal(1), newsize)
    return map(n->1, newsize)
end
function _computereshapestrides(newsize::Dims{1}, oldsize::Dims{1}, strides::Dims{1})
    newsize[1] == oldsize[1] || throw(DimensionMismatch())
    return (strides[1],)
end
function _computereshapestrides(newsize::Dims, oldsize::Dims{1}, strides::Dims{1})
    newsize[1] == 1 && return (strides[1], _computereshapestrides(tail(newsize), oldsize, strides)...)

    if newsize[1] <= oldsize[1]
        d,r = divrem(oldsize[1], newsize[1])
        r == 0 || throw(ReshapeException())

        return (strides[1], _computereshapestrides(tail(newsize), (d,), (newsize[1]*strides[1],))...)
    else
        throw(DimensionMismatch())
    end
end
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, strides::Dims{N}) where {N}
    newsize[1] == 1 && return (strides[1], _computereshapestrides(tail(newsize), oldsize, strides)...)
    oldsize[1] == 1 && return _computereshapestrides(newsize, tail(oldsize), tail(strides))

    d,r = divrem(oldsize[1], newsize[1])
    if r == 0
        return (strides[1], _computereshapestrides(tail(newsize), (d, tail(oldsize)...), (newsize[1]*strides[1], tail(strides)...))...)
    else
        if oldsize[1]*strides[1] == strides[2]
            return _computereshapestrides(newsize, (oldsize[1]*oldsize[2], TupleTools.tail2(oldsize)...), (strides[1], TupleTools.tail2(strides)...))
        else
            throw(ReshapeException())
        end
    end
end

function _computethreadedmulblocks end
let
    mranges = Vector{UnitRange{Int}}(undef, Threads.nthreads())
    nranges = Vector{UnitRange{Int}}(undef, Threads.nthreads())
    factors = simpleprimefactorization(Threads.nthreads())
    global _computethreadedmulblocks
    @inbounds function _computethreadedmulblocks(m,n)
        divm = 1
        divn = 1
        for i = length(factors):-1:1
            if m*divn > n*divm
                divm *= factors[i]
            else
                divn *= factors[i]
            end
        end
        bm, rm = divrem(m, divm)
        bn, rn = divrem(n, divn)
        moffset = 0
        noffset = 0

        for i = 1:divm
            mnew = moffset + bm + ifelse(rm >= i, 1, 0)
            for j = 1:divn
                mranges[i+(j-1)*divm] = (moffset+1):(mnew)
            end
            moffset = mnew
        end
        for j = 1:divn
            nnew = noffset + bn + ifelse(rn >= j, 1, 0)
            for i = 1:divm
                nranges[i+(j-1)*divm] = (noffset+1):(nnew)
            end
            noffset = nnew
        end
        return mranges, nranges
    end
end
