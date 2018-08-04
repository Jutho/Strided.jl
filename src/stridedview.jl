const FN = typeof(identity)
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)

# StridedView
struct StridedView{T,N,A<:DenseArray{T},F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N}
    parent::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    op::F
end

StridedView(a::A, size::NTuple{N,Int}, strides::NTuple{N,Int}, offset::Int) where {T,N,A<:DenseArray{T}} = StridedView{T,N,A,FN}(a, size, strides, offset, identity)
StridedView(a::StridedArray) = StridedView(parent(a), size(a), strides(a), offset(a))

offset(a::DenseArray) = 0
offset(a::SubArray) = Base.first_index(a) - 1
offset(a::Base.ReshapedArray) = 0
offset(a::Base.ReinterpretArray) = 0

# Methods for StridedView
Base.parent(a::StridedView) = a.parent
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
Base.stride(a::StridedView{<:Any, N}, n::Int) where {N} = (n <= N) ? a.strides[n] : a.strides[N]*a.size[N]
offset(a::StridedView) = a.offset
Base.first_index(a::StridedView) = a.offset + 1

Base.dataids(a::StridedView) = Base.dataids(a.parent)

Base.IndexStyle(::Type{<:StridedView}) = Base.IndexCartesian()

# Indexing with N integer arguments
@inline function Base.getindex(a::StridedView{<:Any,N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds r = a.op(a.parent[a.offset+_computeind(I, a.strides)])
    return r
end
@inline function Base.setindex!(a::StridedView{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds a.parent[a.offset+_computeind(I, a.strides)] = a.op(v)
    return a
end

# ParentIndex: index directly into parent array
struct ParentIndex
    i::Int
end

@propagate_inbounds @inline Base.getindex(a::StridedView, I::ParentIndex) = a.op(getindex(a.parent, I.i))
@propagate_inbounds @inline Base.setindex!(a::StridedView, v, I::ParentIndex) = (setindex!(a.parent, a.op(v), I.i); return a)

Base.similar(a::StridedView, ::Type{T}, dims::NTuple{N,Int}) where {N,T}  = StridedView(similar(a.parent, T, dims))
Base.copy(a::StridedView) = copyto!(similar(a), a)

# Specialized methods for `StridedView` which produce views/share data
Base.conj(a::StridedView{<:Real}) = a
Base.conj(a::StridedView{T,N,A,FN}) where {T,N,A} = StridedView{T,N,A,FC}(a.parent, a.size, a.strides, a.offset, conj)
Base.conj(a::StridedView{T,N,A,FC}) where {T,N,A} = StridedView{T,N,A,FN}(a.parent, a.size, a.strides, a.offset, identity)
Base.conj(a::StridedView{T,N,A,FT}) where {T,N,A} = StridedView{T,N,A,FA}(a.parent, a.size, a.strides, a.offset, adjoint)
Base.conj(a::StridedView{T,N,A,FA}) where {T,N,A} = StridedView{T,N,A,FT}(a.parent, a.size, a.strides, a.offset, transpose)

function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
    (length(p) == N && TupleTools.isperm(p)) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = TupleTools._permute(a.size, p)
    newstrides = TupleTools._permute(a.strides, p)
    return StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end

LinearAlgebra.transpose(a::StridedView{<:Any,2}) = permutedims(a, (2,1))
LinearAlgebra.adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a), (2,1))
function LinearAlgebra.adjoint(a::StridedView{<:Any,2}) # act recursively, like Base
    if isa(a.f, FN)
        return permutedims(StridedView(a.parent, a.size, a.strides, a.offset, adjoint), (2,1))
    elseif isa(a.f, FC)
        return permutedims(StridedView(a.parent, a.size, a.strides, a.offset, transpose), (2,1))
    elseif isa(a.f, FA)
        return permutedims(StridedView(a.parent, a.size, a.strides, a.offset, identity), (2,1))
    else
        return permutedims(StridedView(a.parent, a.size, a.strides, a.offset, conj), (2,1))
    end
end

function sreshape(a::StridedView, newsize::Dims)
    if any(isequal(0), newsize)
        any(isequal(0), size(a)) || throw(DimensionMismatch())
        newstrides = _defaultstrides(newsize)
    else
        newstrides = _computereshapestrides(newsize, size(a), strides(a))
    end
    StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end
_defaultstrides(sz::Tuple{}, s = 1) = ()
_defaultstrides(sz::Dims, s = 1) = (s, _defaultstrides(tail(sz), s*sz[1])...)

struct ReshapeException <: Exception
end
Base.show(io::IO, e::ReshapeException) = print(io, "Cannot produce a reshaped StridedView without allocating, try sreshape(copy(array), newsize) or fall back to reshape(array, newsize)")

# Methods based on map!
Base.copyto!(dst::StridedView{<:Any,N}, src::StridedView{<:Any,N}) where {N} = map!(identity, dst, src)
Base.conj!(a::StridedView{<:Real}) = a
Base.conj!(a::StridedView) = map!(conj, a, a)
LinearAlgebra.adjoint!(dst::StridedView{<:Any,N}, src::StridedView{<:Any,N}) where {N} = copyto!(dst, adjoint(src))
Base.permutedims!(dst::StridedView{<:Any,N}, src::StridedView{<:Any,N}, p) where {N} = copyto!(dst, permutedims(src, p))

# Converting back to other DenseArray type:
Base.convert(T::Type{<:StridedView}, a::StridedView) = a
function Base.convert(T::Type{<:DenseArray}, a::StridedView)
    b = T(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
function Base.convert(::Type{Array}, a::StridedView{T}) where {T}
    b = Array{T}(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where {T} = pointer(a.parent, a.offset+1)

const StridedMatVecView{T} = Union{StridedView{T,1},StridedView{T,2}}


LinearAlgebra.rmul!(dst::StridedView, α::Number) = mul!(dst, dst, α)
LinearAlgebra.lmul!(α::Number, dst::StridedView) = mul!(dst, α, dst)

LinearAlgebra.mul!(dst::StridedView{<:Number,N}, α, src::StridedView{<:Number,N}) where {N} = α == 1 ? copyto!(dst, src) : map!(x->α*x, dst, src)
LinearAlgebra.mul!(dst::StridedView{<:Number,N}, src::StridedView{<:Number,N}, α::Number) where {N} = α == 1 ? copyto!(dst, src) : map!(x->x*α, dst, src)
LinearAlgebra.axpy!(a::Number, X::StridedView{<:Number,N}, Y::StridedView{<:Number,N}) where {N} = a == 1 ? map!(+, Y, X, Y) : map!((x,y)->(a*x+y), Y, X, Y)
LinearAlgebra.axpby!(a::Number, X::StridedView{<:Number,N}, b::Number, Y::StridedView{<:Number,N}) where {N} = b == 1 ? axpy!(a, X, Y) : map!((x,y)->(a*x+b*y), Y, X, Y)

function LinearAlgebra.mul!(C::StridedView{<:Any,2}, A::StridedView{<:Any,2}, B::StridedView{<:Any,2})
    if C.op == conj
        if stride(C,1) < stride(C,2)
            mul!(conj(C), conj(A), conj(B))
        else
            mul!(C', B', A')
        end
    elseif stride(C,1) > stride(C,2)
        mul!(transpose(C), transpose(B), transpose(A))
    else
        if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(size(C)) < Threads.nthreads()*1024
            _mul!(C, A, B)
        else
            mranges, nranges = _computethreadedmulblocks(size(C,1), size(C,2))
            @inbounds Threads.@threads for i = 1:Threads.nthreads()
                _mul!(sview(C, mranges[i], nranges[i]), sview(A, mranges[i], :), sview(B, :, nranges[i]))
            end
        end
    end
    return C
end

_mul!(C::StridedView{<:Any,2}, A::StridedView{<:Any,2}, B::StridedView{<:Any,2}) = __mul!(C, A, B)
function __mul!(C::StridedView{<:Any,2}, A::StridedView{<:Any,2}, B::StridedView{<:Any,2})
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
function _mul!(C::StridedView{T,2}, A::StridedView{T,2}, B::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
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

# Auxiliary routines
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} = (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

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
