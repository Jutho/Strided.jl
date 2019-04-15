const FN = typeof(identity)
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)
_conj(::FN) = conj
_conj(::FC) = identity
_conj(::FA) = transpose
_conj(::FT) = adjoint
_transpose(::FN) = transpose
_transpose(::FC) = adjoint
_transpose(::FA) = conj
_transpose(::FT) = identity
_adjoint(::FN) = adjoint
_adjoint(::FC) = transpose
_adjoint(::FA) = identity
_adjoint(::FT) = conj

abstract type AbstractStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N} end

Base.elsize(::Type{<:AbstractStridedView{T}}) where {T} =
    Base.isbitstype(T) ? sizeof(T) :
        (Base.isbitsunion(T) ? Base.bitsunionsize(T) : sizeof(Ptr))

# Converting back to other DenseArray type:
function Base.convert(T::Type{<:DenseArray}, a::AbstractStridedView)
    b = T(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
# following method because of ambiguity warning
function Base.convert(::Type{T}, a::AbstractStridedView) where {T<:Array}
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

function LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, α::Number, src::AbstractStridedView{<:Number,N}) where {N}
    if α == 1
        copyto!(dst, src)
    else
        dst .= α .* src
    end
    return dst
end
function LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, src::AbstractStridedView{<:Number,N}, α::Number) where {N}
    if α == 1
        copyto!(dst, src)
    else
        dst .= src .* α
    end
    return dst
end
function LinearAlgebra.axpy!(a::Number, X::AbstractStridedView{<:Number,N}, Y::AbstractStridedView{<:Number,N}) where {N}
    if a == 1
        Y .= X .+ Y
    else
        Y .= a .* X .+ Y
    end
    return Y
end
function LinearAlgebra.axpby!(a::Number, X::AbstractStridedView{<:Number,N}, b::Number, Y::AbstractStridedView{<:Number,N}) where {N}
    if b == 1
        axpy!(a, X, Y)
    elseif b == 0
        mul!(Y, a, X)
    else
        Y .= a .* X .+ b .* Y
    end
    return Y
end

function LinearAlgebra.mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{<:Any,2}, B::AbstractStridedView{<:Any,2}, α = true, β = false) where {T}
    if !(eltype(C) <: LinearAlgebra.BlasFloat && eltype(A) == eltype(B) == eltype(C))
        return __mul!(C, A, B, α, β)
    end
    # C.op is identity or conj
    if C.op == conj
        if stride(C,1) < stride(C,2)
            _mul!(conj(C), conj(A), conj(B), conj(α), conj(β))
        else
            _mul!(C', B', A', conj(α), conj(β))
        end
    elseif stride(C,1) > stride(C,2)
        _mul!(transpose(C), transpose(B), transpose(A), α, β)
    else
        _mul!(C, A, B, α, β)
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

function isblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        return stride(A,1) == 1 || stride(A,2) == 1
    elseif A.op == conj
        return stride(A, 2) == 1
    else # should never happen
        return false
    end
end
function getblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        if stride(A,1) == 1
            return blasstrides(A), 'N'
        else
            return blasstrides(transpose(A)), 'T'
        end
    else
        return blasstrides(adjoint(A)), 'C'
    end
end

# here we will have C.op == :identity && stride(C,1) < stride(C,2)
function _mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2}, B::AbstractStridedView{T,2}, α, β) where {T<:LinearAlgebra.BlasFloat}
    if stride(C,1) == 1 && isblasmatrix(A) && isblasmatrix(B)
        A2, CA = getblasmatrix(A)
        B2, CB = getblasmatrix(B)
        C2 = blasstrides(C)
        LinearAlgebra.BLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C2)
    else
        return __mul!(C, A, B, α, β)
    end
end

# This implementation is faster than LinearAlgebra.generic_matmatmul
function __mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2}, B::AbstractStridedView{<:Any,2}, α, β)
    (size(C,1) == size(A,1) && size(C,2) == size(B,2) && size(A,2) == size(B,1)) ||
        throw(DimensionMatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))

    m,n = size(C)
    k = size(A,2)
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B,(2,1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    if α == 1
        if β == 0
            _mapreducedim!(*, +, zero, (m,n,k), (C2,A2,B2))
        elseif β == 1
            _mapreducedim!(*, +, nothing, (m,n,k), (C2,A2,B2))
        else
            _mapreducedim!(*, +, x->x*β, (m,n,k), (C2,A2,B2))
        end
    elseif α != 0
        f = (x,y)->(x*y*α)
        if β == 0
            _mapreducedim!(f, +, zero, (m,n,k), (C2,A2,B2))
        elseif β == 1
            _mapreducedim!(f, +, nothing, (m,n,k), (C2,A2,B2))
        else
            _mapreducedim!(f, +, x->x*β, (m,n,k), (C2,A2,B2))
        end
    else
        rmul!(C, β)
    end
    return C
end

# ParentIndex: index directly into parent array
struct ParentIndex
    i::Int
end

function sreshape end

sreshape(a::AbstractStridedView,args::Vararg{Int}) = sreshape(a, args)

struct ReshapeException <: Exception
end
Base.show(io::IO, e::ReshapeException) = print(io, "Cannot produce a reshaped StridedView without allocating, try sreshape(copy(array), newsize) or fall back to reshape(array, newsize)")

@inline sview(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(a, I...)
@inline sview(a::AbstractStridedView, I::SliceIndex) =
    getindex(sreshape(a, (length(a),)), I...)

@inline Base.view(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(a, I...)

@inline sview(a::DenseArray{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(StridedView(a), I...)
@inline sview(a::DenseArray, I::SliceIndex) =
    getindex(sreshape(StridedView(a), (length(a),)), I...)

# Auxiliary routines
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} =
    (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

@inline function _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
_computeviewsize(::Tuple{}, ::Tuple{}) = ()

@inline function _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewstrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else
        return (oldstrides[1]*step(I[1]), _computeviewstrides(tail(oldstrides), tail(I))...)
    end
end
_computeviewstrides(::Tuple{}, ::Tuple{}) = ()

@inline function _computeviewoffset(strides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Colon)
        return _computeviewoffset(tail(strides), tail(I))
    else
        return (first(I[1])-1)*strides[1]+_computeviewoffset(tail(strides), tail(I))
    end
end
_computeviewoffset(::Tuple{}, ::Tuple{}) = 0

_simplify(size::Tuple{}, strides::Tuple{}) = size, strides
_simplify(size::Dims{1}, strides::Dims{1}) = size, strides
function _simplify(size::Dims{N}, strides::Dims{N}) where {N}
    tailsize, tailstrides = _simplify(tail(size), tail(strides))
    if size[1] == 1
        return (tailsize..., 1), (tailstrides..., strides[1])
    elseif size[1]*strides[1] == tailstrides[1]
        return (size[1]*tailsize[1], tail(tailsize)..., 1),
            (strides[1], tail(tailstrides)..., tailsize[1]*tailstrides[1])
    else
        return (size[1], tailsize...), (strides[1], tailstrides...)
    end
end

_computereshapestrides(newsize::Tuple{}, oldsize::Tuple{}, strides::Tuple{}) = strides
function _computereshapestrides(newsize::Tuple{}, oldsize::Dims{N}, strides::Dims{N}) where {N}
    all(isequal(1), oldsize) || throw(DimensionMismatch())
    return ()
end

function _computereshapestrides(newsize::Dims, oldsize::Tuple{}, strides::Tuple{})
    all(isequal(1), newsize) || throw(DimensionMismatch())
    return newsize
end
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, strides::Dims{N}) where {N}
    d,r = divrem(oldsize[1], newsize[1])
    if r == 0
        s1 = strides[1]
        if d == 1
            oldsize = (tail(oldsize)..., 1)
            strides = (tail(strides)..., newsize[1]*s1)
            return (s1, _computereshapestrides(tail(newsize), oldsize, strides)...)
        else
            oldsize = (d, tail(oldsize)...)
            strides = (newsize[1]*s1, tail(strides)...)
            return (s1, _computereshapestrides(tail(newsize), oldsize, strides)...)
        end
    else
        if prod(newsize) != prod(oldsize)
            throw(DimensionMismatch())
        else
            throw(ReshapeException())
        end
    end
end

function _computethreadedmulblocks end
let
    mranges = Vector{UnitRange{Int}}(undef, Threads.nthreads())
    nranges = Vector{UnitRange{Int}}(undef, Threads.nthreads())
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
