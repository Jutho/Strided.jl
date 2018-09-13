# StridedView
struct StridedView{T,N,A<:DenseArray{T},F<:Union{FN,FC,FA,FT}} <: AbstractStridedView{T,N,F}
    parent::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    op::F
end

StridedView(a::A, size::NTuple{N,Int}, strides::NTuple{N,Int}, offset::Int = 0) where {T,N,A<:DenseArray{T}} = StridedView{T,N,A,FN}(a, size, strides, offset, identity)
StridedView(a::DenseArray) = StridedView(a, size(a), strides(a))

# Methods for StridedView
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
Base.stride(a::StridedView{<:Any, N}, n::Int) where {N} = (n <= N) ? a.strides[n] : a.strides[N]*a.size[N]
offset(a::StridedView) = a.offset

Base.similar(a::StridedView, ::Type{T}, dims::NTuple{N,Int}) where {N,T}  = StridedView(similar(a.parent, T, dims))
Base.copy(a::StridedView) = copyto!(similar(a), a)

Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where {T} = pointer(a.parent, a.offset+1)

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

# force inlining so that view typically does not need to be created
@inline function Base.getindex(a::StridedView{<:Any,N}, I::Vararg{Union{RangeIndex,Colon},N}) where {N}
    StridedView(a.parent, _computeviewsize(a.size, I), _computeviewstrides(a.strides, I), a.offset + _computeviewoffset(a.strides, I), a.op)
end

@propagate_inbounds Base.getindex(a::StridedView, I::ParentIndex) = a.op(getindex(a.parent, I.i))
@propagate_inbounds Base.setindex!(a::StridedView, v, I::ParentIndex) = (setindex!(a.parent, a.op(v), I.i); return a)

# Specialized methods for `StridedView` which produce views/share data
Base.conj(a::StridedView{<:Real}) = a
Base.conj(a::StridedView) = StridedView(a.parent, a.size, a.strides, a.offset, _methodconj(a.op))

@inline function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
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

@inline function sreshape(a::StridedView, newsize::Dims)
    if any(isequal(0), newsize)
        any(isequal(0), size(a)) || throw(DimensionMismatch())
        newstrides = _defaultstrides(newsize)
    else
        newstrides = _computereshapestrides(newsize, size(a), strides(a))
    end
    StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end
