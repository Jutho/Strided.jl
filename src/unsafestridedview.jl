# UnsafeStridedView
struct UnsafeStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractStridedView{T,N,F}
    ptr::Ptr{T}
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    op::F
end

UnsafeStridedView(a::Ptr{T}, size::NTuple{N,Int}, strides::NTuple{N,Int}, offset::Int = 0) where {T,N} = UnsafeStridedView{T,N,FN}(a, size, strides, offset, identity)
UnsafeStridedView(a::StridedArray) = UnsafeStridedView(pointer(a), size(a), strides(a))

# Methods for UnsafeStridedView
Base.size(a::UnsafeStridedView) = a.size
Base.strides(a::UnsafeStridedView) = a.strides
Base.stride(a::UnsafeStridedView{<:Any, N}, n::Int) where {N} = (n <= N) ? a.strides[n] : a.strides[N]*a.size[N]
offset(a::UnsafeStridedView) = a.offset

Base.similar(a::UnsafeStridedView, ::Type{T}, dims::NTuple{N,Int}) where {N,T}  = StridedView(Array{T}(undef, dims))
Base.copy(a::UnsafeStridedView) = copyto!(similar(a), a)

Base.unsafe_convert(::Type{Ptr{T}}, a::UnsafeStridedView{T}) where {T} = a.ptr + Base.elsize(a)*a.offset

Base.dataids(a::UnsafeStridedView) = (UInt(a.ptr),)

Base.IndexStyle(::Type{<:UnsafeStridedView}) = Base.IndexCartesian()

# Indexing with N integer arguments
@inline function Base.getindex(a::UnsafeStridedView{<:Any,N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    r = a.op(unsafe_load(a.ptr, a.offset+_computeind(I, a.strides)))
    return r
end
@inline function Base.setindex!(a::UnsafeStridedView{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    unsafe_store!(a.ptr, a.op(v), a.offset+_computeind(I, a.strides))
    return a
end

function Base.getindex(a::UnsafeStridedView{<:Any,N}, I::Vararg{Union{RangeIndex,Colon},N}) where {N}
    UnsafeStridedView(a.ptr, _computeviewsize(a.size, I), _computeviewstrides(a.strides, I), a.offset + _computeviewoffset(a.strides, I), a.op)
end

@propagate_inbounds Base.getindex(a::UnsafeStridedView, I::ParentIndex) = a.op(unsafe_load(a.ptr, I.i))
@propagate_inbounds Base.setindex!(a::UnsafeStridedView, v, I::ParentIndex) = (unsafe_store!(a.ptr, a.op(v), I.i); return a)

# Specialized methods for `UnsafeStridedView` which produce views/share data
Base.conj(a::UnsafeStridedView{<:Real}) = a
Base.conj(a::UnsafeStridedView) = UnsafeStridedView(a.ptr, a.size, a.strides, a.offset, _methodconj(a.op))

function Base.permutedims(a::UnsafeStridedView{<:Any,N}, p) where {N}
    (length(p) == N && TupleTools.isperm(p)) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = TupleTools._permute(a.size, p)
    newstrides = TupleTools._permute(a.strides, p)
    return UnsafeStridedView(a.ptr, newsize, newstrides, a.offset, a.op)
end

LinearAlgebra.transpose(a::UnsafeStridedView{<:Any,2}) = permutedims(a, (2,1))
LinearAlgebra.adjoint(a::UnsafeStridedView{<:Number,2}) = permutedims(conj(a), (2,1))
function LinearAlgebra.adjoint(a::UnsafeStridedView{<:Any,2}) # act recursively, like Base
    if isa(a.f, FN)
        return permutedims(UnsafeStridedView(a.ptr, a.size, a.strides, a.offset, adjoint), (2,1))
    elseif isa(a.f, FC)
        return permutedims(UnsafeStridedView(a.ptr, a.size, a.strides, a.offset, transpose), (2,1))
    elseif isa(a.f, FA)
        return permutedims(UnsafeStridedView(a.ptr, a.size, a.strides, a.offset, identity), (2,1))
    else
        return permutedims(UnsafeStridedView(a.ptr, a.size, a.strides, a.offset, conj), (2,1))
    end
end

function sreshape(a::UnsafeStridedView, newsize::Dims)
    if any(isequal(0), newsize)
        any(isequal(0), size(a)) || throw(DimensionMismatch())
        newstrides = _defaultstrides(newsize)
    else
        newstrides = _computereshapestrides(newsize, size(a), strides(a))
    end
    UnsafeStridedView(a.ptr, newsize, newstrides, a.offset, a.op)
end
