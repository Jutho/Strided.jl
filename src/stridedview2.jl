
# StridedView
struct StridedView{T,N,A<:DenseArray{T}} <: DenseArray{T,N}
    parent::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
end
StridedView(a::DenseArray{T}, size::NTuple{N,Int}, strides::NTuple{N,Int}, offset::Int) where {T,N} = StridedView{T,N,typeof(a)}(a, size, strides, offset)

StridedView(a::StridedArray) = StridedView(parent(a), size(a), strides(a), offset(a))

offset(a::DenseArray) = 0
offset(a::SubArray) = Base.first_index(a) - 1
offset(a::Base.ReshapedArray) = 0
# if VERSION >= v"0.7-"
#     offset(a::ReinterpretedArray) = 0
# end

# Methods for StridedView
Base.parent(a::StridedView) = a.size
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
offset(a::StridedView) = a.offset

Base.IndexStyle(::Type{<:StridedView}) = Base.IndexCartesian()

# Indexing with N integer arguments
@inline function Base.getindex(a::StridedView{<:Any,N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds r = a.parent[a.offset+_computeind(I, a.strides)]
    return r
end
@inline function Base.setindex!(a::StridedView{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds a.parent[a.offset+_computeind(I, a.strides)] = v
    return a
end

# ParentIndex: index directly into parent array
struct ParentIndex
    i::Int
end

@propagate_inbounds Base.getindex(a::StridedView, I::ParentIndex) = getindex(a.parent, I.i)
@propagate_inbounds Base.setindex!(a::StridedView, v, I::ParentIndex) = (setindex!(a.parent, v, I.i); return a)

Base.similar(a::StridedView, args...) = similar(a.parent, args...)

# Specialized methods for `StridedView`
function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
    (length(p) == N && isperm(p)) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = TupleTools.permute(a.size, p)
    newstrides = TupleTools.permute(a.strides, p)
    return StridedView(a.parent, newsize, newstrides, a.offset)
end

const SizeType = Union{Int, Tuple{Vararg{Int}}}

splitdims(a::StridedArray, args...) = splitdims(StridedView(a), args...)

function splitdims(a::StridedView{<:Any,N}, newsizes::Vararg{SizeType,N}) where {N}
    map(prod, newsizes) == size(a) || throw(DimensionMismatch())
    newstrides = _computenewstrides(strides(a), newsizes)
    newsize = TupleTools.vcat(newsizes...)
    return StridedView(parent(a), newsize, newstrides, offset(a))
end
function splitdims(a::StridedView, s::Pair{Int,SizeType})
    i = s[1]
    isize = s[2]
    prod(isize) == size(a, i) || throw(DimensionMismatch())
    istrides = _computestrides(stride(a, i), isize)
    newsize = TupleTools.insertat(size(a), i, isize)
    newstrides = TupleTools.insertat(strides(a), i, istrides)
    return StridedView(parent(a), newsize, newstrides, offset(a))
end

function splitdims(a::StridedView, s1::Pair{Int,SizeType}, s2::Pair{Int,SizeType}, S::Vararg{Pair{Int,SizeType}})
    p = sortperm((s1,s2, S...), by=first, rev=true)
    args = permute((s1, s2, S...), p)
    splitdims(splitdims(a, args[1]), tail(args)...)
end

# Auxiliary routines
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} = (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

@inline _computenewstrides(stride::Int, size::Int) = (stride,)
@inline _computenewstrides(stride::Int, size::Tuple{}) = ()
@inline _computenewstrides(stride::Int, size::Tuple{Vararg{Int}}) = (stride, _computenewstrides(stride*size[1], tail(size))...)

@inline _computenewstrides(strides::Tuple{Int}, sizes::Tuple{SizeType}) = _computenewstrides(strides[1], sizes[1])
@inline _computenewstrides(strides::NTuple{N,Int}, sizes::NTuple{N,SizeType}) where {N} =
    (_computenewstrides(strides[1], sizes[1])..., _computenewstrides(tail(strides), tail(sizes))...)
