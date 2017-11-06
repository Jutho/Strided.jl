const FN = typeof(identity)
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)

# StridedView
struct StridedView{T,N,A<:DenseArray{T},F<:Union{FN,FC,FA,FT}} <: DenseArray{T,N}
    parent::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    op::F
end

StridedView(a::A, size::NTuple{N,Int}, strides::NTuple{N,Int}, offset::Int, op::F = identity) where {T,N,A<:DenseArray{T},F<:Union{FN,FC,FA,FT}} = StridedView{T,N,A,F}(a, size, strides, offset, op)

StridedView(a::StridedArray) = StridedView(parent(a), size(a), strides(a), offset(a), identity)

offset(a::DenseArray) = 0
offset(a::SubArray) = Base.first_index(a) - 1
offset(a::Base.ReshapedArray) = 0
# if VERSION >= v"0.7-"
#     offset(a::ReinterpretedArray) = 0
# end

# Methods for StridedView
Base.parent(a::StridedView) = a.parent
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
Base.stride(a::StridedView, n::Integer) = a.strides[n]
offset(a::StridedView) = a.offset

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

@propagate_inbounds Base.getindex(a::StridedView, I::ParentIndex) = a.op(getindex(a.parent, I.i))
@propagate_inbounds Base.setindex!(a::StridedView, v, I::ParentIndex) = (setindex!(a.parent, a.op(v), I.i); return a)

Base.similar(a::StridedView, ::Type{T}, dims::NTuple{N,Int}) where {N,T}  = StridedView(similar(a.parent, T, dims))
Base.copy(a::StridedView) = copy!(similar(a), a)

# Specialized methods for `StridedView` which produce views/share data
Base.conj(a::StridedView{<:Real}) = a
Base.conj(a::StridedView{T,N,A,FN}) where {T,N,A} = StridedView{T,N,A,FC}(a.parent, a.size, a.strides, a.offset, conj)
Base.conj(a::StridedView{T,N,A,FC}) where {T,N,A} = StridedView{T,N,A,FN}(a.parent, a.size, a.strides, a.offset, identity)
Base.conj(a::StridedView{T,N,A,FT}) where {T,N,A} = StridedView{T,N,A,FA}(a.parent, a.size, a.strides, a.offset, adjoint)
Base.conj(a::StridedView{T,N,A,FA}) where {T,N,A} = StridedView{T,N,A,FT}(a.parent, a.size, a.strides, a.offset, transpose)

function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
    (length(p) == N && isperm(p)) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = TupleTools.permute(a.size, p)
    newstrides = TupleTools.permute(a.strides, p)
    return StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end

Base.transpose(a::StridedView{<:Any,2}) = permutedims(a,(2,1))
adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a),(2,1))
function adjoint(a::StridedView{<:Any,2}) # act recursively, like base
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

splitdims(a::StridedArray, args...) = splitdims(StridedView(a), args...)
fusedims(a::StridedArray, args...) = fusedims(StridedView(a), args...)

const SizeType = Union{Int, Tuple{Vararg{Int}}}

function splitdims(a::StridedView{<:Any,N}, newsizes::Vararg{SizeType,N}) where {N}
    map(prod, newsizes) == size(a) || throw(DimensionMismatch())
    newstrides = _computenewstrides(strides(a), newsizes)
    newsize = TupleTools.vcat(newsizes...)
    return StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end
function splitdims(a::StridedView, s::Pair{Int,<:SizeType})
    i = s[1]
    isize = s[2]
    prod(isize) == size(a, i) || throw(DimensionMismatch())
    istrides = _computenewstride(stride(a, i), isize)
    newsize = TupleTools.insertat(size(a), i, isize)
    newstrides = TupleTools.insertat(strides(a), i, istrides)
    return StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end

function splitdims(a::StridedView, s1::Pair{Int,<:SizeType}, s2::Pair{Int,<:SizeType}, S::Vararg{Pair{Int,<:SizeType}})
    p = TupleTools._sortperm((s1, s2, S...), isless, first, true)
    args = TupleTools.permute((s1, s2, S...), p)
    splitdims(splitdims(a, args[1]), tail(args)...)
end

function fusedims(a::StridedView, i1::Int, i2::Int)
    if stride(a, i1)*size(a, i1) == stride(a, i2)
        newsize = TupleTools.deleteat(TupleTools.setindex(size(a), size(a,i1)*size(a,i2), i1), i2)
        newstrides = TupleTools.deleteat(strides(a), i2)
        StridedView(a.parent, newsize, newstrides, a.offset, a.op)
    else
        error("Can only fuse dimensions with matching strides")
    end
end

# Methods based on map!
Base.copy!(dst::StridedView{<:Any,N}, src::StridedView{<:Any,N}) where {N} = map!(identity, dst, src)
Base.conj!(a::StridedView) = map!(conj, a, a)
Base.permutedims!(dst::StridedView{<:Any,N}, src::StridedView{<:Any,N}, p) where {N} = copy!(dst, permutedims(src, p))
Base.scale!(dst::StridedView{<:Number,N}, α::Number, src::StridedView{<:Number,N}) where {N} = map!(x->α*x, dst, src)
Base.scale!(dst::StridedView{<:Number,N}, src::StridedView{<:Number,N}, α::Number) where {N} = map!(x->x*α, dst, src)
axpy!(a::Number, X::StridedView{<:Number,N}, Y::StridedView{<:Number,N}) where {N} = a == 1 ? map!(+, Y, X, Y) : map!((x,y)->(a*x+y), Y, X, Y)
axpby!(a::Number, X::StridedView{<:Number,N}, b::Number, Y::StridedView{<:Number,N}) where {N} = map!((x,y)->(a*x+b*y), Y, X, Y)

# Converting back to other DenseArray type:
Base.convert(T::Type{<:StridedView}, a::StridedView) = a
Base.convert(T::Type{<:DenseArray}, a::StridedView) = copy!(StridedView(T(size(a))), a)
Base.convert(::Type{Array}, a::StridedView{T}) where {T} = copy!(StridedView(Array{T}(size(a))), a)

Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where {T} = pointer(a.parent, a.offset+1)

const StridedMatVecView{T} = Union{StridedView{T,1},StridedView{T,2}}

function Base.A_mul_B!(C::StridedMatVecView{T}, A::StridedMatVecView{T}, B::StridedMatVecView{T}) where {T<:Base.LinAlg.BlasFloat}
    if !(any(equalto(1), strides(A)) && any(equalto(1), strides(B)) && any(equalto(1), strides(C)))
        Base.LinAlg.generic_matmatmul!(C,'N','N',A,B)
        return C
    end

    if C.op == conj
        if stride(C,1) == 1
            A_mul_B!(conj(C), conj(A), conj(B))
        else
            A_mul_B!(C', B', A')
        end
        return C
    elseif stride(C,1) != 1
        A_mul_B!(transpose(C), transpose(B), transpose(A))
        return C
    end
    # if we reach here, we know that C is in standard form
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
            return Base.LinAlg.generic_matmatmul!(C,'N','N',A,B)
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
            return Base.LinAlg.generic_matmatmul!(C,'N','N',A,B)
        end
    end
    Base.LinAlg.gemm_wrapper!(C,cA,cB,A2,B2)
end

Base.Ac_mul_B!(C::StridedView, A::StridedView, B::StridedView) = Base.A_mul_B!(C, A', B)
Base.A_mul_Bc!(C::StridedView, A::StridedView, B::StridedView) = Base.A_mul_B!(C, A, B')
Base.Ac_mul_Bc!(C::StridedView, A::StridedView, B::StridedView) = Base.A_mul_B!(C, A', B')

# Auxiliary routines
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} = (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

@inline _computenewstride(stride::Int, size::Int) = (stride,)
@inline _computenewstride(stride::Int, size::Tuple{}) = ()
@inline _computenewstride(stride::Int, size::Tuple{Vararg{Int}}) = (stride, _computenewstride(stride*size[1], tail(size))...)

@inline _computenewstrides(strides::Tuple{Int}, sizes::Tuple{SizeType}) = _computenewstride(strides[1], sizes[1])
@inline _computenewstrides(strides::NTuple{N,Int}, sizes::NTuple{N,SizeType}) where {N} =
    (_computenewstride(strides[1], sizes[1])..., _computenewstrides(tail(strides), tail(sizes))...)
