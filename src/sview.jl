function sview(a::StridedArray{<:Any,N}, I::Vararg{Union{RangeIndex,Colon},N}) where {N}
    StridedView(parent(a), _computesize(size(a), I), _computestrides(strides(a), I), offset(a) + _computeoffset(strides(a), I), identity)
end
function sview(a::StridedView{<:Any,N}, I::Vararg{Union{RangeIndex,Colon},N}) where {N}
    StridedView(a.parent, _computesize(a.size, I), _computestrides(a.strides, I), a.offset + _computeoffset(a.strides, I), a.op)
end

@inline function _computesize(oldsize::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Int)
        return _computesize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computesize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computesize(tail(oldsize), tail(I))...)
    end
end
_computesize(::Tuple{}, ::Tuple{}) = ()
@inline function _computestrides(oldstrides::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Int)
        return _computestrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computestrides(tail(oldstrides), tail(I))...)
    else
        return (oldstrides[1]*step(I[1]), _computestrides(tail(oldstrides), tail(I))...)
    end
end
_computestrides(::Tuple{}, ::Tuple{}) = ()

@inline function _computeoffset(strides::NTuple{N,Int}, I::NTuple{N,Union{RangeIndex,Colon}}) where {N}
    if isa(I[1], Int)
        return (I[1]-1)*strides[1]+_computeoffset(tail(strides), tail(I))
    elseif isa(I[1], Colon)
        return _computeoffset(tail(strides), tail(I))
    else
        return (first(I[1])-1)*strides[1]+_computeoffset(tail(strides), tail(I))
    end
end
_computeoffset(::Tuple{}, ::Tuple{}) = 0
