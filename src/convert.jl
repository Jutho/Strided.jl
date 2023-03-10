function Base.convert(::Type{T}, a::StridedView) where {T<:Array}
    b = T(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
function Base.convert(::Type{Array}, a::StridedView{T}) where {T}
    b = Array{T}(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end
