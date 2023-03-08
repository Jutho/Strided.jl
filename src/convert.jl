# Converting back to other DenseArray type:
function Base.convert(T::Type{<:DenseArray}, a::StridedView)
    b = T(undef, size(a))
    copyto!(StridedView(b), a)
    return b
end

# following method because of ambiguity warning
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

