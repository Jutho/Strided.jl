Base.map!(f::F, b::StridedView{<:Any,N}, a1::StridedView{<:Any,N}, A::Vararg{StridedView{<:Any,N}}) where {F,N} = mapi!(f, b, a1, A...)
function mapr!(f::F, b::StridedView{<:Any,N}, a1::StridedView{<:Any,N}, A::Vararg{StridedView{<:Any,N}}) where {F,N}
    dims = size(b)
    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch)
    for a in A
        size(a) == dims || throw(DimensionMismatch)
    end

    # Sort loops based on minimal memory jumps
    allstrides = map(strides, (b, a1, A...))
    minstrides = map(min, allstrides...)
    p = TupleTools._sortperm(map((d,s)->(d-1)*s, dims, minstrides))
    dims = TupleTools.getindices(dims, p)
    minstrides = TupleTools.getindices(minstrides, p)
    allstrides = map(s->TupleTools.getindices(s, p), allstrides)

    # Fuse dimensions if possible
    for i = N:-1:2
        merge = true
        for s in allstrides
            if s[i] != dims[i-1]*s[i-1]
                merge = false
                break
            end
        end
        if merge
            dims = setindex(dims, dims[i-1]*dims[i], i-1)
            dims = setindex(dims, 1, i)
        end
    end
    map_recursive!(f, dims, minstrides, (b, a1, A...), allstrides)
    return b
end
function mapi!(f::F, b::StridedView{<:Any,N}, a1::StridedView{<:Any,N}, A::Vararg{StridedView{<:Any,N}}) where {F,N}
    dims = size(b)
    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch)
    for a in A
        size(a) == dims || throw(DimensionMismatch)
    end

    # Sort loops based on minimal memory jumps
    allstrides = map(strides, (b, a1, A...))
    minstrides = map(min, allstrides...)
    p = TupleTools._sortperm(map((d,s)->(d-1)*s, dims, minstrides))
    dims = TupleTools.getindices(dims, p)
    minstrides = TupleTools.getindices(minstrides, p)
    allstrides = map(s->TupleTools.getindices(s, p), allstrides)

    # Fuse dimensions if possible
    for i = N:-1:2
        merge = true
        for s in allstrides
            if s[i] != dims[i-1]*s[i-1]
                merge = false
                break
            end
        end
        if merge
            dims = setindex(dims, dims[i-1]*dims[i], i-1)
            dims = setindex(dims, 1, i)
        end
    end

    blocks = _computeblocks(dims, minstrides, allstrides)
    map_iterative!(f, dims, blocks, (b, a1, A...), allstrides)
    return b
end

const BLOCKSIZE = 1024
function map_recursive!(f::F, dims::NTuple{N,Int}, minstrides::NTuple{N,Int}, arrs::NTuple{M,StridedView{<:Any,N}}, arrstrides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int} = map(offset, arrs)) where {F,N,M}
    l = TupleTools.prod(dims)
    if l <= BLOCKSIZE || (dims[1] == l && all(equalto(1), map(TupleTools.indmin, arrstrides)))
        map_kernel!(f, dims, arrs, arrstrides, offsets)
    else
        i = TupleTools.indmax( (dims .- 1) .* minstrides )
        diold = dims[i]
        dinew = diold >> 1
        map_recursive!(f, setindex(dims, dinew, i), minstrides, arrs, arrstrides, offsets)
        stridesi = let j = i
            map(s->s[j], arrstrides)
        end
        offsets = offsets .+ dinew .* stridesi
        dinew = diold - dinew
        map_recursive!(f, setindex(dims, dinew, i), minstrides, arrs, arrstrides, offsets)
    end
    return arrs[1]
end

@generated function map_kernel!(f::F, dims::NTuple{N,Int}, arrs::NTuple{M,StridedView{<:Any,N}}, arrstrides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {F,N,M}
    loopvars = [gensym() for k = 1:N]
    stridesym = [Symbol("strides_$(i)_$(j)") for i = 1:N, j=1:M]
    Isym = [Symbol("I_$(j)") for j=1:M]
    pre1 = Expr(:block, [:($(stridesym[i,j]) = arrstrides[$j][$i]) for i = 1:N, j=1:M]...)
    pre2 = Expr(:block, [:($(Isym[j]) = offsets[$j]+1) for j = 1:M]...)
    ex = :(arrs[1][ParentIndex($(Isym[1]))] = f($([:(arrs[$j][ParentIndex($(Isym[j]))]) for j = 2:M]...)))
    if N >= 1
        i = 1
        ex = quote
            @simd for $(loopvars[i]) = 1:dims[$i]
                $ex
                $(Expr(:block, [:($(Isym[j]) += $(stridesym[i,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Isym[j]) -=  dims[$i] * $(stridesym[i,j])) for j = 1:M]...))
        end
    end
    for i = 2:N
        ex = quote
            for $(loopvars[i]) = 1:dims[$i]
                $ex
                $(Expr(:block, [:($(Isym[j]) += $(stridesym[i,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Isym[j]) -=  dims[$i] * $(stridesym[i,j])) for j = 1:M]...))
        end
    end
    quote
        $pre1
        $pre2
        @inbounds $ex
    end
end

@generated function map_iterative!(f::F, dims::NTuple{N,Int}, blocks::NTuple{N,Int}, As::NTuple{M,StridedView{<:Any,N}}, strides::NTuple{M,NTuple{N,Int}}) where {F,N,M}
    blockloopvars = [Symbol("J$i") for i = 1:N]
    blockdimvars = [Symbol("d$i") for i = 1:N]
    innerloopvars = [Symbol("j$i") for i = 1:N]

    stridevars = [Symbol("stride_$(i)_$(j)") for i = 1:N, j = 1:M]
    Ivars = [Symbol("I$j") for j = 1:M]
    pre1 = Expr(:block, [:($(stridevars[i,j]) = strides[$j][$i]) for i = 1:N, j=1:M]...)
    pre2 = Expr(:block, [:($(Ivars[j]) = offset(As[$j])+1) for j = 1:M]...)

    ex = :(As[1][ParentIndex($(Ivars[1]))] = f($([:(As[$j][ParentIndex($(Ivars[j]))]) for j = 2:M]...)))
    if N >= 1
        i = 1
        ex = quote
            @simd for $(innerloopvars[i]) = Base.OneTo($(blockdimvars[i]))
                $ex
                $(Expr(:block, [:($(Ivars[j]) += $(stridevars[i,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Ivars[j]) -=  $(blockdimvars[i]) * $(stridevars[i,j])) for j = 1:M]...))
        end
    end
    for i = 2:N
        ex = quote
            for $(innerloopvars[i]) = Base.OneTo($(blockdimvars[i]))
                $ex
                $(Expr(:block, [:($(Ivars[j]) += $(stridevars[i,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Ivars[j]) -=  $(blockdimvars[i]) * $(stridevars[i,j])) for j = 1:M]...))
        end
    end
    for i = 1:N
        ex = quote
            for $(blockloopvars[i]) = 1:blocks[$i]:dims[$i]
                $(blockdimvars[i]) = min(blocks[$i], dims[$i]-$(blockloopvars[i])+1)
                $ex
                $(Expr(:block, [:($(Ivars[j]) +=  $(blockdimvars[i]) * $(stridevars[i,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Ivars[j]) -=  dims[$i] * $(stridevars[i,j])) for j = 1:M]...))
        end
    end
    quote
        $pre1
        $pre2
        @inbounds $ex
        return As[1]
    end
end

_computeblocks(dims::Tuple{}, minstrides::Tuple{}, allstrides::Tuple{Vararg{Tuple{}}}, blocksize::Int = BLOCKSIZE) = ()
function _computeblocks(dims::NTuple{N,Int}, minstrides::NTuple{N,Int}, allstrides::Tuple{Vararg{NTuple{N,Int}}}, blocksize::Int = BLOCKSIZE) where {N}
    # strides1 is assumed to be sorted
    if all(equalto(1), map(indmin, allstrides))
        return (dims[1], _computeblocks(tail(dims), tail(minstrides), map(tail, allstrides), div(blocksize, dims[1]))...)
    elseif blocksize == 0
        return ntuple(n->1, StaticLength(N))
    else
        blocks = dims
        while prod(blocks) >= 2*blocksize
            i = indmax(map((d,s)->(d-1)*s, blocks, minstrides))
            blocks = setindex(blocks, (blocks[i]+1)>>1, i)
        end
        while prod(blocks) > blocksize
            i = indmax(map((d,s)->(d-1)*s, blocks, minstrides))
            blocks = setindex(blocks, blocks[i]-1, i)
        end
        return blocks
    end
end
