const BLOCKSIZE = 1024

function Base.map!(f::F, b::StridedView{<:Any,N}, a1::StridedView{<:Any,N}, A::Vararg{StridedView{<:Any,N}}) where {F,N}
    dims = size(b)
    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch)
    for a in A
        size(a) == dims || throw(DimensionMismatch)
    end
    l = prod(dims)
    if l == 0
        # don't do anything
    elseif l <= BLOCKSIZE
        allstrides = map(strides, (b, a1, A...))
        offsets = map(offset, (b, a1, A...))
        map_iterative!(f, dims, dims, (b, a1, A...), allstrides, offsets)
    else
        mapi!(f, dims, b, a1, A...)
    end
    return b
end

function mapi!(f::F, dims::NTuple{N,Int}, b::StridedView{<:Any,N}, a::StridedView{<:Any,N}) where {F,N}
    # Sort loops based on minimal memory jumps
    # As = (b, a1, A...)
    bstrides = strides(b)
    astrides = strides(a)
    minstrides = map(min, bstrides, astrides)
    p = TupleTools._sortperm((dims .- 1) .* minstrides)
    dims = TupleTools.getindices(dims, p)
    minstrides = TupleTools.getindices(minstrides, p)
    allstrides = (TupleTools.getindices(bstrides, p), TupleTools.getindices(astrides, p))

    # Fuse dimensions if possible
    @inbounds for i = N:-1:2
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
    offsets = (offset(b), offset(a))
    if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(dims) < Threads.nthreads()*prod(blocks)
        map_iterative!(f, dims, blocks, (b, a), allstrides, offsets)
    else
        n = Threads.nthreads()
        threadblocks, threadoffsets = _computethreadblocks(n, dims, minstrides, allstrides, offsets)
        _mapit!(threadblocks, threadoffsets, f, blocks, (b, a), allstrides)
    end
    return b
end
function mapi!(f::F, dims::NTuple{N,Int}, b::StridedView{<:Any,N}, a1::StridedView{<:Any,N}, a2::StridedView{<:Any,N}, A::Vararg{StridedView{<:Any,N}}) where {F,N}
    # Sort loops based on minimal memory jumps
    bstrides = strides(b)
    a1strides = strides(a1)
    a2strides = strides(a2)
    Astrides = map(strides, A)
    allstrides = (bstrides, a1strides, a2strides, Astrides...)
    minstrides = map(min, map(min, map(min, bstrides, a1strides), a2strides), Astrides...)
    p = TupleTools._sortperm(map((x,y)->(x-1)*y,dims,minstrides))
    dims = TupleTools.getindices(dims, p)
    minstrides = TupleTools.getindices(minstrides, p)
    allstrides = let q = p
        map(s->TupleTools.getindices(s, q), allstrides)
    end

    # Fuse dimensions if possible
    @inbounds for i = N:-1:2
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
    offsets = map(offset, (b, a1, a2, A...))
    if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(dims) < Threads.nthreads()*prod(blocks)
        map_iterative!(f, dims, blocks, (b, a1, a2, A...), allstrides, offsets)
    else
        n = Threads.nthreads()
        threadblocks, threadoffsets = _computethreadblocks(n, dims, minstrides, allstrides, offsets)
        _mapit!(threadblocks, threadoffsets, f, blocks, (b, a1, a2, A...), allstrides)
    end
    return b
end
@noinline function _mapit!(threadblocks, threadoffsets, f, blocks, As, allstrides)
    @inbounds Threads.@threads for i = 1:length(threadblocks)
        map_iterative!(f, threadblocks[i], blocks, As, allstrides, threadoffsets[i])
    end
end

@generated function map_iterative!(f::F, dims::NTuple{N,Int}, blocks::NTuple{N,Int}, As::NTuple{M,StridedView{<:Any,N}}, strides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {F,N,M}
    blockloopvars = [Symbol("J$i") for i = 1:N]
    blockdimvars = [Symbol("d$i") for i = 1:N]
    innerloopvars = [Symbol("j$i") for i = 1:N]

    stridevars = [Symbol("stride_$(i)_$(j)") for i = 1:N, j = 1:M]
    Ivars = [Symbol("I$j") for j = 1:M]
    Avars = [Symbol("A$j") for j = 1:M]
    pre1 = Expr(:block, [:($(Avars[j]) = As[$j]) for j = 1:M]...)
    pre2 = Expr(:block, [:($(stridevars[i,j]) = strides[$j][$i]) for i = 1:N, j=1:M]...)
    pre3 = Expr(:block, [:($(Ivars[j]) = offsets[$j]+1) for j = 1:M]...)

    ex = :(A1[ParentIndex($(Ivars[1]))] = f($([:($(Avars[j])[ParentIndex($(Ivars[j]))]) for j = 2:M]...)))
    i = 1
    if N >= 1
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
    for i2 = 1:N
        ex = quote
            for $(blockloopvars[i2]) = 1:blocks[$i2]:dims[$i2]
                $(blockdimvars[i2]) = min(blocks[$i2], dims[$i2]-$(blockloopvars[i2])+1)
                $ex
                $(Expr(:block, [:($(Ivars[j]) +=  $(blockdimvars[i2]) * $(stridevars[i2,j])) for j = 1:M]...))
            end
            $(Expr(:block, [:($(Ivars[j]) -=  dims[$i2] * $(stridevars[i2,j])) for j = 1:M]...))
        end
    end
    quote
        $pre1
        $pre2
        $pre3
        @inbounds $ex
        return A1
    end
end

_computeblocks(dims::Tuple{}, minstrides::Tuple{}, allstrides::Tuple{Vararg{Tuple{}}}, blocksize::Int = BLOCKSIZE) = ()
function _computeblocks(dims::NTuple{N,Int}, minstrides::NTuple{N,Int}, allstrides::Tuple{Vararg{NTuple{N,Int}}}, blocksize::Int = BLOCKSIZE) where {N}
    # minstrides is assumed to be sorted
    if prod(dims) <= blocksize
        return dims
    elseif all(isequal(1), map(TupleTools.argmin, allstrides))
        return (dims[1], _computeblocks(tail(dims), tail(minstrides), map(tail, allstrides), div(blocksize, dims[1]))...)
    elseif blocksize == 0
        return ntuple(n->1, StaticLength(N))
    else
        blocks = dims
        while prod(blocks) >= 2*blocksize
            i = TupleTools.argmax((blocks .- 1).*minstrides)
            blocks = setindex(blocks, (blocks[i]+1)>>1, i)
        end
        while prod(blocks) > blocksize
            i = TupleTools.argmax((blocks .- 1).*minstrides)
            blocks = setindex(blocks, blocks[i]-1, i)
        end
        return blocks
    end
end
function _computethreadblocks(n::Int, dims::NTuple{N,Int}, minstrides::NTuple{N,Int}, allstrides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {N,M}
    factors = reverse!(simpleprimefactorization(n))

    threadblocks = [dims]
    threadoffsets = [offsets]
    for k in factors
        l = length(threadblocks)
        for j = 1:l
            dims = popfirst!(threadblocks)
            offsets = popfirst!(threadoffsets)
            i = TupleTools.argmax((dims.-k).*minstrides) # make sure that ndi is at least 1 by subtracting k from dims
            ndi = div(dims[i], k)
            newdims = setindex(dims, ndi, i)
            stridesi = let j = i
                map(s->s[j], allstrides)
            end
            for m = 1:k-1
                push!(threadblocks, newdims)
                push!(threadoffsets, offsets)
                offsets = offsets .+ ndi.* stridesi
            end
            ndi = dims[i]-(k-1)*ndi
            newdims = setindex(dims, ndi, i)
            push!(threadblocks, newdims)
            push!(threadoffsets, offsets)
        end
    end
    return threadblocks, threadoffsets
end
