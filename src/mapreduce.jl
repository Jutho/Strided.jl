const BLOCKSIZE = 1024

function Base.map(f::F, a1::AbstractStridedView{<:Any,N}, A::Vararg{AbstractStridedView{<:Any,N}}) where {F,N}
    T = Base.promote_eltype(a1, A...)
    map!(f, similar(a1, T), a1, A...)
end

function Base.map!(f::F, b::AbstractStridedView{<:Any,N}, a1::AbstractStridedView{<:Any,N}, A::Vararg{AbstractStridedView{<:Any,N}}) where {F,N}
    dims = size(b)

    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch())
    for a in A
        size(a) == dims || throw(DimensionMismatch())
    end

    any(isequal(0), dims) && return b # don't do anything

    _mapreducedim1!(f, nothing, dims, (b, a1, A...))

    return b
end

function Base.mapreducedim!(f::F, op::G, b::AbstractStridedView{<:Any,N}, a1::AbstractStridedView{<:Any,N}, A::Vararg{AbstractStridedView{<:Any,N}}) where {F,G,N}
    outdims = size(b)
    dims = map(max, outdims, map(max, map(size, (a1, A...))...))

    # Check dimesions
    Broadcast.check_broadcast_axes(map(Base.OneTo, dims), b, a1, A...)

    any(isequal(0), dims) && return b # don't do anything

    _mapreducedim1!(f, op, dims, promoteshape(dims, b, a1, A...))

    return b
end

function _mapreducedim1!(f::F, op::G, dims::NTuple{N,Int}, arrays::NTuple{M,AbstractStridedView}) where {F,G,N,M}
    # Level 1: fuse dimensions if possible: assume that at least one array, e.g. the output array in arrays[1],
    # has its strides sorted
    allstrides = map(strides, arrays)
    @inbounds for i = N:-1:2
        merge = true
        for s in allstrides
            if s[i] != dims[i-1]*s[i-1]
                merge = false
                break
            end
        end
        if merge
            dims = TupleTools.setindex(dims, dims[i-1]*dims[i], i-1)
            dims = TupleTools.setindex(dims, 1, i)
        end
    end
    _mapreducedim2!(f, op, dims, allstrides, arrays)
    return
end

function _mapreducedim2!(f::F, op::G, dims::NTuple{N,Int}, strides::NTuple{M, NTuple{N,Int}}, arrays::NTuple{M,AbstractStridedView}) where {F,G,N,M}
    # Level 2: recursively delete dimensions of size 1
    i = findfirst(isequal(1), dims)
    if !(i isa Nothing)
        newdims = TupleTools.deleteat(dims, i)
        newstrides = broadcast(TupleTools.deleteat, strides, (i,))
        _mapreducedim2!(f, op, newdims, newstrides, arrays)
    else
        _mapreducedim_impl!(f, op, dims, strides, arrays)
    end
    return
end

function _mapreducedim_impl!(f::F, op::G, dims::NTuple{N,Int}, strides::NTuple{M, NTuple{N,Int}}, arrays::NTuple{M,AbstractStridedView}) where {F,G,N,M}
    # sort order of loops/dimensions by modelling the importance of each dimension
    g = 8*sizeof(Int) - leading_zeros(M+1) # ceil(Int, log2(M+2)) # to account for the fact that there are M arrays, where the first one is counted with a factor 2
    importance = 2 .* ( 1 .<< (g.*(N .- indexorder(strides[1]))))  # first array is output and is more important by a factor 2
    for k = 2:M
        importance = importance .+ ( 1 .<< (g.*(N .- indexorder(strides[k]))))
    end

    p = TupleTools.sortperm(importance, rev = true)

    dims = TupleTools.getindices(dims, p)
    strides = broadcast(TupleTools.getindices, strides, (p,))
    offsets = map(offset, arrays)

    if all(l -> l<=BLOCKSIZE, broadcast(_length, (dims,), strides))
        _mapreduce_kernel!(f, op, dims, dims, arrays, strides, offsets)
    else
        minstrides = map(min, strides...)
        mincosts = map(a->ifelse(iszero(a), 1, a << 1), minstrides)
        blocks = _computeblocks(dims, mincosts, strides)

        # @show dims, strides, blocks, p
        #
        if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(dims) < Threads.nthreads()*prod(blocks)
            _mapreduce_kernel!(f, op, dims, blocks, arrays, strides, offsets)
        else
            mincosts = mincosts .* .!(iszero.(strides[1]))
            # make cost of dimensions with zero stride in output array (reduction dimensions),
            # so that they are not divided in threading (which would lead to race errors)

            n = Threads.nthreads()
            threadblocks, threadoffsets = _computethreadblocks(n, dims, mincosts, strides, offsets)
            _mapreduce_threaded!(threadblocks, threadoffsets, f, op, blocks, arrays, strides)
        end
    end
    return
end

@noinline function _mapreduce_threaded!(threadblocks, threadoffsets, f::F, op::G, blocks::NTuple{N,Int}, arrays::NTuple{M,AbstractStridedView}, strides::NTuple{M,NTuple{N,Int}}) where {F,G,N,M}
    @inbounds Threads.@threads for i = 1:length(threadblocks)
        _mapreduce_kernel!(f, op, threadblocks[i], blocks, arrays, strides, threadoffsets[i])
    end
end

@generated function _mapreduce_kernel!(f::F, op::G, dims::NTuple{N,Int}, blocks::NTuple{N,Int}, arrays::NTuple{M,AbstractStridedView}, strides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {F,G,N,M}
    blockloopvars = [Symbol("J$i") for i = 1:N]
    blockdimvars = [Symbol("d$i") for i = 1:N]
    innerloopvars = [Symbol("j$i") for i = 1:N]

    stridevars = [Symbol("stride_$(i)_$(j)") for i = 1:N, j = 1:M]
    Ivars = [Symbol("I$j") for j = 1:M]
    Avars = [Symbol("A$j") for j = 1:M]
    pre1 = Expr(:block, [:($(Avars[j]) = arrays[$j]) for j = 1:M]...)
    pre2 = Expr(:block, [:($(stridevars[i,j]) = strides[$j][$i]) for i = 1:N, j=1:M]...)
    pre3 = Expr(:block, [:($(Ivars[j]) = offsets[$j]+1) for j = 1:M]...)

    if G == Nothing
        ex = :(A1[ParentIndex($(Ivars[1]))] = f($([:($(Avars[j])[ParentIndex($(Ivars[j]))]) for j = 2:M]...)))
    else
        ex = :(A1[ParentIndex($(Ivars[1]))] = op(A1[ParentIndex($(Ivars[1]))], f($([:($(Avars[j])[ParentIndex($(Ivars[j]))]) for j = 2:M]...))))
    end
    i = 1
    if N >= 1
        ex = quote
            $(innerloopvars[i]) = 0
            while $(innerloopvars[i]) < $(blockdimvars[i])
                $ex
                $(Expr(:block, [:($(Ivars[j]) += $(stridevars[i,j])) for j = 1:M]...))
                $(innerloopvars[i]) += 1
                $(Expr(:simdloop, true))  # Mark loop as SIMD loop
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

function indexorder(strides::NTuple{N,Int}) where {N}
    # returns order such that strides[i] is the order[i]th smallest element of strides, not counting zero strides
    # zero strides have order N
    return ntuple(Val(N)) do i
        si = strides[i]
        si == 0 && return N
        k = 1
        for s in strides
            if s != 0 && s < si
                k += 1
            end
        end
        return k
    end
end

_length(dims::Tuple, strides::Tuple) = ifelse(iszero(strides[1]), 1, dims[1]) * _length(Base.tail(dims), Base.tail(strides))
_length(dims::Tuple{}, strides::Tuple{}) = 1
_maxlength(dims::Tuple, strides::Tuple{Vararg{Tuple}}) = maximum(broadcast(_length, (dims,), strides))

function _lastargmax(t::Tuple)
    i = 1
    for j = 2:length(t)
        @inbounds if t[j] >= t[i]
            i = j
        end
    end
    return i
end

_computeblocks(dims::Tuple{}, costs::Tuple{}, strides::Tuple{Vararg{Tuple{}}}, blocksize::Int = BLOCKSIZE) = ()
function _computeblocks(dims::NTuple{N,Int}, costs::NTuple{N,Int}, strides::Tuple{Vararg{NTuple{N,Int}}}, blocksize::Int = BLOCKSIZE) where {N}
    if _maxlength(dims, strides) <= blocksize
        return dims
    elseif all(isequal(1), map(TupleTools.argmin, strides))
        return (dims[1], _computeblocks(tail(dims), tail(costs), map(tail, strides), div(blocksize, dims[1]))...)
    elseif blocksize == 0
        return ntuple(n->1, StaticLength(N))
    else
        blocks = dims
        while _maxlength(blocks, strides) >= 2*blocksize
            i = _lastargmax((blocks .- 1) .* costs)
            blocks = TupleTools.setindex(blocks, (blocks[i]+1)>>1, i)
        end
        while _maxlength(blocks, strides) > blocksize
            i = _lastargmax((blocks .- 1) .* costs)
            blocks = TupleTools.setindex(blocks, blocks[i]-1, i)
        end
        return blocks
    end
end

function _computethreadblocks end
let factors = reverse!(simpleprimefactorization(Threads.nthreads()))
    global _computethreadblocks
    @inbounds function _computethreadblocks(n::Int, dims::NTuple{N,Int}, costs::NTuple{N,Int}, strides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {N,M}
        threadblocks = [dims]
        threadoffsets = [offsets]
        for k in factors
            l = length(threadblocks)
            for j = 1:l
                dims = popfirst!(threadblocks)
                offsets = popfirst!(threadoffsets)
                i = _lastargmax((dims .- (k-1)) .* costs)
                ndi = div(dims[i], k)
                newdims = setindex(dims, ndi, i)
                stridesi = getindex.(strides, i)
                for m = 1:k-1
                    push!(threadblocks, newdims)
                    push!(threadoffsets, offsets)
                    offsets = offsets .+ ndi .* stridesi
                end
                ndi = dims[i]-(k-1)*ndi
                newdims = setindex(dims, ndi, i)
                push!(threadblocks, newdims)
                push!(threadoffsets, offsets)
            end
        end
        return threadblocks, threadoffsets
    end
end
