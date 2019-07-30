const BLOCKSIZE = 1024

Base.mapreduce(f, op, A::AbstractStridedView; dims=:, kw...) =
    Base._mapreduce_dim(f, op, kw.data, A, dims)

Base._mapreduce_dim(f, op, nt::NamedTuple{(:init,)}, A::AbstractStridedView, ::Colon) =
    _mapreduce(f, op, A, nt)
Base._mapreduce_dim(f, op, ::NamedTuple{()}, A::AbstractStridedView, ::Colon) =
    _mapreduce(f, op, A)

Base._mapreduce_dim(f, op, nt::NamedTuple{(:init,)}, A::AbstractStridedView, dims) =
    Base.mapreducedim!(f, op, Base.reducedim_initarray(A, dims, nt.init), A)
Base._mapreduce_dim(f, op, ::NamedTuple{()}, A::AbstractStridedView, dims) =
    Base.mapreducedim!(f, op, Base.reducedim_init(f, op, A, dims), A)

function Base.map(f::F, a1::AbstractStridedView{<:Any,N},
        A::Vararg{AbstractStridedView{<:Any,N}}) where {F,N}
    T = Base.promote_eltype(a1, A...)
    map!(f, similar(a1, T), a1, A...)
end

function Base.map!(f::F, b::AbstractStridedView{<:Any,N}, a1::AbstractStridedView{<:Any,N},
        A::Vararg{AbstractStridedView{<:Any,N}}) where {F,N}
    dims = size(b)

    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch())
    for a in A
        size(a) == dims || throw(DimensionMismatch())
    end

    any(isequal(0), dims) && return b # don't do anything

    _mapreduce_fuse!(f, nothing, nothing, dims, (b, a1, A...))

    return b
end

function _mapreduce(f, op, A::AbstractStridedView{T}, nt = nothing) where {T}
    if length(A) == 0
        b = Base.mapreduce_empty(f, op, T)
        return nt === nothing ? b : op(b, nt.init)
    end

    dims = size(A)
    a = Base.mapreduce_first(f, op, first(A))
    a2 = nt === nothing ? a : op(a, nt.init)
    out = similar(A, typeof(a2), (1,))
    if nt === nothing
        _init_reduction!(out, f, op, a)
    else
        out[ParentIndex(1)] = nt.init
    end
    _mapreducedim!(f, op, nothing, dims, (sreshape(out, one.(dims)), A))
    return out[ParentIndex(1)]
end

@inline function Base.mapreducedim!(f, op, b::AbstractStridedView{<:Any,N},
        a1::AbstractStridedView{<:Any,N}, A::Vararg{AbstractStridedView{<:Any,N}}) where {N}
    outdims = size(b)
    dims = map(max, outdims, map(max, map(size, (a1, A...))...))

    # Check dimesions
    Broadcast.check_broadcast_axes(map(Base.OneTo, dims), b, a1, A...)

    _mapreducedim!(f, op, nothing, dims, (b, a1, A...))
end

function _mapreducedim!(@nospecialize(f), @nospecialize(op), @nospecialize(initop),
        dims::Dims, arrays::Tuple{Vararg{AbstractStridedView}})
    any(isequal(0), dims) && return arrays[1] # don't do anything

    _mapreduce_fuse!(f, op, initop, dims, promoteshape(dims, arrays...))

    return arrays[1]
end

function _mapreduce_fuse!(@nospecialize(f), @nospecialize(op), @nospecialize(initop),
        dims::Dims, arrays::Tuple{Vararg{AbstractStridedView}})
    # Fuse dimensions if possible: assume that at least one array, e.g. the output
    # array in arrays[1], has its strides sorted
    allstrides = map(strides, arrays)
    @inbounds for i = length(dims):-1:2
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
    _mapreduce_order!(f, op, initop, dims, allstrides, arrays)
end

function _mapreduce_order!(@nospecialize(f), @nospecialize(op), @nospecialize(initop),
        dims, strides, arrays)
    M = length(arrays)
    N = length(dims)
    # sort order of loops/dimensions by modelling the importance of each dimension
    g = 8*sizeof(Int) - leading_zeros(M+1) # ceil(Int, log2(M+2)) # to account for the fact
    # that there are M arrays, where the first one is counted with a factor 2
    importance = 2 .* ( 1 .<< (g.*(N .- indexorder(strides[1]))))  # first array is output
    # and is more important by a factor 2
    for k = 2:M
        importance = importance .+ ( 1 .<< (g.*(N .- indexorder(strides[k]))))
    end

    importance = importance .* (dims .> 1) # put dims 1 at the back
    p = TupleTools.sortperm(importance, rev = true)
    dims = TupleTools.getindices(dims, p)
    strides = broadcast(TupleTools.getindices, strides, (p,))
    _mapreduce_block!(f, op, initop, dims, strides, arrays)
end

function _mapreduce_block!(@nospecialize(f), @nospecialize(op), @nospecialize(initop),
        dims, strides, arrays)
    offsets = map(offset, arrays)

    if all(l -> l<=BLOCKSIZE, broadcast(_length, (dims,), strides))
        _mapreduce_kernel!(f, op, initop, dims, dims, arrays, strides, offsets)
    else
        minstrides = map(min, strides...)
        mincosts = map(a->ifelse(iszero(a), 1, a << 1), minstrides)
        blocks = _computeblocks(dims, mincosts, strides)

        if Threads.nthreads() == 1 || Threads.in_threaded_loop[] || prod(dims) < BLOCKSIZE
            _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, offsets)
        elseif _length(dims, strides[1]) == 1 # complete reduction
            T = eltype(arrays[1])
            spacing = isbitstype(T) ? min(1, div(64, sizeof(T))) : 1# to avoid false sharing
            threadblocks, threadoffsets =
                _computethreadblocks(dims, mincosts, strides, offsets)
            threadedout = similar(arrays[1], spacing*length(threadblocks))
            a = arrays[1][ParentIndex(1)]
            if initop !== nothing
                a = initop(a)
            end
            _init_reduction!(threadedout, f, op, a)
            for i = 1:length(threadoffsets)
                threadoffsets[i] = (spacing*(i-1), Base.tail(threadoffsets[i])...)
            end
            _mapreduce_threaded!(threadblocks, threadoffsets, f, op, nothing, blocks,
                (threadedout, Base.tail(arrays)...), strides)
            for i = 1:length(threadblocks)
                a = op(a, threadedout[(i-1)*spacing+1])
            end
            arrays[1][ParentIndex(1)] = a
        else
            mincosts = mincosts .* .!(iszero.(strides[1]))
            # make cost of dimensions with zero stride in output array (reduction
            # dimensions), so that they are not divided in threading (which would lead to
            # race errors)

            threadblocks, threadoffsets =
                _computethreadblocks(dims, mincosts, strides, offsets)
            _mapreduce_threaded!(threadblocks, threadoffsets, f, op, initop, blocks,
                arrays, strides)
        end
    end
    return
end

_init_reduction!(out, f, op::Union{typeof(+),typeof(Base.add_sum)}, a) = fill!(out, zero(a))
_init_reduction!(out, f, op::Union{typeof(*),typeof(Base.mul_prod)}, a) = fill!(out, one(a))
_init_reduction!(out, f, op::typeof(min), a) = fill!(out, a)
_init_reduction!(out, f, op::typeof(max), a) = fill!(out, a)
_init_reduction!(out, f, op::typeof(&), a) = fill!(out, true)
_init_reduction!(out, f, op::typeof(|), a) = fill!(out, false)
_init_reduction!(out, f, op, a) = op(a,a) == a ? fill!(out, a) : error("unknown reduction; incompatible with multithreading")

@noinline function _mapreduce_threaded!(threadblocks, threadoffsets, f, op, initop, blocks, arrays, strides)
    @inbounds Threads.@threads for i = 1:length(threadblocks)
        _mapreduce_kernel!(f, op, initop, threadblocks[i], blocks, arrays, strides,
            threadoffsets[i])
    end
end

@generated function _mapreduce_kernel!(@nospecialize(f), @nospecialize(op),
        @nospecialize(initop), dims::NTuple{N,Int}, blocks::NTuple{N,Int},
        arrays::NTuple{M,AbstractStridedView}, strides::NTuple{M,NTuple{N,Int}},
        offsets::NTuple{M,Int}) where {N,M}

    # many variables
    blockloopvars = Array{Symbol}(undef, N)
    blockdimvars = Array{Symbol}(undef, N)
    innerloopvars = Array{Symbol}(undef, N)
    initblockdimvars = Array{Symbol}(undef, N)
    initvars = Array{Symbol}(undef, N+1)
    for i = 1:N
        blockloopvars[i] = Symbol(:J,i)
        blockdimvars[i] = Symbol(:d,i)
        innerloopvars[i] = Symbol(:j,i)
        initblockdimvars[i] = Symbol(:d′,i)
        initvars[i] = Symbol(:init,i)
    end
    initvars[N+1] = Symbol(:init,N+1)

    Ivars = Array{Symbol}(undef, M)
    Avars = Array{Symbol}(undef, M)
    stridevars = Array{Symbol}(undef, (N,M))
    for j = 1:M
        Ivars[j] = Symbol(:I,j)
        Avars[j] = Symbol(:A,j)
        for i = 1:N
            stridevars[i,j] = Symbol(:stride_,i,:_,j)
        end
    end

    # useful expressions
    pre1ex = Expr(:block)
    pre2ex = Expr(:block)
    pre3ex = Expr(:block)
    for j = 1:M
        push!(pre1ex.args, :($(Avars[j]) = arrays[$j]))
        push!(pre3ex.args, :($(Ivars[j]) = offsets[$j]+1))
        for i = 1:N
            push!(pre2ex.args, :($(stridevars[i,j]) = strides[$j][$i]))
        end
    end

    fcallex = Expr(:call, :f)
    for j = 2:M
        push!(fcallex.args, :($(Avars[j])[ParentIndex($(Ivars[j]))]))
    end
    lhsex = :(A1[ParentIndex($(Ivars[1]))])

    stepstride1ex = Vector{Expr}(undef, N)
    stepstride2ex = Vector{Expr}(undef, N)
    returnstride1ex = Vector{Expr}(undef, N)
    returnstride2ex = Vector{Expr}(undef, N)
    for i = 1:N
        stepstride1ex[i] = :($(Ivars[1]) += $(stridevars[i,1]))
        returnstride1ex[i] = :($(Ivars[1]) -=  $(blockdimvars[i]) * $(stridevars[i,1]))
        stepex = Expr(:block)
        returnex = Expr(:block)
        for j = 2:M
            push!(stepex.args, :($(Ivars[j]) += $(stridevars[i,j])))
            push!(returnex.args, :($(Ivars[j]) -=  $(blockdimvars[i]) * $(stridevars[i,j])))
        end
        stepstride2ex[i] = stepex
        returnstride2ex[i] = returnex
    end

    outerstepstrideex = Vector{Expr}(undef, N)
    outerreturnstrideex = Vector{Expr}(undef, N)
    for i = 1:N
        stepex = Expr(:block)
        returnex = Expr(:block)
        for j = 1:M
            push!(stepex.args, :($(Ivars[j]) +=  $(blockdimvars[i]) * $(stridevars[i,j])))
            push!(returnex.args, :($(Ivars[j]) -=  dims[$i] * $(stridevars[i,j])))
        end
        outerstepstrideex[i] = stepex
        outerreturnstrideex[i] = returnex
    end

    if op == Nothing
        ex = Expr(:(=), lhsex, fcallex)
        exa = Expr(:(=), :a, fcallex)
    else
        ex = Expr(:(=), lhsex, Expr(:call, :op, lhsex, fcallex))
        exa = Expr(:(=), :a, Expr(:call, :op, :a, fcallex))
    end
    i = 1
    if N >= 1
        ex = quote
            if $(stridevars[1,1]) == 0 # explicitly hoist A1[I1] out of loop
                a = $lhsex
                @simd for $(innerloopvars[i]) = Base.OneTo($(blockdimvars[i]))
                    $exa
                    $(stepstride2ex[i])
                end
                $lhsex = a
                $(returnstride2ex[i])
            else
                @simd for $(innerloopvars[i]) = Base.OneTo($(blockdimvars[i]))
                    $ex
                    $(stepstride1ex[i])
                    $(stepstride2ex[i])
                end
                $(returnstride1ex[i])
                $(returnstride2ex[i])
            end
        end
    end
    for outer i = 2:N
        ex = quote
            for $(innerloopvars[i]) = Base.OneTo($(blockdimvars[i]))
                $ex
                $(stepstride1ex[i])
                $(stepstride2ex[i])
            end
            $(returnstride1ex[i])
            $(returnstride2ex[i])
        end
    end

    if initop !== Nothing
        initex = Expr(:(=), lhsex, Expr(:call, :initop, lhsex))
        i = 1
        if N >= 1
            initex = quote
                $(initblockdimvars[i]) = ifelse($(stridevars[i,1]) == 0, 1, $(blockdimvars[i]))
                @simd for $(innerloopvars[i]) in Base.OneTo($(initblockdimvars[i]))
                    $initex
                    $(stepstride1ex[i])
                end
                $(Ivars[1]) -=  $(initblockdimvars[i]) * $(stridevars[i,1])
            end
        end
        for outer i = 2:N
            initex = quote
                $(initblockdimvars[i]) = ifelse($(stridevars[i,1]) == 0, 1, $(blockdimvars[i]))
                for $(innerloopvars[i]) in Base.OneTo($(initblockdimvars[i]))
                    $initex
                    $(stepstride1ex[i])
                end
                $(Ivars[1]) -=  $(initblockdimvars[i]) * $(stridevars[i,1])
            end
        end
        ex = quote
            if $(initvars[1])
                $initex
            end
            $ex
        end
    end

    if initop === Nothing
        for outer i = 1:N
            ex = quote
                for $(blockloopvars[i]) = 1:blocks[$i]:dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i]-$(blockloopvars[i])+1)
                    $ex
                    $(outerstepstrideex[i])
                end
                $(outerreturnstrideex[i])
            end
        end
        ex = quote
            $pre1ex
            $pre2ex
            $pre3ex
            @inbounds $ex
            return A1
        end
    else
        for outer i = 1:N
            ex = quote
                $(initvars[i]) = $(initvars[i+1])
                for $(blockloopvars[i]) = 1:blocks[$i]:dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i]-$(blockloopvars[i])+1)
                    $ex
                    $(initvars[i]) &= $(stridevars[i,1]) > 0
                    $(outerstepstrideex[i])
                end
                $(outerreturnstrideex[i])
            end
        end
        ex = quote
            $pre1ex
            $pre2ex
            $pre3ex
            $(initvars[N+1]) = true
            @inbounds $ex
            return A1
        end
    end
    return ex
end

function indexorder(strides::NTuple{N,Int}) where {N}
    # returns order such that strides[i] is the order[i]th smallest element of strides, not counting zero strides
    # zero strides have order N
    return ntuple(Val(N)) do i
        si = strides[i]
        si == 0 && return 1
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
    elseif all(isequal(1), first.(indexorder.(strides)))
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

@inbounds function _computethreadblocks(dims::NTuple{N,Int}, costs::NTuple{N,Int}, strides::NTuple{M,NTuple{N,Int}}, offsets::NTuple{M,Int}) where {N,M}
    threadblocks = [dims]
    threadoffsets = [offsets]
    for k in factors
        l = length(threadblocks)
        for j = 1:l
            dims = popfirst!(threadblocks)
            offsets = popfirst!(threadoffsets)
            i = _lastargmax((dims .- (k-1)) .* costs)
            if costs[i] == 0
                push!(threadblocks, dims)
                push!(threadoffsets, offsets)
                return threadblocks, threadoffsets
            end
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
