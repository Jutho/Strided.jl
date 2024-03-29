using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted

struct StridedArrayStyle{N} <: AbstractArrayStyle{N}
end

Broadcast.BroadcastStyle(::Type{<:StridedView{<:Any,N}}) where {N} = StridedArrayStyle{N}()

StridedArrayStyle(::Val{N}) where {N} = StridedArrayStyle{N}()
StridedArrayStyle{M}(::Val{N}) where {M,N} = StridedArrayStyle{N}()

Broadcast.BroadcastStyle(a::StridedArrayStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::StridedArrayStyle{N}, a::DefaultArrayStyle) where {N}
    return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(::StridedArrayStyle{N},
                                  ::Broadcast.Style{Tuple}) where {N}
    return DefaultArrayStyle{N}()
end

function Base.similar(bc::Broadcasted{<:StridedArrayStyle{N}}, ::Type{T}) where {N,T}
    return StridedView(similar(convert(Broadcasted{DefaultArrayStyle{N}}, bc), T))
end

Base.dotview(a::StridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} = getindex(a, I...)

# Broadcasting implementation
@inline function Base.copyto!(dest::StridedView{<:Any,N},
                              bc::Broadcasted{StridedArrayStyle{N}}) where {N}
    # convert to map

    # flatten and only keep the StridedView arguments
    # promote StridedView to have same size, by giving artificial zero strides
    stridedargs = promoteshape(size(dest), capturestridedargs(bc)...)
    c = make_capture(bc)
    _mapreduce_fuse!(c, nothing, nothing, size(dest), (dest, stridedargs...))
    return dest
end

const WrappedScalarArgs = Union{AbstractArray{<:Any,0},Ref{<:Any}}

@inline function capturestridedargs(t::Broadcasted, rest...)
    return (capturestridedargs(t.args...)..., capturestridedargs(rest...)...)
end
@inline capturestridedargs(t::StridedView, rest...) = (t, capturestridedargs(rest...)...)
@inline capturestridedargs(t, rest...) = capturestridedargs(rest...)
@inline capturestridedargs() = ()

# broadcast by promoting size 1 dimensions to full size
# making the stride in that dimension zero
function promoteshape(sz::Dims, a1::StridedView, As...)
    return (promoteshape1(sz, a1), promoteshape(sz, As...)...)
end
promoteshape(sz::Dims) = ()
function promoteshape1(sz::Dims{N}, a::StridedView) where {N}
    newstrides = ntuple(Val(N)) do d
        if size(a, d) == sz[d]
            stride(a, d)
        elseif size(a, d) == 1
            0
        else
            throw(DimensionMismatch("array could not be broadcasted to match destination"))
        end
    end
    return StridedView(a.parent, sz, newstrides, a.offset, a.op)
end

struct CaptureArgs{F,Args<:Tuple}
    f::F
    args::Args
end
struct Arg
end

# construct CaptureArgs
@inline function make_capture(bc::Broadcasted)
    args = make_tcapture(bc.args)
    return CaptureArgs(bc.f, args)
end
@inline make_tcapture(t::Tuple{}) = t
@inline make_tcapture(t::Tuple) = (make_capture(t[1]), make_tcapture(tail(t))...)
@inline make_capture(a::WrappedScalarArgs) = a[]
@inline make_capture(a::StridedView) = Arg()
@inline make_capture(a) = a

# Evaluate CaptureArgs
(c::CaptureArgs)(vals...) = consume(c, vals)[1]
@inline function consume(c::CaptureArgs{F,Args}, vals) where {F,Args}
    args, newvals = t_consume(c.args, vals)
    return c.f(args...), newvals
end
@inline consume(a::Arg, vals::Tuple) = vals[1], tail(vals)
@inline consume(a, vals) = a, vals
@inline t_consume(t::Tuple{}, vals) = t, vals
@inline function t_consume(t::Tuple, vals)
    t1, newvals1 = consume(t[1], vals)
    ttail, newvals = t_consume(tail(t), newvals1)
    return (t1, ttail...), newvals
end
