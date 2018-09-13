macro strided(ex)
    esc(_strided(ex))
end

function _strided(ex::Expr)
    if ex.head == :call && ex.args[1] isa Symbol
        if ex.args[1] == :reshape
            return Expr(:call, :(Strided.sreshape), _strided.(ex.args[2:end])...)
        else
            return Expr(:call, ex.args[1], _strided.(ex.args[2:end])...)
        end
    elseif ex.head == :(=) && ex.args[1] isa Symbol
        return Expr(:(=), ex.args[1], Expr(:call, :(Strided.maybeunstrided), _strided(ex.args[2])))
    else
        return Expr(ex.head, _strided.(ex.args)...)
    end
end
const exclusionlist = Symbol[:(:)]
_strided(ex::Symbol) =  ex in exclusionlist ? ex : Expr(:call, :(Strided.maybestrided), ex)
_strided(ex) = ex

maybestrided(A::DenseArray) = StridedView(A)
maybestrided(A) = A
maybeunstrided(A::StridedView) = A.parent
maybeunstrided(A) = A

macro unsafe_strided(args...)
    syms = args[1:end-1]
    ex = _strided(args[end])
    all(isa(s, Symbol) for s in syms) || error("The first arguments to `@unsafe_strided` must be variable names")
    ex = Expr(:let, Expr(:block, [:($s = Strided.UnsafeStridedView($s)) for s in syms]...), ex)
    return esc(:(GC.@preserve $(syms...) $ex))
end

# macro sfor(args...)
#     syms = args[1:end-1]
#     all(isa(s, Symbol) for s in syms) || error("The first arguments to `@sfor` must be variable names that will be usek")
#     ex = args[end]
#     ex = _sfor(syms, ex)
# end
