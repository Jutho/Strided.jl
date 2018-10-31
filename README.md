# Strided.jl

[![Build Status](https://travis-ci.org/Jutho/Strided.jl.svg?branch=master)](https://travis-ci.org/Jutho/Strided.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Coverage Status](https://coveralls.io/repos/Jutho/Strided.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Jutho/Strided.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/Strided.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/Strided.jl?branch=master)

A Julia package for working more efficiently with strided arrays, i.e. dense arrays
whose memory layout has a fixed stride along every dimension. Strided.jl does not
make any assumptions about the strides (such as stride 1 along first dimension, or
monotonously increasing strides) and provides multithreaded and cache friendly
implementations for mapping, reducing, broadcasting such arrays, as well as taking
views, reshaping and permuting dimensions.

---

# Examples

Running Julia with a single thread

```julia
julia> using Strided

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  145.214 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  56.189 ms (6 allocations: 352 bytes)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  2.449 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  1.459 ms (5 allocations: 288 bytes)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  22.493 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  22.240 ms (10 allocations: 480 bytes)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  5.203 ms (2 allocations: 128 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  2.201 ms (4 allocations: 320 bytes)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  21.863 ms (32 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  8.495 ms (9 allocations: 640 bytes)
```
And now with `export JULIA_NUM_THREADS = 4`
```julia
julia> using Strided

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  146.618 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  30.355 ms (12 allocations: 912 bytes)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  2.030 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  808.874 μs (11 allocations: 784 bytes)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  21.971 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  5.811 ms (16 allocations: 1.05 KiB)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  5.334 ms (2 allocations: 128 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  1.192 ms (10 allocations: 928 bytes)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  22.465 ms (32 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  2.796 ms (15 allocations: 1.44 KiB)
```

# Design principles

## `StridedView`

Strided.jl is centered around the type `StridedView`, which provides a view into a parent
array of type `DenseArray` such that the resulting view is strided, i.e. any dimension
has an associated stride, such that e.g.
```julia
getindex(A, i₁, i₂, i₃, ...) = A.op(A.parent[offset + 1 + (i₁-1)*s₁ + (i₂-1)*s₂ + (i₃-1)*s₃ + ...])
```
with `sⱼ = stride(A, iⱼ)`. There are no further assumptions on the strides, e.g. they are
not assumed to be monotonously increasing or have `s₁ == 1`. Furthermore, `A.op` can be
any of the operations `identity`, `conj`, `transpose` or `adjoint` (the latter two are
equivalent to the former two if `eltype(A) <: Number`). Since these operations are their own
inverse, they are also used in the corresponding `setindex!`.

This definition enables a `StridedView` to be lazy (i.e. returns just another `StridedView` over
the same parent data) under application of `conj`, `transpose`, `adjoint`, `permutedims` and
indexing (`getindex`) with `Union{Integer, Colon, AbstractRange{<:Integer}}` (a.k.a slicing).

Furthermore, the strided structure can be retained under certain `reshape` operations, but not
all of them. Any dimension can always be split into smaller dimensions, but two subsequent
dimensions `i` and `i+1` can only be joined if `stride(A,i+1) == size(A,i)*stride(A,i)`. Instead
of overloading `reshape`, Strided.jl provides a separate function `sreshape` which returns a
`StridedView` over the same parent data, or throws a runtime error if this is impossible.

## Broadcasting and `map(reduce)`

Whenever an expression only contains `StridedView`s and non-array objects (scalars),
overloaded methods for broadcasting and functions as `map(!)` and `mapreduce` are used
that exploit the known strided structure in order to evaluate the result in a more
efficient way, at least for sufficiently large arrays where the overhead of the extra preparatory
work is negligible. In particular, this involves choosing a blocking strategy and loop order
that aims to avoid cache misses. This matters in particular if some of the `StridedView`s
involved have strides which are not monotonously increasing, e.g. if `transpose`, `adjoint` or
`permutedims` has been applied. The fact that the latter also acts lazily (whereas it creates
a copy of the data in Julia base) can potentially provide a further speedup.

Furthermore, these optimized methods are implemented with support for multithreading. Thus,
if `Threads.nthreads() > 1` and the arrays involved are sufficiently large, performance can be
boosted even for plain arrays with a strictly sequential memory layout, provided that the
broadcast operation is compute bound and not memory bound (i.e. the broadcast function is
sufficienlty complex).

## The `@strided` macro annotation
Rather than manually wrapping every array in a `StridedView`, there is the macro annotation
`@strided some_expression`, which will wrap all `DenseArray`s appearing in `some_expression` in
a `StridedView`. Note that, because `StridedView`s behave lazily under indexing with ranges,
this acts similar to the `@views` macro in Julia Base, i.e. there is no need to use a view.

The macro `@strided` acts as a contract, i.e. the user ensures that all array manipulations in
the following expressions will preserve the strided structure. Therefore, `reshape` and `view`
are are replaced by `sreshape` and `sview` respectively. As mentioned above, `sreshape` will
throw an error if the requested new shape is incompatible with preserving the strided structure.
The function `sview` is only defined for index arguments which are ranges, `Int`s or `Colon` (`:`),
and will thus also throw an error if indexed by anything else.

## `StridedView` versus `StridedArray` and BLAS/LAPACK compatibility

`StridedArray` is a union type to denote arrays with a strided structure in Julia Base. Because
of its definition as a type union rather than an abstract type, it is impossible to have user
types be recognized as `StridedArray`. This is rather unfortunate, since dispatching to BLAS
and LAPACK routines is based on `StridedArray`. As a consequence, `StridedView` will not fall
back to BLAS or LAPACK by default. Currently, only matrix multiplication is overloaded so as
to fall back to BLAS (i.e. `gemm!`) if possible. In general, one should not attempt use e.g.
matrix factorizations or other lapack operations within the `@strided` context. Support for
this is on the TODO list. Some BLAS inspired methods (`axpy!`, `axpby!`, scalar multiplication
via `mul!`, `rmul!` or `lmul!`) are however overloaded by relying on the optimized yet generic
`map!` implementation.

`StridedView`s can currently only be created with certainty from `DenseArray` (typically just
`Array` in Julia Base). For `Base.SubArray` or `Base.ReshapedArray` instances, the `StridedView`
constructor will first act on the underlying parent array, and then try to mimic the corresponding
view or reshape operation using `sview` and `sreshape`. These, however, are more limited then
their Base counterparts (because they need to guarantee that the result still has a strided
memory layout with respect to the new dimensions), so an error can result. However, this approach
can also succeed in creating `StridedView` wrappers around combinations of `view` and `reshape`
that are not recognised as `Base.StridedArray`. For example, `reshape(view(randn(40,40), 1:36, 1:20), (6,6,5,4))`
is not a `Base.StridedArrray`, and indeed, it cannot statically be inferred to be strided, from
only knowing the argument types provided to `view` and `reshape`. For example, the similarly looking
`reshape(view(randn(40,40), 1:36, 1:20), (6,3,10,4))` is not strided. The `StridedView` constructor
will try to act on both, and yield a runtime error in the second case. Note that `Base.ReinterpretArray`
is currently not supported.

Note again that, unlike `StridedArray`s, `StridedView`s behave lazily (i.e. still produce a view
on the same parent array) under `permutedims` and regular indexing with ranges.

## `UnsafeStridedView` and `@unsafe_strided`
Based on the work of [UnsafeArrays.jl](https://github.com/oschulz/UnsafeArrays.jl) there is
also an `UnsafeStridedView`, which references the parent array via a pointer, and therefore
is itself a stack allocated `struct` (i.e. `isbitstype(UnsafeStridedView{...})` is true).

It behaves in all respects the same as `StridedView` (they are both subtypes of `AbstractStridedView`),
except that by itself it does not keep a reference to the parent array in a way that is visible
to Julia's garbage collector. It can therefore not be the return value of an operation (in
particular `similar(::UnsafeStridedView, ...) -> ::StridedView`) and an explicit reference to
the parent array needs to be kept alive. Furthermore, `UnsafeStridedView` wrappers can only
be created of `AbstractArray{T}` instances with `isbitstype(T)`.

There is a corresponding `@unsafe_strided` macro annotation. However, in this case the arrays
in the expression need to be identified explicitly as
```julia
@unsafe_strided A₁ A₂ ... some_expression
```
because this will be translated into the expression
```julia
GC.@preserve A₁ A₂ ... let A₁ = UnsafeStridedView(A₁), A₂ = ...
    some_expression
end
```

# Planned features / wish list

*   Support for `GPUArray`s with dedicated GPU kernels?
