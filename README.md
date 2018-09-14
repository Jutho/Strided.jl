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
views and reshaping and permuting dimensions.

---

# Examples
```julia
using Strided
using BenchmarkTools

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
`StridedView` over the same parent data, or throws an error if this is impossible.

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

Furthermore, these optimized methods are implemented with support for multithreading. Thus if `Threads.nthreads() > 1` and the arrays involved are sufficiently large, performance can be
boosted even for plain arrays with a strictly sequential memory layout.

## The `@strided` macro annotation
Rather than manually wrapping every array in a `StridedView`, there is the macro annotation
`@strided some_expression`, which will wrap all `DenseArray`s appearing in this expression in
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
this is on the TODO list. Some BLAS methods (`axpy!`, `axpby!`, scalar multiplication via
`mul!`, `rmul!` or `lmul!`) are however overloaded by relying on the optimized yet generic
`map!` implementation.

Vice versa, `StridedView`s can currently only be created from `DenseArray` (typically just `Array`
in Julia Base). While other `StridedArray`s, e.g. strided `SubArray`s or `ReshapedArray`s, could
in principle be converted to `StridedView`, this is currently not supported. It is better to first
wrap the parent array in a `StridedView` and only then apply slicing, reshaping and permuting dimension.

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
the parent array needs to be kept alive.

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
