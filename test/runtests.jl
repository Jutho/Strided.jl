using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView, UnsafeStridedView

Random.seed!(1234)

println("Running tests single-threaded:")
Strided.set_num_threads(1)
include("alltests.jl")

println("Running tests multi-threaded:")
Strided.set_num_threads(Base.Threads.nthreads())
include("alltests.jl")
