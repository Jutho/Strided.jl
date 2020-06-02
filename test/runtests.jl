using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView, UnsafeStridedView

Random.seed!(1234)

println("Base.Threads.nthreads() =  $(Base.Threads.nthreads())")

println("Running tests single-threaded:")
Strided.disable_threads()
include("othertests.jl")
include("multests.jl")

println("Running tests multi-threaded:")
Strided.enable_threads()
include("othertests.jl")
include("multests.jl")

Strided.enable_threaded_mul()
include("multests.jl")
