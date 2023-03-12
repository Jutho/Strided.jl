using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView

Random.seed!(1234)

println("Base.Threads.nthreads() =  $(Base.Threads.nthreads())")

println("Running tests single-threaded:")
Strided.disable_threads()
include("othertests.jl")
include("blasmultests.jl")

println("Running tests multi-threaded:")
Strided.enable_threads()
Strided.set_num_threads(Base.Threads.nthreads()+1)
include("othertests.jl")
include("blasmultests.jl")

Strided.enable_threaded_mul()
include("blasmultests.jl")
Strided.disable_threaded_mul()

using Aqua
Aqua.test_all(Strided)