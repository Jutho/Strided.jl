using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView, UnsafeStridedView

Random.seed!(1234)

@testset "construction of $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A1 = randn(T1, (60,60))
        B1 = SV(A1)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A1) == op1(B1) == SV(op1(A1))
            else
                @test op1(A1) == op1(B1)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A1)) == op2(op1(B1))
            end
        end

        A2 = view(A1, 1:36, 1:20)
        B2 = SV(A2)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A2) == op1(B2) == SV(op1(A2))
            else
                @test op1(A2) == op1(B2)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A2)) == op2(op1(B2))
            end
        end

        A3 = reshape(A1, 360, 10)
        B3 = SV(A3)
        @test size(A3) == size(B3)
        @test strides(A3) == strides(B3)
        @test stride(A3, 1) == stride(B3, 1)
        @test stride(A3, 2) == stride(B3, 2)
        @test stride(A3, 3) == stride(B3, 3)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A3) == op1(B3) == SV(op1(A3))
            else
                @test op1(A3) == op1(B3)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A3)) == op2(op1(B3))
            end
        end

        A4 = reshape(view(A1, 1:36, 1:20), (6,6,5,4))
        B4 = SV(A4)
        for op1 in (identity, conj)
            @test op1(A4) == op1(B4)
            for op2 in (identity, conj)
                @test op2(op1(A4)) == op2(op1(B4))
            end
        end

        A5 = reshape(view(A1, 1:36, 1:20), (6,120))
        @test_throws Strided.ReshapeException SV(A5)

        A6 = [randn(T1, (5,5)) for i=1:5, j=1:5]
        if SV == UnsafeStridedView
            @test_throws AssertionError SV(A6)
        else
            B6 = SV(A6)
            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A6) == op1(B6) == SV(op1(A6))
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A6)) == op2(op1(B6))
                end
            end
        end

        if SV == StridedView
            C3 = UnsafeStridedView(B3)
            D3 = StridedView(B3)
            @test D3 === B3
            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A3) == op1(C3)
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A3)) == op2(op1(C3))
                end
            end
        end
    end
end

@testset "elementwise conj, transpose and adjoint" begin
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A = [randn(T,(3,3)) for i=1:5, b=1:4, c=1:3, d=1:2]
        Ac = deepcopy(A)
        B = StridedView(A)

        @test conj(B) == conj(A)
        @test conj(B) == map(conj, B)
        @test map(transpose, B) == map(transpose, A)
        @test map(adjoint, B) == map(adjoint, A)
    end
end

@testset "reshape and permutedims with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A0 = randn(T, 10)
        GC.@preserve A0 begin
            @test permutedims(SV(A0), (1,)) == A0
        end

        @testset "in-place matrix operations" begin
            A1 = randn(T, (1000,1000))
            A2 = similar(A1)
            A1c = copy(A1)
            A2c = copy(A2)
            GC.@preserve A1c A2c begin
                B1 = SV(A1c)
                B2 = SV(A2c)

                @test conj!(A1) == conj!(B1)
                @test adjoint!(A2, A1) == adjoint!(B2, B1)
                @test transpose!(A2, A1) == transpose!(B2, B1)
                @test permutedims!(A2, A1, (2,1)) == permutedims!(B2, B1, (2,1))
            end
        end

        @testset "reshape and permutedims with $N-dimensional arrays" for N = 2:6
            dims = ntuple(n->rand(1:div(60,N)), N)
            A = rand(T, dims)
            Ac = copy(A)
            GC.@preserve Ac begin
                B = SV(Ac)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2
                @test copy(B2) == A2
                @test convert(Array, B2) == A2
            end

            dims = ntuple(n->10, N)
            A = rand(T, dims)
            Ac = copy(A)
            GC.@preserve Ac begin
            B = SV(Ac)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2
                @test copy(B2) == A2
                @test convert(Array, B2) == A2

                B2 = sreshape(B, (2, 5, ntuple(n->10, N-2)..., 5, 2))
                A2 = reshape(A, (2, 5, ntuple(n->10, N-2)..., 5, 2))
                A3 = reshape(copy(A), size(A2))
                @test B2 == A3
                @test B2 == A2
                p = randperm(N+2)
                @test conj(permutedims(B2, p)) == conj(permutedims(A3, p))
            end
        end

        @testset "more reshape" begin
            A = randn(4,0)
            B = SV(A)
            @test_throws DimensionMismatch sreshape(B, (4,1))
            C = sreshape(B, (2, 1, 2, 0, 1))
            @test sreshape(C, (4,0)) == A

            A = randn(4,1,2)
            B = SV(A)
            @test_throws DimensionMismatch sreshape(B, (4,4))
            C = sreshape(B, (2,1,1,4,1,1))
            @test C == reshape(A, (2,1,1,4,1,1))
            @test sreshape(C, (4,1,2)) == A
        end
    end
end

@testset "map, scale!, axpy! and axpby! with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for N = 2:6
            dims = ntuple(n->div(60,N), N)
            R1, R2, R3 = rand(T, dims), rand(T, dims), rand(T, dims)
            GC.@preserve R1 R2 R3 begin
                B1 = permutedims(SV(R1), randperm(N))
                B2 = permutedims(SV(R2), randperm(N))
                B3 = permutedims(SV(R3), randperm(N))
                A1 = convert(Array, B1)
                A2 = convert(Array{T}, B2) # test different converts
                A3 = convert(Array{T,N}, B3)
                C1 = deepcopy(B1)

                @test rmul!(B1, 1//2) ≈ rmul!(A1, 1//2)
                @test lmul!(1//3, B2) ≈ lmul!(1//3, A2)
                @test axpy!(1//3, B1, B2) ≈ axpy!(1//3, A1, A2)
                @test axpby!(1//3, B1, 1//2, B3) ≈ axpby!(1//3, A1, 1//2, A3)
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, B2, B3) ≈ map((x,y,z)->sin(x)+y/exp(-abs(z)), A1, A2, A3)
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, B2, B3) isa StridedView
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, A2, B3) isa Array
            end
        end
    end
end

@testset "views with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = randn(T,(10,10,10,10))
        B = SV(A)
        @test isa(view(B,:,1:5,3,1:5), SV)
        @test isa(view(B,:,[1,2,3],3,1:5), Base.SubArray)
        @test isa(sview(B,:,1:5,3,1:5), SV)
        @test_throws MethodError sview(B,:,[1,2,3],3,1:5)

        @test view(B,:,1:5,3,1:5) == view(A,:,1:5,3,1:5)
        @test view(B,:,1:5,3,1:5) === sview(B,:,1:5,3,1:5) === B[:,1:5,3,1:5]
        @test view(B,:,1:5,3,1:5) == SV(view(A,:,1:5,3,1:5)) 
        @test pointer(view(B,:,1:5,3,1:5)) == pointer(SV(view(A,:,1:5,3,1:5)))
        @test Strided.offset(view(B,:,1:5,3,1:5)) == 2*stride(B,3)
    end
end

@testset "broadcast with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1, R2, R3 = rand(T, (10,)), rand(T, (10,10)), rand(T, (10,10,10))
        GC.@preserve R1 R2 R3 begin
            B1 = SV(R1)
            B2 = permutedims(SV(R2), randperm(2))
            B3 = permutedims(SV(R3), randperm(3))
            A1 = convert(Array, B1)
            A2 = convert(Array{T}, B2)
            A3 = convert(Array{T,3}, B3)

            @test @inferred(B1 .+ sin.(B2 .- 3)) ≈ A1 .+ sin.(A2 .- 3)
            @test @inferred(B2' .* B3 .- Ref(0.5)) ≈ A2' .* A3 .- Ref(0.5)
            @test @inferred(B2' .* B3 .- max.(abs.(B1),real.(B3))) ≈ A2' .* A3 .- max.(abs.(A1),real.(A3))

            @test (B1 .+ sin.(B2 .- 3)) isa StridedView
            @test (B2' .* B3 .- Ref(0.5)) isa StridedView
            @test (B2' .* B3 .- max.(abs.(B1),real.(B3))) isa StridedView
            @test (B2' .* A3 .- max.(abs.(B1),real.(B3))) isa Array
        end
    end
end

@testset "mapreduce with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1 = rand(T, (10, 10, 10, 10, 10, 10))
        @test sum(R1; dims = (1, 3, 5)) ≈ sum(SV(R1); dims = (1, 3, 5))
        @test mapreduce(sin, +, R1; dims = (1, 3, 5)) ≈ mapreduce(sin, +, SV(R1); dims = (1, 3, 5))
        R2 = rand(T, (10, 10, 10))
        R2c = copy(R2)
        @test Strided._mapreducedim!(sin, +, identity, (10,10,10,10,10,10), (sreshape(SV(R2c),(10,1,1,10,10,1)), SV(R1))) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+ reshape(R2, (10,1,1,10,10,1))
        R2c = copy(R2)
        @test Strided._mapreducedim!(sin, +, x->0, (10,10,10,10,10,10), (sreshape(SV(R2c),(10,1,1,10,10,1)), SV(R1))) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6))
        R2c = copy(R2)
        β = rand(T)
        @test Strided._mapreducedim!(sin, +, x->β*x, (10,10,10,10,10,10), (sreshape(SV(R2c),(10,1,1,10,10,1)), SV(R1))) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+ β .* reshape(R2, (10,1,1,10,10,1))
        R2c = copy(R2)
        @test Strided._mapreducedim!(sin, +, x->β, (10,10,10,10,10,10), (sreshape(SV(R2c),(10,1,1,10,10,1)), SV(R1))) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6), init = β)
        R2c = copy(R2)
        @test Strided._mapreducedim!(sin, +, conj, (10,10,10,10,10,10), (sreshape(SV(R2c),(10,1,1,10,10,1)), SV(R1))) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+ conj.(reshape(R2, (10,1,1,10,10,1)))

        R3 = rand(T, (100, 100, 2))
        @test sum(R3; dims = (1, 2)) ≈ sum(SV(R3); dims = (1, 2))
    end
end

@testset "complete reductions with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1 = rand(T, (10, 10, 10, 10, 10, 10))

        @test sum(R1) ≈ sum(SV(R1))
        @test maximum(abs, R1) ≈ maximum(abs, SV(R1))
        @test minimum(real, R1) ≈ minimum(real, SV(R1))
        @test sum(x->real(x)<0, R1) == sum(x->real(x)<0, SV(R1))

        R1 = permutedims(R1, (randperm(6)...,))

        @test sum(R1) ≈ sum(SV(R1))
        @test maximum(abs, R1) ≈ maximum(abs, SV(R1))
        @test minimum(real, R1) ≈ minimum(real, SV(R1))
        @test sum(x->real(x)<0, R1) == sum(x->real(x)<0, SV(R1))

        R2 = rand(T, (5,5,5))
        @test prod(exp, SV(R2)) ≈ exp(sum(SV(R2)))
    end
end

for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
    for T2 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        @testset "multiplication with $SV: $T1 times $T2" for SV in (StridedView, UnsafeStridedView)
            d = 20
            A1 = rand(T1, (d,d))
            A1c = copy(A1)
            B1 = SV(A1c)
            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A1) == op1(B1)
            end
            A2 = rand(T2, (d,d))
            T3 = promote_type(T1,T2)
            A3 = rand(T3, (d,d))
            A2c = copy(A2)
            A3c = copy(A3)
            GC.@preserve A1c A2c A3c begin
                B2 = SV(A2c)
                B3 = SV(A3c)

                for op1 in (identity, conj, transpose, adjoint)
                    for op2 in (identity, conj, transpose, adjoint)
                        @test op1(A1)*op2(A2) ≈ op1(B1)*op2(B2)
                        for op3 in (identity, conj, transpose, adjoint)
                            mul!(op3(B3), op1(B1), op2(B2))
                            @test B3 ≈ op3(op1(A1)*op2(A2)) # op3 is its own inverse
                        end
                    end
                end
            end
        end
    end
end

@testset "multiplication with $SV: Complex{Int}" for SV in (StridedView, UnsafeStridedView)
    d = 10
    A1 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    A2 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    A3 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    A1c = copy(A1)
    A2c = copy(A2)
    A3c = copy(A3)
    GC.@preserve A1c A2c A3c begin
        B1 = SV(A1c)
        B2 = SV(A2c)
        B3 = SV(A3c)

        for op1 in (identity, conj, transpose, adjoint)
            @test op1(A1) == op1(B1)
            for op2 in (identity, conj, transpose, adjoint)
                @test op1(A1)*op2(A2) ≈ op1(B1)*op2(B2)
                for op3 in (identity, conj, transpose, adjoint)
                    Strided.mul!(op3(B3), op1(B1), op2(B2))
                    @test B3 ≈ op3(op1(A1)*op2(A2)) # op3 is its own inverse
                end
            end
        end
    end
end

@testset "multiplication with $SV: Rational{Int}" for SV in (StridedView, UnsafeStridedView)
    d = 10
    A1 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    A2 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    A3 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    A1c = copy(A1)
    A2c = copy(A2)
    A3c = copy(A3)
    GC.@preserve A1c A2c A3c begin
        B1 = SV(A1c)
        B2 = SV(A2c)
        B3 = SV(A3c)

        for op1 in (identity, conj, transpose, adjoint)
            @test op1(A1) == op1(B1)
            for op2 in (identity, conj, transpose, adjoint)
                @test op1(A1)*op2(A2) ≈ op1(B1)*op2(B2)
                for op3 in (identity, conj, transpose, adjoint)
                    mul!(op3(B3), op1(B1), op2(B2))
                    @test B3 ≈ op3(op1(A1)*op2(A2)) # op3 is its own inverse
                end
            end
        end
    end
end

@testset "@strided macro" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1, A2, A3 = rand(T, (10,)), rand(T, (10,10)), rand(T, (10,10,10))

        @test (@strided(A1 .+ sin.(A2 .- 3))) isa StridedView
        @test (@strided(A1 .+ sin.(A2 .- 3))) ≈ A1 .+ sin.(A2 .- 3)
        @test (@strided(A2' .* A3 .- Ref(0.5))) ≈ A2' .* A3 .- Ref(0.5)
        @test (@strided(A2' .* A3 .- max.(abs.(A1),real.(A3)))) ≈ A2' .* A3 .- max.(abs.(A1),real.(A3))

        B2 = view(A2, :, 1:2:10)
        @test (@strided(A1 .+ sin.(view(A2,:,1:2:10) .- 3))) ≈
            (@strided(A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(view(A2,:,1:2:10) .- 3)

        B2 = view(A2', :, 1:6)
        B3 = view(A3,:,1:6,4)
        @test (@strided(view(A2',:,1:6) .* view(A3,:,1:6,4) .- Ref(0.5))) ≈
            (@strided(B2 .* B3 .- Ref(0.5))) ≈
            view(A2',:,1:6) .* view(A3,:,1:6,4) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = view(A3, 1:5, :, 2:2:10)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (@strided(view(A2,:,3)' .* view(A3,1:5,:,2:2:10) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10))))) ≈
            (@strided(B2' .* B3 .- max.(abs.(B1),real.(B3b)))) ≈
            view(A2,:,3)' .* view(A3,1:5,:,2:2:10) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10)))

        B2 = reshape(A2, (10,2,5))
        @test (@strided(A1 .+ sin.(reshape(A2, (10,2,5)) .- 3))) ≈
            (@strided(A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(reshape(A2, (10,2,5)) .- 3)

        B2 = reshape(A2, 1, 100)
        B3 = reshape(A3, 100, 1, 10)
        @test (@strided(reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5))) ≈
            (@strided(B2' .* B3 .- Ref(0.5))) ≈
            reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = reshape(view(A3, 1:5, :, :), 5, 10, 5, 2)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (@strided(view(A2,:,3)' .* reshape(view(A3,1:5,:,:), 5, 10, 5, 2) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10))))) ≈
            (@strided(B2' .* B3 .- max.(abs.(B1),real.(B3b)))) ≈
            view(A2,:,3)' .* reshape(view(A3,1:5,:,:), 5, 10, 5, 2) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10)))
    end
end

@testset "@unsafe_strided macro" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1, A2, A3 = rand(T, (10,)), rand(T, (10,10)), rand(T, (10,10,10))

        @test (@unsafe_strided(A1,A2,A1 .+ sin.(A2 .- 3))) isa StridedView

        @test (@unsafe_strided(A1,A2,A1 .+ sin.(A2 .- 3))) ≈ A1 .+ sin.(A2 .- 3)
        @test (@unsafe_strided(A2,A3, A2' .* A3 .- Ref(0.5))) ≈ A2' .* A3 .- Ref(0.5)
        @test (@unsafe_strided(A1,A2,A3, A2' .* A3 .- max.(abs.(A1),real.(A3)))) ≈ A2' .* A3 .- max.(abs.(A1),real.(A3))

        B2 = view(A2, :, 1:2:10)
        @test (@unsafe_strided(A1,A2,A1 .+ sin.(view(A2,:,1:2:10) .- 3))) ≈
            (@unsafe_strided(A1,B2,A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(view(A2,:,1:2:10) .- 3)

        B2 = view(A2', :, 1:6)
        B3 = view(A3,:,1:6,4)
        @test (@unsafe_strided(A2,A3,view(A2',:,1:6) .* view(A3,:,1:6,4) .- Ref(0.5))) ≈
            (@unsafe_strided(B2,B3,B2 .* B3 .- Ref(0.5))) ≈
            view(A2',:,1:6) .* view(A3,:,1:6,4) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = view(A3, 1:5, :, 2:2:10)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (@unsafe_strided(A1,A2,A3,view(A2,:,3)' .* view(A3,1:5,:,2:2:10) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10))))) ≈
            (@unsafe_strided(B1,B2,B3,B2' .* B3 .- max.(abs.(B1),real.(B3b)))) ≈
            view(A2,:,3)' .* view(A3,1:5,:,2:2:10) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10)))

        B2 = reshape(A2, (10,2,5))
        @test (@unsafe_strided(A1,A2,A1 .+ sin.(reshape(A2, (10,2,5)) .- 3))) ≈
            (@unsafe_strided(A1,B2,A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(reshape(A2, (10,2,5)) .- 3)

        B2 = reshape(A2, 1, 100)
        B3 = reshape(A3, 100, 1, 10)
        @test (@unsafe_strided(A2,A3,reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5))) ≈
            (@unsafe_strided(B2,B3,B2' .* B3 .- Ref(0.5))) ≈
            reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = reshape(view(A3, 1:5, :, :), 5, 10, 5, 2)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (@unsafe_strided(A1,A2,A3,view(A2,:,3)' .* reshape(view(A3,1:5,:,:), 5, 10, 5, 2) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10))))) ≈
            (@unsafe_strided(B1,B2,B3,B2' .* B3 .- max.(abs.(B1),real.(B3b)))) ≈
            view(A2,:,3)' .* reshape(view(A3,1:5,:,:), 5, 10, 5, 2) .- max.(abs.(view(A1,1:5)),real.(view(A3,4:4,4:4,2:2:10)))
    end
end
