using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView, UnsafeStridedView

@testset "multiplication with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        d = 20
        A1 = rand(T1, (d,d))
        A1c = copy(A1)
        B1 = SV(A1c)
        for op1 in (identity, conj, transpose, adjoint)
            @test op1(A1) == op1(B1)
        end
        for T2 in (Float32, Float64, Complex{Float32}, Complex{Float64})
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

    let T = Complex{Int}
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

    let T = Rational{Int}
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
end

@testset "reshape and permutedims with $SV" for SV in (StridedView, UnsafeStridedView)
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        @testset for N = 2:6
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
                A2 = convert(Array, B2)
                A3 = convert(Array, B3)

                @test rmul!(B1, 1//2) ≈ rmul!(A1, 1//2)
                @test axpy!(1//3, B1, B2) ≈ axpy!(1//3, A1, A2)
                @test axpby!(1//3, B1, 1//2, B3) ≈ axpby!(1//3, A1, 1//2, A3)
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, B2, B3) ≈ map((x,y,z)->sin(x)+y/exp(-abs(z)), A1, A2, A3)
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, B2, B3) isa StridedView
                @test map((x,y,z)->sin(x)+y/exp(-abs(z)), B1, A2, B3) isa Array
            end
        end
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
            A2 = convert(Array, B2)
            A3 = convert(Array, B3)

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
