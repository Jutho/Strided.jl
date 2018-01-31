if VERSION < v"0.7.0-DEV.2005"
    const Test = Base.Test
end
if VERSION >= v"0.7.0-DEV.3406"
    using Random
end

using Test
using Strided
const adjoint = Strided.adjoint

for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
    d = 20
    A1 = rand(T, (d,d))
    A2 = rand(T, (d,d))
    A3 = rand(T, (d,d))
    B1 = StridedView(copy(A1))
    B2 = StridedView(copy(A2))
    B3 = StridedView(copy(A3))

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

let T = Complex{Int}
    d = 10
    A1 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    A2 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    A3 = map(complex, rand(-100:100, (d,d)), rand(-100:100, (d,d)))
    B1 = StridedView(copy(A1))
    B2 = StridedView(copy(A2))
    B3 = StridedView(copy(A3))

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

let T = Rational{Int}
    d = 10
    A1 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    A2 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    A3 = map(//, rand(-10:10, (d,d)), rand(1:10, (d,d)))
    B1 = StridedView(copy(A1))
    B2 = StridedView(copy(A2))
    B3 = StridedView(copy(A3))

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


for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
    for N = 2:6
        dims = ntuple(n->rand(1:10), N)
        A = rand(T, dims)
        B = StridedView(copy(A))
        @test conj(A) == conj(B)
        p = randperm(N)
        B2 = permutedims(B, p)
        A2 = permutedims(A, p)
        @test B2 == A2
        @test copy(B2) == A2
        @test convert(Array, B2) == A2

        dims = ntuple(n->10, N)
        A = rand(T, dims)
        B = StridedView(copy(A))
        @test conj(A) == conj(B)
        p = randperm(N)
        B2 = permutedims(B, p)
        A2 = permutedims(A, p)
        @test B2 == A2
        @test copy(B2) == A2
        @test convert(Array, B2) == A2

        B2 = splitdims(B, 1=>(2,5), N=>(5,2))
        A2 = splitdims(A, 1=>(2,5), N=>(5,2))
        A3 = reshape(A, size(A2))
        @test B2 == A3
        @test B2 == A2
        p = randperm(N+2)
        @test conj(permutedims(B2, p)) == conj(permutedims(A3, p))
    end
end
