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
