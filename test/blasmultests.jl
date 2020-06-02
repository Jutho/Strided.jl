
for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
    for T2 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        @testset "multiplication with $SV: $T1 times $T2" for SV in (StridedView, UnsafeStridedView)
            d = 103
            A1 = rand(T1, (d,d))
            A2 = rand(T2, (d,d))
            T3 = promote_type(T1,T2)
            A3 = rand(T3, (d,d))
            A4 = rand(T3, (d,d))
            A1c = copy(A1)
            A2c = copy(A2)
            A3c = copy(A3)
            A4c = copy(A4)
            GC.@preserve A1c A2c A3c A4c begin
                B1 = SV(A1c)
                B2 = SV(A2c)
                B3 = SV(A3c)
                B4 = SV(A4c)

                for op1 in (identity, conj, transpose, adjoint)
                    @test op1(A1) == op1(B1)
                    for op2 in (identity, conj, transpose, adjoint)
                        @test op1(A1)*op2(A2) ≈ op1(B1)*op2(B2)
                        for op3 in (identity, conj, transpose, adjoint)
                            α = randn(T3)
                            β = randn(T3)
                            copyto!(B3, B4)
                            mul!(op3(B3), op1(B1), op2(B2), α, β)
                            @test B3 ≈ op3(β)*A4 + op3(α * op1(A1)*op2(A2)) # op3 is its own inverse
                        end
                    end
                end
            end
        end
    end
end
