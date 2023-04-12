for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
    for T2 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        @testset "Matrix multiplication with StridedView: $T1 times $T2" begin
            d = 103
            A1 = rand(T1, (d, d))
            A2 = rand(T2, (d, d))
            T3 = promote_type(T1, T2)
            A3 = rand(T3, (d, d))
            A4 = rand(T3, (d, d))
            A1c = copy(A1)
            A2c = copy(A2)
            A3c = copy(A3)
            A4c = copy(A4)
            B1 = StridedView(A1c)
            B2 = StridedView(A2c)
            B3 = StridedView(A3c)
            B4 = StridedView(A4c)

            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A1) == op1(B1)
                for op2 in (identity, conj, transpose, adjoint)
                    @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
                    for op3 in (identity, conj, transpose, adjoint)
                        α = randn(T3)
                        β = randn(T3)
                        copyto!(B3, B4)
                        mul!(op3(B3), op1(B1), op2(B2), α, β)
                        @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                    end
                end
            end
        end
        
        @testset "Tensor product with StridedView: $T1 times $T2" begin
            d = 10
            A1 = rand(T1, (d, 1))
            A2 = rand(T2, (1, d))
            T3 = promote_type(T1, T2)
            A3 = rand(T3, (d, d))
            A4 = rand(T3, (d, d))
            B1 = StridedView(copy(A1))
            B2 = StridedView(copy(A2))
            B3 = StridedView(copy(A3))
            B4 = StridedView(copy(A4))
            α = randn(T3)
            β = randn(T3)
            for op1 in (identity, conj)
                @test op1(A1) == op1(B1)
                for op2 in (identity, conj)
                    @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
                    for op3 in (identity, conj, transpose, adjoint)
                        α = randn(T3)
                        β = randn(T3)
                        copyto!(B3, B4)
                        mul!(op3(B3), op1(B1), op2(B2), α, β)
                        @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                    end
                end
            end
        end
    end
end
