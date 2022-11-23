function KSP_GCR_Jacobian!( x, M, b, eps::Float64, noisy::Int64, Kxxf, f::Vector{Float64}, v::Vector{Float64}, s::Vector{Float64}, val::Vector{Float64}, VV::Matrix{Float64}, SS::Matrix{Float64}, restart::Int64 )
    # Initialise
    val .= 0.0
    s   .= 0.0
    v   .= 0.0
    VV  .= 0.0
    SS  .= 0.0
    # KSP GCR solver
    norm_r, norm0 = 0.0, 0.0
    N               = length(x)
    maxit           = 10*restart
    ncyc, its       = 0, 0
    i1, i2, success = 0, 0, 0
    # Initial residual
    f     .= b .- M*x 
    norm_r = sqrt(mydotavx( f, f ) )#norm(v)norm(f)
    norm0  = norm_r;
    ndof   = size(M,1)
    ndofu  = Int64(ndof/2)
    # ldiv_test!(P::Factor{Float64}, v) = (P \ v)
    # Solving procedure
     while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            # s     .= ldiv_test!(Kxxf, f)
            s .= Kxxf\f
            # Action of Jacobian on s: v = J*s
            mul!(v, M, s)
            # Approximation of the Jv product
            for i2=1:i1
                val[i2] = mydotavx( v, view(VV, :, i2 ) )   
            end
            # Scaling
            for i2=1:i1
                 v .-= val[i2] .* view(VV, :, i2 )
                 s .-= val[i2] .* view(SS, :, i2 )
            end
            # -----------------
            nrm_inv = 1.0 / sqrt(mydotavx( v, v ) )
            r_dot_v = mydotavx( f, v )  * nrm_inv
            # -----------------
             v     .*= nrm_inv
             s     .*= nrm_inv
            # -----------------
             x     .+= r_dot_v.*s
             f     .-= r_dot_v.*v
            # -----------------
            norm_r  = sqrt(mydotavx( f, f ) )
            if norm_r/sqrt(length(f)) < eps || its==23
                @printf("It. %04d: res. = %2.2e\n", its, norm_r/sqrt(length(f)))
                success = 1
                println("converged")
                break
            end
            # Store 
             VV[:,i1] .= v
             SS[:,i1] .= s
            its      += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end
export KSP_GCR_Stokes!


function mydotavx(A, B)
    s = zero(promote_type(eltype(A), eltype(B)))
     for i in eachindex(A,B)
        s += A[i] * B[i]
    end
    s
end