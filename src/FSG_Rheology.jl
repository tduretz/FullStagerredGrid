#=
dVxdy = ((3 // 2) * P .* b .* h_x .^ 2 - 3 // 2 * P .* b + 3 * P .* d .* h_x - 2 * a .* b .* dudx .* eta .* h_x .^ 2 - a .* b .* dudx .* eta - 2 * a .* d .* dudx .* eta .* h_x - a .* d .* dvdx .* eta .* h_x .^ 2 - 2 * a .* d .* dvdx .* eta + b .* c .* eta .* h_x .^ 2 + 2 * b .* c .* eta - c .* d .* eta .* h_x .^ 2 - 2 * c .* d .* eta) ./ (eta .* (2 * b .^ 2 .* h_x .^ 2 + b .^ 2 + 2 * b .* d .* h_x + d .^ 2 .* h_x .^ 2 + 2 * d .^ 2))
dVydy = (3 * P .* b .* h_x - 3 // 2 * P .* d .* h_x .^ 2 + (3 // 2) * P .* d - 2 * a .* b .* dvdx .* eta .* h_x .^ 2 - a .* b .* dvdx .* eta + 2 * a .* d .* dudx .* eta .* h_x .^ 2 + a .* d .* dudx .* eta - 2 * b .* c .* eta .* h_x .^ 2 - 2 * b .* c .* eta .* h_x - b .* c .* eta - c .* d .* eta .* h_x .^ 2 - 2 * c .* d .* eta) ./ (eta .* (2 * b .^ 2 .* h_x .^ 2 + b .^ 2 + 2 * b .* d .* h_x + d .^ 2 .* h_x .^ 2 + 2 * d .^ 2))
=#

@views function eval_∂Vx∂η( dudx, dvdx, P, C0, C1, C2 )
    # ∂Vx∂η = ((3 // 2) * P .* b .* h_x .^ 2 - 3 // 2 * P .* b + 3 * P .* d .* h_x - 2 * a .* b .* dudx .* eta .* h_x .^ 2 - a .* b .* dudx .* eta - 2 * a .* d .* dudx .* eta .* h_x - a .* d .* dvdx .* eta .* h_x .^ 2 - 2 * a .* d .* dvdx .* eta + b .* c .* eta .* h_x .^ 2 + 2 * b .* c .* eta - c .* d .* eta .* h_x .^ 2 - 2 * c .* d .* eta) ./ (eta .* (2 * b .^ 2 .* h_x .^ 2 + b .^ 2 + 2 * b .* d .* h_x + d .^ 2 .* h_x .^ 2 + 2 * d .^ 2))
    ∂Vx∂η = C0*P + C1*dudx + C2*dvdx
    return ∂Vx∂η
end

@views function eval_∂Vy∂η( dudx, dvdx, P, D0, D1, D2 )
    # ∂Vy∂η = (3 * P .* b .* h_x - 3 // 2 * P .* d .* h_x .^ 2 + (3 // 2) * P .* d - 2 * a .* b .* dvdx .* eta .* h_x .^ 2 - a .* b .* dvdx .* eta + 2 * a .* d .* dudx .* eta .* h_x .^ 2 + a .* d .* dudx .* eta - 2 * b .* c .* eta .* h_x .^ 2 - 2 * b .* c .* eta .* h_x - b .* c .* eta - c .* d .* eta .* h_x .^ 2 - 2 * c .* d .* eta) ./ (eta .* (2 * b .^ 2 .* h_x .^ 2 + b .^ 2 + 2 * b .* d .* h_x + d .^ 2 .* h_x .^ 2 + 2 * d .^ 2))
    ∂Vy∂η = D0*P + D1*dudx + D2*dvdx
    return ∂Vy∂η
end

@views function DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    @time for j in axes(ε̇.xx.ex, 2), i in axes(ε̇.xx.ex, 1)
        if BC.p.ex[i,j] != -1
            # Velocity gradient
            if BC.p.ex[i,j] == 0  
                ∂Vx∂ξ = (V.x.v[i+1,j] - V.x.v[i,j]  ) / Δ.x
                ∂Vy∂ξ = (V.y.v[i+1,j] - V.y.v[i,j]  ) / Δ.x
                ∂Vx∂η = (V.x.c[i,j]   - V.x.c[i,j-1]) / Δ.y
                ∂Vy∂η = (V.y.c[i,j]   - V.y.c[i,j-1]) / Δ.y
            else BC.p.ex[i,j] == 2
                ∂Vx∂ξ = (V.x.v[i+1,j] - V.x.v[i,j]  ) / Δ.x
                ∂Vy∂ξ = (V.y.v[i+1,j] - V.y.v[i,j]  ) / Δ.x
                ∂Vx∂η = eval_∂Vx∂η( ∂Vx∂ξ, ∂Vy∂ξ, P.ex[i,j], BC.C0[i], BC.C1[i], BC.C2[i] )
                ∂Vy∂η = eval_∂Vy∂η( ∂Vx∂ξ, ∂Vy∂ξ, P.ex[i,j], BC.D0[i], BC.D1[i], BC.D2[i] )
            end
            ∂Vx∂x = ∂_∂1(∂Vx∂ξ, ∂Vx∂η, ∂ξ.∂x.ex[i,j], ∂η.∂x.ex[i,j]) 
            ∂Vy∂x = ∂_∂1(∂Vy∂ξ, ∂Vy∂η, ∂ξ.∂x.ex[i,j], ∂η.∂x.ex[i,j])
            ∂Vx∂y = ∂_∂1(∂Vx∂ξ, ∂Vx∂η, ∂ξ.∂y.ex[i,j], ∂η.∂y.ex[i,j]) 
            ∂Vy∂y = ∂_∂1(∂Vy∂ξ, ∂Vy∂η, ∂ξ.∂y.ex[i,j] ,∂η.∂y.ex[i,j])
            # Deviatoric strain rate
            ε̇.xx.ex[i,j] = ∂Vx∂x - 1//3*(∂Vx∂x + ∂Vy∂y)
            ε̇.yy.ex[i,j] = ∂Vy∂y - 1//3*(∂Vx∂x + ∂Vy∂y)
            ε̇.xy.ex[i,j] = 1//2 * (∂Vx∂y + ∂Vy∂x)
            ∇v.ex[i,j]   = ∂Vx∂x + ∂Vy∂y
            # Deviatoric stress
            τ.xx.ex[i,j] = D.v11.ex[i,j]*ε̇.xx.ex[i,j] + D.v12.ex[i,j]*ε̇.yy.ex[i,j] + D.v13.ex[i,j]*ε̇.xy.ex[i,j]
            τ.yy.ex[i,j] = D.v21.ex[i,j]*ε̇.xx.ex[i,j] + D.v22.ex[i,j]*ε̇.yy.ex[i,j] + D.v23.ex[i,j]*ε̇.xy.ex[i,j]
            τ.xy.ex[i,j] = D.v31.ex[i,j]*ε̇.xx.ex[i,j] + D.v32.ex[i,j]*ε̇.yy.ex[i,j] + D.v33.ex[i,j]*ε̇.xy.ex[i,j]
        end
     end
     @time for j in axes(ε̇.xx.ey, 2), i in axes(ε̇.xx.ey, 1)
        if BC.p.ey[i,j] != -1
            # Velocity gradient
            ∂Vx∂ξ = (V.x.c[i,j]   - V.x.c[i-1,j]) / Δ.x
            ∂Vy∂ξ = (V.y.c[i,j]   - V.y.c[i-1,j]) / Δ.x 
            ∂Vx∂η = (V.x.v[i,j+1] - V.x.v[i,j]  ) / Δ.y
            ∂Vy∂η = (V.y.v[i,j+1] - V.y.v[i,j]  ) / Δ.y
            ∂Vx∂x = ∂_∂1(∂Vx∂ξ, ∂Vx∂η, ∂ξ.∂x.ey[i,j], ∂η.∂x.ey[i,j]) 
            ∂Vx∂y = ∂_∂1(∂Vx∂ξ, ∂Vx∂η, ∂ξ.∂y.ey[i,j], ∂η.∂y.ey[i,j]) 
            ∂Vy∂x = ∂_∂1(∂Vy∂ξ, ∂Vy∂η, ∂ξ.∂x.ey[i,j], ∂η.∂x.ey[i,j])
            ∂Vy∂y = ∂_∂1(∂Vy∂ξ, ∂Vy∂η, ∂ξ.∂y.ey[i,j], ∂η.∂y.ey[i,j])
            # Deviatoric strain rate
            ε̇.xx.ey[i,j] = ∂Vx∂x - 1//3*(∂Vx∂x + ∂Vy∂y)
            ε̇.yy.ey[i,j] = ∂Vy∂y - 1//3*(∂Vx∂x + ∂Vy∂y)
            ε̇.xy.ey[i,j] = 1//2 * (∂Vx∂y + ∂Vy∂x)
            ∇v.ey[i,j]   = ∂Vx∂x + ∂Vy∂y
            # Deviatoric stress
            τ.xx.ey[i,j] = D.v11.ey[i,j]*ε̇.xx.ey[i,j] + D.v12.ey[i,j]*ε̇.yy.ey[i,j] + D.v13.ey[i,j]*ε̇.xy.ey[i,j]
            τ.yy.ey[i,j] = D.v21.ey[i,j]*ε̇.xx.ey[i,j] + D.v22.ey[i,j]*ε̇.yy.ey[i,j] + D.v23.ey[i,j]*ε̇.xy.ey[i,j]
            τ.xy.ey[i,j] = D.v31.ey[i,j]*ε̇.xx.ey[i,j] + D.v32.ey[i,j]*ε̇.yy.ey[i,j] + D.v33.ey[i,j]*ε̇.xy.ey[i,j]
         end
     end
 end