@views function DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    @time for j in axes(ε̇.xx.ex, 2), i in axes(ε̇.xx.ex, 1)
         if BC.p.ex[i,j] == 0
            # Velocity gradient
            ∂Vx∂ξ = (V.x.v[i+1,j] - V.x.v[i,j]  ) / Δ.x
            ∂Vx∂η = (V.x.c[i,j]   - V.x.c[i,j-1]) / Δ.y
            ∂Vx∂x = ∂_∂1(∂Vx∂ξ, ∂Vx∂η ,∂ξ.∂x.ex[i,j], ∂η.∂x.ex[i,j]) 
            ∂Vx∂y = ∂_∂1(∂Vx∂ξ, ∂Vx∂η ,∂ξ.∂y.ex[i,j], ∂η.∂y.ex[i,j]) 
            ∂Vy∂ξ = (V.y.v[i+1,j] - V.y.v[i,j]  ) / Δ.x
            ∂Vy∂η = (V.y.c[i,j]   - V.y.c[i,j-1]) / Δ.y
            ∂Vy∂x = ∂_∂1(∂Vy∂ξ, ∂Vy∂η ,∂ξ.∂x.ex[i,j],∂η.∂x.ex[i,j])
            ∂Vy∂y = ∂_∂1(∂Vy∂ξ, ∂Vy∂η ,∂ξ.∂y.ex[i,j],∂η.∂y.ex[i,j])
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
            if BC.ε̇.ey[i,j] == 0
                # Velocity gradient
                ∂Vx∂ξ = (V.x.c[i,j]   - V.x.c[i-1,j]) / Δ.x
                ∂Vy∂ξ = (V.y.c[i,j]   - V.y.c[i-1,j]) / Δ.x 
                ∂Vx∂η = (V.x.v[i,j+1] - V.x.v[i,j]  ) / Δ.y
                ∂Vy∂η = (V.y.v[i,j+1] - V.y.v[i,j]  ) / Δ.y
            elseif BC.ε̇.ey[i,j] == 2
                ∂Vx∂ξ = (V.x.c[i,j]   - V.x.c[i-1,j]) / Δ.x
                ∂Vy∂ξ = (V.y.c[i,j]   - V.y.c[i-1,j]) / Δ.x  
                ∂Vx∂η = -∂Vy∂ξ
                ∂Vy∂η = 1//2*∂Vx∂ξ + 3//4*P.ey[i,j]/(D.v11.ey[i,j]/2)
            end
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