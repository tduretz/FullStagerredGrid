@views function DevStrainRateStressTensor!( ε̇, τ, D, ∇v, V, Δ, ∂ξ, ∂η )
    @time for j in axes(ε̇.xx.ex, 2), i in axes(ε̇.xx.ex, 1)
         if i>1 && i<size(ε̇.xx.ex,1) && j>1 && j<size(ε̇.xx.ex,2)
             # Velocity gradient
             VxW   = V.x.v[i,j]
             VxE   = V.x.v[i+1,j]
             VxS   = V.x.c[i,j-1]
             VxN   = V.x.c[i,j]
             ∂Vx∂x = ∂_∂(VxE,VxW,VxN,VxS,Δ,∂ξ.∂x.ex[i,j],∂η.∂x.ex[i,j]) 
             ∂Vx∂y = ∂_∂(VxE,VxW,VxN,VxS,Δ,∂ξ.∂y.ex[i,j],∂η.∂y.ex[i,j]) 
             VyW   = V.y.v[i,j]
             VyE   = V.y.v[i+1,j]
             VyS   = V.y.c[i,j-1]
             VyN   = V.y.c[i,j]
             ∂Vy∂x = ∂_∂(VyE,VyW,VyN,VyS,Δ,∂ξ.∂x.ex[i,j],∂η.∂x.ex[i,j])
             ∂Vy∂y = ∂_∂(VyE,VyW,VyN,VyS,Δ,∂ξ.∂x.ex[i,j],∂η.∂y.ex[i,j])
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
         if i>1 && i<size(ε̇.xx.ey,1) && j>1 && j<size(ε̇.xx.ey,2)
             # Velocity gradient
             VxW   = V.x.c[i-1,j]
             VxE   = V.x.c[i,j]
             VxS   = V.x.v[i,j]
             VxN   = V.x.v[i,j+1]
             ∂Vx∂x = ∂_∂(VxE,VxW,VxN,VxS,Δ,∂ξ.∂x.ey[i,j],∂η.∂x.ey[i,j]) 
             ∂Vx∂y = ∂_∂(VxE,VxW,VxN,VxS,Δ,∂ξ.∂y.ey[i,j],∂η.∂y.ey[i,j]) 
             VyW   = V.y.c[i-1,j]
             VyE   = V.y.c[i,j]
             VyS   = V.y.v[i,j]
             VyN   = V.y.v[i,j+1]
             ∂Vy∂x = ∂_∂(VyE,VyW,VyN,VyS,Δ,∂ξ.∂x.ey[i,j],∂η.∂x.ey[i,j])
             ∂Vy∂y = ∂_∂(VyE,VyW,VyN,VyS,Δ,∂ξ.∂x.ey[i,j],∂η.∂y.ey[i,j])
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