@views function LinearMomentumResidual!( R, ∇v, τ, P, ρ, g, ∂ξ, ∂η, Δ, BC )
    # Loop on vertices
    for j in axes(R.x.v, 2), i in axes(R.x.v, 1)
        if i>1 && i<size(R.x.v,1) && j>1 && j<size(R.x.v,2)
            # Stencil
            τxxW = τ.xx.ex[i-1,j]; τyyW = τ.yy.ex[i-1,j]; τxyW = τ.xy.ex[i-1,j]; PW   = P.ex[i-1,j]; 
            τxxE = τ.xx.ex[i,j];   τyyE = τ.yy.ex[i,j];   τxyE = τ.xy.ex[i,j];   PE   = P.ex[i,j];   
            τxxS = τ.xx.ey[i,j-1]; τyyS = τ.yy.ey[i,j-1]; τxyS = τ.xy.ey[i,j-1]; PS   = P.ey[i,j-1]; 
            τxxN = τ.xx.ey[i,j];   τyyN = τ.yy.ey[i,j];   τxyN = τ.xy.ey[i,j];   PN   = P.ey[i,j];      
            ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.v[i,j];   ∂η∂y = ∂η.∂y.v[i,j]
            # x
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                R.x.v[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x )
            end
            # y
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                R.y.v[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z) 
            end
        end
    end 
    # Loop on centroids
    for j in axes(R.x.c, 2), i in axes(R.x.c, 1)
        if i>1 && i<size(R.x.c,1) && j>1 && j<size(R.x.c,2)
            # Stencil
            τxxW = τ.xx.ey[i,j];   τyyW = τ.yy.ey[i,j];   τxyW = τ.xy.ey[i,j];   PW   = P.ey[i,j]
            τxxE = τ.xx.ey[i+1,j]; τyyE = τ.yy.ey[i+1,j]; τxyE = τ.xy.ey[i+1,j]; PE   = P.ey[i+1,j]
            τxxS = τ.xx.ex[i,j];   τyyS = τ.yy.ex[i,j];   τxyS = τ.xy.ex[i,j];   PS   = P.ex[i,j]
            τxxN = τ.xx.ex[i,j+1]; τyyN = τ.yy.ex[i,j+1]; τxyN = τ.xy.ex[i,j+1]; PN   = P.ex[i,j+1]
            ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.c[i,j];   ∂η∂y = ∂η.∂y.c[i,j]
            # x
            R.x.c[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.c[i,j]*g.x)
            # y
            R.y.c[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.c[i,j]*g.z) 
        end
    end 
    # Loop on horizontal edges
    for j in axes(R.p.ex, 2), i in axes(R.p.ex, 1)
        if i>1 && i<size(R.p.ex, 1)
            R.p.ex[i,j] = -∇v.ex[i,j]
        end
    end
    # Loop on vertical edges
    for j in axes(R.p.ey, 2), i in axes(R.p.ey, 1)
        if j>1 && j<size(R.p.ey, 2)
            R.p.ey[i,j] = -∇v.ey[i,j]
        end
    end
    return nothing
end