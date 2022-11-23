@views function LinearMomentumResidual!( R, ∇v, τ, P, P0, β, K, ρ, g, ∂ξ, ∂η, Δ, BC, comp, symmetric )
    # Loop on vertices
    for j in axes(R.x.v, 2), i in axes(R.x.v, 1)
        if i>1 && i<size(R.x.v,1) && j>1 && j<size(R.x.v,2)
            # Stencil
            τxxW = τ.xx.ex[i-1,j]; τyyW = τ.yy.ex[i-1,j]; τxyW = τ.xy.ex[i-1,j]; pW   = P.ex[i-1,j]; 
            τxxE = τ.xx.ex[i,j];   τyyE = τ.yy.ex[i,j];   τxyE = τ.xy.ex[i,j];   pE   = P.ex[i,j];   
            τxxS = τ.xx.ey[i,j-1]; τyyS = τ.yy.ey[i,j-1]; τxyS = τ.xy.ey[i,j-1]; pS   = P.ey[i,j-1]; 
            τxxN = τ.xx.ey[i,j];   τyyN = τ.yy.ey[i,j];   τxyN = τ.xy.ey[i,j];   pN   = P.ey[i,j];      
            ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.v[i,j];   ∂η∂y = ∂η.∂y.v[i,j]
            aW   = ∂ξ.∂x.ex[i-1,j];  cW = ∂ξ.∂y.ex[i-1,j]; bW   = ∂η.∂x.ex[i-1,j];  dW = ∂η.∂y.ex[i-1,j];
            aE   = ∂ξ.∂x.ex[i,j];    cE = ∂ξ.∂y.ex[i,j];   bE   = ∂η.∂x.ex[i,j];    dE = ∂η.∂y.ex[i,j];
            aS   = ∂ξ.∂x.ey[i,j-1];  cS = ∂ξ.∂y.ey[i,j-1]; bS   = ∂η.∂x.ey[i,j-1];  dS = ∂η.∂y.ey[i,j-1];
            aN   = ∂ξ.∂x.ey[i,j];    cN = ∂ξ.∂y.ey[i,j];   bN   = ∂η.∂x.ey[i,j];    dN = ∂η.∂y.ey[i,j];
            # x
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                if symmetric
                    # R.x.v[i,j] = - ( (aE*τxxE-aW*τxxW)/Δ.x + (bN*τxxN-bS*τxxS)/Δ.y + (cE*τxyE-cW*τxyW)/Δ.x + (dN*τxyN-dS*τxyS)/Δ.y - ((aE*pE-aW*pW)/Δ.x + (bN*pN-bS*pS)/Δ.y) + ρ.v[i,j]*g.x )
                    R.x.v[i,j] = - ( (aE*τxxE-aW*τxxW)/Δ.x + (bN*τxxN-bS*τxxS)/Δ.y + (cE*τxyE-cW*τxyW)/Δ.x + (dN*τxyN-dS*τxyS)/Δ.y - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x )
                else
                    R.x.v[i,j] = - ( ∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x ) / (∂ξ∂x*∂η∂y)
                end
            end
            # y
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                if symmetric
                    # R.y.v[i,j] = - ( (cE*τyyE-cW*τyyW)/Δ.x + (dN*τyyN-dS*τyyS)/Δ.y + (aE*τxyE-aW*τxyW)/Δ.x + (bN*τxyN-bS*τxyS)/Δ.y - ((cE*pE-cW*pW)/Δ.x + (dN*pN-dS*pS)/Δ.y) + ρ.v[i,j]*g.z ) 
                    R.y.v[i,j] = - ( (cE*τyyE-cW*τyyW)/Δ.x + (dN*τyyN-dS*τyyS)/Δ.y + (aE*τxyE-aW*τxyW)/Δ.x + (bN*τxyN-bS*τxyS)/Δ.y - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z ) 
                else
                    R.y.v[i,j] = - ( ∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z ) / (∂ξ∂x*∂η∂y) 
                end
            end
        end
    end 
    # Loop on centroids
    for j in axes(R.x.c, 2), i in axes(R.x.c, 1)
        if i>1 && i<size(R.x.c,1) && j>1 && j<size(R.x.c,2)
            # Stencil
            τxxW = τ.xx.ey[i,j];   τyyW = τ.yy.ey[i,j];   τxyW = τ.xy.ey[i,j];   pW   = P.ey[i,j]
            τxxE = τ.xx.ey[i+1,j]; τyyE = τ.yy.ey[i+1,j]; τxyE = τ.xy.ey[i+1,j]; pE   = P.ey[i+1,j]
            τxxS = τ.xx.ex[i,j];   τyyS = τ.yy.ex[i,j];   τxyS = τ.xy.ex[i,j];   pS   = P.ex[i,j]
            τxxN = τ.xx.ex[i,j+1]; τyyN = τ.yy.ex[i,j+1]; τxyN = τ.xy.ex[i,j+1]; pN   = P.ex[i,j+1]
            ∂ξ∂x = ∂ξ.∂x.c[i,j];   ∂ξ∂y = ∂ξ.∂y.c[i,j];   ∂η∂x = ∂η.∂x.c[i,j];   ∂η∂y = ∂η.∂y.c[i,j]
            aW   = ∂ξ.∂x.ey[i,j];  cW = ∂ξ.∂y.ey[i,j];    bW   = ∂η.∂x.ey[i,j];  dW = ∂η.∂y.ey[i,j];
            aE   = ∂ξ.∂x.ey[i+1,j];cE = ∂ξ.∂y.ey[i+1,j];  bE   = ∂η.∂x.ey[i+1,j];dE = ∂η.∂y.ey[i+1,j];
            aS   = ∂ξ.∂x.ex[i,j];  cS = ∂ξ.∂y.ex[i,j];    bS   = ∂η.∂x.ex[i,j];  dS = ∂η.∂y.ex[i,j];
            aN   = ∂ξ.∂x.ex[i,j+1];cN = ∂ξ.∂y.ex[i,j+1];  bN   = ∂η.∂x.ex[i,j+1];dN = ∂η.∂y.ex[i,j+1];            
            if symmetric 
                # R.x.c[i,j] = - ( (aE*τxxE-aW*τxxW)/Δ.x + (bN*τxxN-bS*τxxS)/Δ.y + (cE*τxyE-cW*τxyW)/Δ.x + (dN*τxyN-dS*τxyS)/Δ.y - ((aE*pE-aW*pW)/Δ.x + (bN*pN-bS*pS)/Δ.y) + ρ.c[i,j]*g.x ) 
                # R.y.c[i,j] = - ( (cE*τyyE-cW*τyyW)/Δ.x + (dN*τyyN-dS*τyyS)/Δ.y + (aE*τxyE-aW*τxyW)/Δ.x + (bN*τxyN-bS*τxyS)/Δ.y - ((cE*pE-cW*pW)/Δ.x + (dN*pN-dS*pS)/Δ.y) + ρ.c[i,j]*g.z ) 
                R.x.c[i,j] = - ( (aE*τxxE-aW*τxxW)/Δ.x + (bN*τxxN-bS*τxxS)/Δ.y + (cE*τxyE-cW*τxyW)/Δ.x + (dN*τxyN-dS*τxyS)/Δ.y - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂x,∂η∂x) + ρ.c[i,j]*g.x ) 
                R.y.c[i,j] = - ( (cE*τyyE-cW*τyyW)/Δ.x + (dN*τyyN-dS*τyyS)/Δ.y + (aE*τxyE-aW*τxyW)/Δ.x + (bN*τxyN-bS*τxyS)/Δ.y - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂y,∂η∂y) + ρ.c[i,j]*g.z ) 
            else
                R.x.c[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂x,∂η∂x) + ρ.c[i,j]*g.x ) / (∂ξ∂x*∂η∂y)
                R.y.c[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(pE,pW,pN,pS,Δ,∂ξ∂y,∂η∂y) + ρ.c[i,j]*g.z ) / (∂ξ∂x*∂η∂y) 
            end
            
             
        end
    end 
    # Loop on horizontal edges
    for j in axes(R.p.ex, 2), i in axes(R.p.ex, 1)
        if i>1 && i<size(R.p.ex, 1)
            R.p.ex[i,j] = -∇v.ex[i,j] - β.ex[i,j]*(P.ex[i,j] - P0.ex[i,j]) / (K.ex[i,j]*Δ.t)
        end
    end
    # Loop on vertical edges
    for j in axes(R.p.ey, 2), i in axes(R.p.ey, 1)
        if j>1 && j<size(R.p.ey, 2)
            R.p.ey[i,j] = -∇v.ey[i,j] - β.ey[i,j]*(P.ey[i,j] - P0.ey[i,j]) / (K.ey[i,j]*Δ.t)
        end
    end
    return nothing
end