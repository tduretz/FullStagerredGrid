using Plots, Printf, LinearAlgebra, SpecialFunctions
using ExtendableSparse
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
@views    ∂_∂(fE,fW,fN,fS,Δ,a,b) = a*(fE - fW) / Δ.x .+ b*(fN - fS) / Δ.y
function Main_2D_DI()
    # Physics
    # x          = (min=-3.0, max=3.0)  
    # y          = (min=-5.0, max=0.0)
    x          = (min=-3.0, max=3.0)  
    y          = (min=-3.0, max=3.0)
    ε̇bg        = -1.0
    rad        = 0.5
    y0         = -2*0.
    g          = (x = 0., z=-1.0*0.0)
    inclusion  = true
    adapt_mesh = true
    solve      = true
    # Numerics
    nc         = (x=11,     y=11    )  # numerical grid resolution
    nv         = (x=nc.x+1, y=nc.y+1)  # numerical grid resolution
    ε          = 1e-8          # nonlinear tolerance
    iterMax    = 20            # max number of iters
    # Preprocessing
    Δ          = (x=(x.max-x.min)/nc.x, y=(y.max-y.min)/nc.y)
    # Array initialisation
    V        = ( x   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)),  # v: vertices --- c: centroids
                 y   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) ) 
    ε̇        = ( xx  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),  # ex: x edges --- ey: y edges
                 yy  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 xy  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)) )
    τ        = ( xx  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),  # ex: x edges --- ey: y edges
                 yy  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 xy  = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)) )
    ∇v̇       = (        ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)  )
    P        = (        ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)  )
    D        = ( v11 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v12 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),  
                 v13 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v21 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v22 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v23 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v31 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v32 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)),
                 v33 = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)), )
    ∂ξ       = ( ∂x  = (v  =  ones(nv.x,   nv.y), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y), ey =  ones(nv.x,   nc.y+2)), 
                 ∂y  = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)) )
    ∂η       = ( ∂x  = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)), 
                 ∂y  = (v  =  ones(nv.x,   nv.y), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y), ey =  ones(nv.x,   nc.y+2)) ) 
    R        = ( x   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)), 
                 y   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)),
                 p   = (ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)) ) 
    ρ        = ( (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) )
    η        = (        ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)  )
    BC       = (x = zeros(Int, nv.x,   nv.y), y = zeros(Int, nv.x,   nv.y) )
    Num      = ( x   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                 y   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                 p   = (ex = -1*ones(Int, nc.x+2, nv.y), ey = -1*ones(Int, nv.x,   nc.y+2)) )
    # Fine mesh
    xxv, yyv    = LinRange(x.min-Δ.x/2, x.max+Δ.x/2, 2nc.x+3), LinRange(y.min-Δ.y/2, y.max+Δ.y/2, 2nc.y+3)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
    xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    x = merge(x, (v=xv4[2:2:end-1,2:2:end-1  ], c=xv4[1:2:end-0,1:2:end-0  ], ex=xv4[1:2:end,2:2:end-1  ], ey=xv4[2:2:end-1,1:2:end]) ) 
    y = merge(x, (v=yv4[2:2:end-1,2:2:end-1  ], c=yv4[1:2:end-0,1:2:end-0  ], ex=yv4[1:2:end,2:2:end-1  ], ey=yv4[2:2:end-1,1:2:end]) ) 
    # Velocity
    V.x.v .= -ε̇bg.*x.v; V.x.c .= -ε̇bg.*x.c
    V.y.v .=  ε̇bg.*y.v; V.y.c .=  ε̇bg.*y.c
    # Viscosity
    η.ex  .= 1.0; η.ey  .= 1.0
    if inclusion
        η.ex[x.ex.^2 .+ (y.ex.-y0).^2 .< rad] .= 100.
        η.ey[x.ey.^2 .+ (y.ey.-y0).^2 .< rad] .= 100.
    end
    D.v11.ex .= 2 .* η.ex; D.v11.ey .= 2 .* η.ey
    D.v22.ex .= 2 .* η.ex; D.v22.ey .= 2 .* η.ey
    D.v33.ex .= 2 .* η.ex; D.v33.ey .= 2 .* η.ey
    # Density
    ρ.v  .= 1.0; ρ.c  .= 1.0
    if inclusion
        ρ.v[x.v.^2 .+ (y.v.-y0).^2 .< rad] .= 2.
        ρ.c[x.c.^2 .+ (y.c.-y0).^2 .< rad] .= 2.
    end
    # Boundary conditions
    BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    # Vx
    if BCVx.West == :Dirichlet 
        BC.x[1,:]   .= 1
    end
    if BCVx.East == :Dirichlet 
        BC.x[end,:] .= 1
    end
    if BCVx.South == :Dirichlet 
        BC.x[:,1]   .= 1
    end
    if BCVx.North == :Dirichlet 
        BC.x[:,end] .= 1
    end
    # Vy
    if BCVy.West == :Dirichlet 
        BC.y[1,:]   .= 1
    end
    if BCVy.East == :Dirichlet 
        BC.y[end,:] .= 1
    end
    if BCVy.South == :Dirichlet 
        BC.y[:,1]   .= 1
    end
    if BCVy.North == :Dirichlet 
        BC.y[:,end] .= 1
    end
    # Loop on vertices
    @time for j in axes(R.x.v, 2), i in axes(R.x.v, 1)
        # Stencil
        τxxW = τ.xx.ex[i  ,j]; τyyW = τ.yy.ex[i  ,j]; τxyW = τ.xy.ex[i  ,j]; PW   = P.ex[i,  j]
        τxxE = τ.xx.ex[i+1,j]; τyyE = τ.yy.ex[i+1,j]; τxyE = τ.xy.ex[i+1,j]; PE   = P.ex[i+1,j]
        τxxS = τ.xx.ey[i,j  ]; τyyS = τ.yy.ey[i,j  ]; τxyS = τ.xy.ey[i,j  ]; PS   = P.ey[i,j]
        τxxN = τ.xx.ey[i,j+1]; τyyN = τ.yy.ey[i,j+1]; τxyN = τ.xy.ey[i,j+1]; PN   = P.ey[i,j+1]   
        ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.v[i,j];   ∂η∂y = ∂η.∂y.v[i,j]
        # x
        if BC.x[i,j] == 1
            R.x.v[i,j] = 0.
        else
            R.x.v[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x )
        end
        # y
        if BC.x[i,j] == 1
            R.x.v[i,j] = 0.
        else
            R.y.v[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z) 
        end
    end 
    # Loop on centroids
    @time for j in axes(R.x.c, 2), i in axes(R.x.c, 1)
        if i>1 && i<size(R.x.c,1) && j>1 && j<size(R.x.c,2)
        # Stencil
        τxxW = τ.xx.ey[i-1,j]; τyyW = τ.yy.ey[i-1,j]; τxyW = τ.xy.ey[i-1,j]; PW   = P.ey[i-1,j]
        τxxE = τ.xx.ey[i,j  ]; τyyE = τ.yy.ey[i,j  ]; τxyE = τ.xy.ey[i,j  ]; PE   = P.ey[i,j  ]
        τxxS = τ.xx.ex[i,j-1]; τyyS = τ.yy.ex[i,j-1]; τxyS = τ.xy.ex[i,j-1]; PS   = P.ex[i,j-1]
        τxxN = τ.xx.ex[i,j  ]; τyyN = τ.yy.ex[i,j  ]; τxyN = τ.xy.ex[i,j  ]; PN   = P.ex[i,j  ]
        ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.c[i,j];   ∂η∂y = ∂η.∂y.c[i,j]
        # x
        R.x.c[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x)
        # y
        R.y.c[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z) 
        end
    end 
    # Loop on horizontal edges
    @time for j in axes(R.p.ex, 2), i in axes(R.p.ex, 1)
        if i>1 && i<size(R.p.ex, 1)
            R.p.ex[i,j] = -∇v̇.ex[i,j]
        end
    end
    # Loop on horizontal edges
    @time for j in axes(R.p.ey, 2), i in axes(R.p.ey, 1)
        if j>1 && j<size(R.p.ey, 2)
            R.p.ey[i,j] = -∇v̇.ey[i,j]

        end
    end
    # Numbering
    Num.x.v[        :,     :] .= reshape(1:((nv.x)*(nv.y)), nv.x, nv.y)
    Num.x.c[ 2:end-1,2:end-1] .= reshape(1:((nc.x)*(nc.y)), nc.x, nc.y) .+ maximum(Num.x.v)
    Num.y.v[        :,     :] .= reshape(1:((nv.x)*(nv.y)), nv.x, nv.y) .+ maximum(Num.x.c)
    Num.y.c[ 2:end-1,2:end-1] .= reshape(1:((nc.x)*(nc.y)), nc.x, nc.y) .+ maximum(Num.y.v)
    Num.p.ex[2:end-1,      :] .= reshape(1:((nc.x)*(nv.y)), nc.x, nv.y) 
    Num.p.ey[      :,2:end-1] .= reshape(1:((nv.x)*(nc.y)), nv.x, nc.y) .+ maximum(Num.p.ex)
    # Initial sparse
    println("Initial Assembly")
    ndofV = maximum(Num.y.c)
    ndofP = maximum(Num.p.ey)
    Kuu   = ExtendableSparseMatrix(ndofV, ndofV)
    Kup   = ExtendableSparseMatrix(ndofV, ndofP)
    inz   = 1
    @time for j in axes(R.x.v, 2), i in axes(R.x.v, 1)
        iVxC = Num.x.v[i,j] 
        iVyC = Num.y.v[i,j] 
        if i>1  
            iVxW = Num.x.v[i-1,j] 
            iVyW = Num.y.v[i-1,j] 
            iPW  = Num.p.ex[i,j]
        end
        if i<size(R.x.v,1)  
            iVxE = Num.x.v[i+1,j] 
            iVyE = Num.y.v[i+1,j] 
            iPE  = Num.p.ex[i+1,j]
        end
        if j>1  
            iVxS = Num.x.v[i,j-1] 
            iVyS = Num.y.v[i,j-1] 
            iPS  = Num.p.ey[i,j]
        end
        if j<size(R.x.v,2)  
            iVxN = Num.x.v[i,j+1] 
            iVyN = Num.y.v[i,j+1] 
            iPN  = Num.p.ey[i,j+1]
        end
        if i>1 && j>1
            iVxSW = Num.x.c[i,j] 
            iVySW = Num.y.c[i,j] 
        end
        if i<size(R.x.v,1)  && j>1
            iVxSE = Num.x.c[i+1,j] 
            iVySE = Num.y.c[i+1,j] 
        end
        if i>1 && j<size(R.x.v,2) 
            iVxNW = Num.x.c[i,j+1] 
            iVyNW = Num.y.c[i,j+1] 
        end
        if i<size(R.x.v,1)  && j<size(R.x.v,2) 
            iVxNE = Num.x.c[i+1,j+1] 
            iVyNE = Num.y.c[i+1,j+1] 
        end
        # Boundaries for x
        if BC.x[i,j] == 1
            Kuu[iVxC,iVxC]  = 1.0
        else
            Kuu[iVxC,iVxC]  = 1.0
            Kuu[iVxC,iVxW]  = 1.0
            Kuu[iVxC,iVxE]  = 1.0
            Kuu[iVxC,iVxS]  = 1.0
            Kuu[iVxC,iVxN]  = 1.0
            Kuu[iVxC,iVxSW] = 1.0
            Kuu[iVxC,iVxSE] = 1.0
            Kuu[iVxC,iVxNW] = 1.0
            Kuu[iVxC,iVxNE] = 1.0
            Kuu[iVxC,iVyW]  = 1.0
            Kuu[iVxC,iVyE]  = 1.0
            Kuu[iVxC,iVyS]  = 1.0
            Kuu[iVxC,iVyN]  = 1.0
            Kuu[iVxC,iVySW] = 1.0
            Kuu[iVxC,iVySE] = 1.0
            Kuu[iVxC,iVyNW] = 1.0
            Kuu[iVxC,iVyNE] = 1.0
            #--------------
            # println(Kup)
            println(i)
            println(j)
            println(iVxC)
            println(iPW)
            Kup[iVxC,iPW]  = 1.0
            Kup[iVxC,iPE]  = 1.0
            Kup[iVxC,iPS]  = 1.0
            Kup[iVxC,iPN]  = 1.0
        end
        # Boundaries for y
        if BC.y[i,j] == 1
            Kuu[iVyC,iVyC] = inz; inz+=1
        else
            Kuu[iVyC,iVyC]  = 1.0
            Kuu[iVyC,iVxW]  = 1.0
            Kuu[iVyC,iVxE]  = 1.0
            Kuu[iVyC,iVxS]  = 1.0
            Kuu[iVyC,iVxN]  = 1.0
            Kuu[iVyC,iVxSW] = 1.0
            Kuu[iVyC,iVxSE] = 1.0
            Kuu[iVyC,iVxNW] = 1.0
            Kuu[iVyC,iVxNE] = 1.0
            Kuu[iVyC,iVyW]  = 1.0
            Kuu[iVyC,iVyE]  = 1.0
            Kuu[iVyC,iVyS]  = 1.0
            Kuu[iVyC,iVyN]  = 1.0
            Kuu[iVyC,iVySW] = 1.0
            Kuu[iVyC,iVySE] = 1.0
            Kuu[iVyC,iVyNW] = 1.0
            Kuu[iVyC,iVyNE] = 1.0
            #--------------
            Kup[iVyC,iPW]  = 1.0
            Kup[iVyC,iPE]  = 1.0
            Kup[iVyC,iPS]  = 1.0
            Kup[iVyC,iPN]  = 1.0
        end
    end 
    flush!(Kuu), flush!(Kup)
    return Kup
end

Main_2D_DI()