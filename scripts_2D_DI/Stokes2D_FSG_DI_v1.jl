using Plots, Printf, LinearAlgebra, SpecialFunctions
using ExtendableSparse
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
@views    ∂_∂(fE,fW,fN,fS,Δ,a,b) = a*(fE - fW) / Δ.x .+ b*(fN - fS) / Δ.y
@views   function AddToExtSparse!(K,i,j,Tag, v) 
    if ((j!=-1) || (j==i || Tag==1)) K[i,j]  = v end
end
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
    BC       = (x = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                y = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                p = (ex = -1*ones(Int, nc.x+2, nv.y),     ey = -1*ones(Int, nv.x,   nc.y+2))  )
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
    BC.x.v[2:end-1,2:end-1] .= 0 # inner points
    BC.y.v[2:end-1,2:end-1] .= 0 # inner points
    BC.x.c[2:end-1,2:end-1] .= 0 # inner points
    BC.y.c[2:end-1,2:end-1] .= 0 # inner points
    BC.p.ex[2:end-1,2:end-1] .= 0 # inner points
    BC.p.ey[2:end-1,2:end-1] .= 0 # inner points
    # Vx
    if BCVx.West == :Dirichlet 
        BC.x.v[2,2:end-1] .= 1
    end
    if BCVx.East == :Dirichlet 
        BC.x.v[end-1,2:end-1] .= 1
    end
    if BCVx.South == :Dirichlet 
        BC.x.v[2:end-1,2]   .= 1
    end
    if BCVx.North == :Dirichlet 
        BC.x.v[2:end-1,end-1] .= 1
    end
    # Vy
    if BCVy.West == :Dirichlet 
        BC.y.v[2,2:end-1]   .= 1
    end
    if BCVy.East == :Dirichlet 
        BC.y.v[end-1,2:end-1] .= 1
    end
    if BCVy.South == :Dirichlet 
        BC.y.v[2:end-1,2]   .= 1
    end
    if BCVy.North == :Dirichlet 
        BC.y.v[2:end-1,end-1] .= 1
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

    Num      = ( x   = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                    y   = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                    p   = (ex = -1*ones(Int, nc.x+2, nv.y+2), ey = -1*ones(Int, nv.x+2,   nc.y+2)) )


    Num.x.v[ 2:end-1,2:end-1] .= reshape(1:((nv.x)*(nv.y)), nv.x, nv.y)
    Num.y.v[ 2:end-1,2:end-1] .= reshape(1:((nv.x)*(nv.y)), nv.x, nv.y) .+ maximum(Num.x.v)
    Num.x.c[ 2:end-1,2:end-1] .= reshape(1:((nc.x)*(nc.y)), nc.x, nc.y) .+ maximum(Num.y.v)
    Num.y.c[ 2:end-1,2:end-1] .= reshape(1:((nc.x)*(nc.y)), nc.x, nc.y) .+ maximum(Num.x.c)
    Num.p.ex[2:end-1,2:end-1] .= reshape(1:((nc.x)*(nv.y)), nc.x, nv.y) 
    Num.p.ey[2:end-1,2:end-1] .= reshape(1:((nv.x)*(nc.y)), nv.x, nc.y) .+ maximum(Num.p.ex)
    # Initial sparse
    println("Initial Assembly")
    ndofV = maximum(Num.y.c)
    ndofP = maximum(Num.p.ey)
    Kuu   = ExtendableSparseMatrix(ndofV, ndofV)
    Kup   = ExtendableSparseMatrix(ndofV, ndofP)
    iV    = zeros(Int, 18); tV    = zeros(Int, 18); vV   = ones(18)
    iP    = zeros(Int,  4); tP    = zeros(Int,  4); vP   = ones( 4)
    # Loop on vertices
    @time for j in 2:nv.x+1, i in 2:nv.y+1
        iV[1]  = Num.x.v[i,j];     tV[1]  = BC.x.v[i,j]      # C
        iV[2]  = Num.x.v[i-1,j];   tV[2]  = BC.x.v[i-1,j]    # W
        iV[3]  = Num.x.v[i+1,j];   tV[3]  = BC.x.v[i+1,j]    # E
        iV[4]  = Num.x.v[i,j-1];   tV[4]  = BC.x.v[i,j-1]    # S
        iV[5]  = Num.x.v[i,j+1];   tV[5]  = BC.x.v[i,j+1]    # N
        iV[6]  = Num.x.c[i-1,j-1]; tV[6]  = BC.x.c[i-1,j-1]  # SW
        iV[7]  = Num.x.c[i,j-1];   tV[7]  = BC.x.c[i,j-1]    # SE
        iV[8]  = Num.x.c[i-1,j];   tV[8]  = BC.x.c[i-1,j]    # NW
        iV[9]  = Num.x.c[i,j];     tV[9]  = BC.x.c[i,j]      # NE
        #--------------
        iV[10] = Num.y.v[i,j];     tV[10] = BC.y.v[i,j]      # C
        iV[11] = Num.y.v[i-1,j];   tV[11] = BC.y.v[i-1,j]    # W
        iV[12] = Num.y.v[i,j];     tV[12] = BC.y.v[i,j]      # E
        iV[13] = Num.y.v[i,j-1];   tV[13] = BC.y.v[i,j-1]    # S
        iV[14] = Num.y.v[i,j+1];   tV[14] = BC.y.v[i,j+1]    # N
        iV[15] = Num.y.c[i-1,j-1]; tV[15] = BC.y.c[i-1,j-1]  # SW
        iV[16] = Num.y.c[i,j-1];   tV[16] = BC.y.c[i,j-1]    # SE
        iV[17] = Num.y.c[i-1,j];   tV[17] = BC.y.c[i-1,j]    # NW
        iV[18] = Num.y.c[i,j];     tV[18] = BC.y.c[i,j]      # NE
        #--------------
        iP[1]  = Num.p.ex[i-1,j-1];  tP[1]  = BC.p.ex[i-1,j-1]   # W
        iP[2]  = Num.p.ex[i,j-1];    tP[2]  = BC.p.ex[i,j-1]     # E
        iP[3]  = Num.p.ey[i-1,j-1];  tP[3]  = BC.p.ey[i-1,j-1]   # S   
        iP[4]  = Num.p.ey[i-1,j];    tP[4]  = BC.p.ey[i-1,j]     # N
        # Vx
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, iV[1], iV[1], tV[1], 1.0)
        else
            for i in eachindex(vV)
                AddToExtSparse!(Kuu, iV[1], iV[i],  tV[i],  vV[i])
            end
            #--------------
            for i in eachindex(vP)
                AddToExtSparse!(Kup, iV[1], iP[i],  tP[i],  vP[i])
            end
        end
        # Vx
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, iV[10], iV[1], tV[1], 1.0)
        else
            for i in eachindex(vV)
                AddToExtSparse!(Kuu, iV[10], iV[i],  tV[i],  vV[i])
            end
            #--------------
            for i in eachindex(vP)
                AddToExtSparse!(Kup, iV[10], iP[i],  tP[i],  vP[i])
            end
        end
    end 
    # Loop on centroids
    @time for j in 2:nc.x+1, i in 2:nc.y+1
        iV[1]  = Num.x.c[i,j];     tV[1]  = BC.x.c[i,j]      # C
        iV[2]  = Num.x.c[i-1,j];   tV[2]  = BC.x.c[i-1,j]    # W
        iV[3]  = Num.x.c[i+1,j];   tV[3]  = BC.x.c[i+1,j]    # E
        iV[4]  = Num.x.c[i,j-1];   tV[4]  = BC.x.c[i,j-1]    # S
        iV[5]  = Num.x.c[i,j+1];   tV[5]  = BC.x.c[i,j+1]    # N
        iV[6]  = Num.x.v[i,j];     tV[6]  = BC.x.v[i,j]  # SW
        iV[7]  = Num.x.v[i+1,j];   tV[7]  = BC.x.v[i+1,j]    # SE
        iV[8]  = Num.x.v[i,j+1];   tV[8]  = BC.x.v[i,j+1]    # NW
        iV[9]  = Num.x.v[i+1,j+1]; tV[9]  = BC.x.v[i+1,j+1]      # NE
        #--------------
        iV[10] = Num.y.c[i,j];     tV[10] = BC.y.c[i,j]      # C
        iV[11] = Num.y.c[i-1,j];   tV[11] = BC.y.c[i-1,j]    # W
        iV[12] = Num.y.c[i,j];     tV[12] = BC.y.c[i,j]      # E
        iV[13] = Num.y.c[i,j-1];   tV[13] = BC.y.c[i,j-1]    # S
        iV[14] = Num.y.c[i,j+1];   tV[14] = BC.y.c[i,j+1]    # N
        iV[15] = Num.y.v[i,j];     tV[15] = BC.y.v[i,j]  # SW
        iV[16] = Num.y.v[i+1,j];   tV[16] = BC.y.v[i+1,j]    # SE
        iV[17] = Num.y.v[i,j+1];   tV[17] = BC.y.v[i,j+1]    # NW
        iV[18] = Num.y.v[i+1,j+1]; tV[18] = BC.y.v[i+1,j+1]      # NE
        #--------------
        iP[1]  = Num.p.ey[i-1,j];  tP[1]  = BC.p.ey[i-1,j]   # W
        iP[2]  = Num.p.ey[i,j];    tP[2]  = BC.p.ey[i,j]     # E
        iP[3]  = Num.p.ex[i,j-1];  tP[3]  = BC.p.ex[i,j-1]   # S   
        iP[4]  = Num.p.ex[i,j];    tP[4]  = BC.p.ex[i,j]     # N
        # Vx
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, iV[1], iV[1], tV[1], 1.0)
        else
            for i in eachindex(vV)
                AddToExtSparse!(Kuu, iV[1], iV[i],  tV[i],  vV[i])
            end
            #--------------
            for i in eachindex(vP)
                AddToExtSparse!(Kup, iV[1], iP[i],  tP[i],  vP[i])
            end
        end
        # Vx
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, iV[10], iV[1], tV[1], 1.0)
        else
            for i in eachindex(vV)
                AddToExtSparse!(Kuu, iV[10], iV[i],  tV[i],  vV[i])
            end
            #--------------
            for i in eachindex(vP)
                AddToExtSparse!(Kup, iV[10], iP[i],  tP[i],  vP[i])
            end
        end
    end 
    flush!(Kuu), flush!(Kup)
    return Kup
end

Main_2D_DI()