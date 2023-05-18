# Deformed free surface
using FullStaggeredGrid
using Printf, ExtendableSparse, LinearAlgebra, MAT
import SparseArrays:spdiagm, dropzeros!
import Plots:spy
@views   avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
@views h(x,A,σ,b,x0)    = A*exp(-(x-x0)^2/σ^2) + b
@views dhdx(x,A,σ,b,x0) = -2*x/σ^2*A*exp(-(x-x0).^2/σ^2)
@views y_coord(y,ymin,z0,m)   = (y/ymin)*((z0+m))-z0

#--------------------------------------------------------------------#

function Mesh_y( X, A, x0, σ, b, m, ymin0, ymax0, σy, mesh_def )
    y0    = ymax0
    y     = X[2]
    if mesh_def.swiss_y
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    end
    if mesh_def.topo
        z0    = -(A*exp(-(X[1]-x0)^2/σ^2) + b) # topography height
        y     = (y/ymin0)*((z0+m))-z0          # shift grid vertically
    end
    return y   
end

function Mesh_x( X, A, x0, σ, b, m, xmin0, xmax0, σx, mesh_def )
   if mesh_def.swiss_x
        xmin1 = (sinh.( σx.*(xmin0.-x0) ))
        xmax1 = (sinh.( σx.*(xmax0.-x0) ))
        sx    = (xmax0-xmin0)/(xmax1-xmin1)
        x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0
    else
        x = X[1]
    end
    return x
end

function Main_2D_DI()
    # Physics
    x          = (min=-3.0, max=3.0)  
    y          = (min=-5.0, max=0.0)
    ε̇bg        = -0.0
    rad        = 0.5
    y0         = -2.0*1.0
    g          = (x = 0., z=-1.0)
    inclusion  = false
    symmetric  = false
    adapt_mesh = true
    mesh_def   = ( swiss_x=true, swiss_y=true, topo=true)
    # Boundary conditions
    BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)
    BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)

    # x          = (min=-3.0, max=3.0)  
    # y          = (min=-3.0, max=3.0)
    # ε̇bg        = -1.0
    # rad        = 0.5
    # y0         = -2*0.
    # g          = (x = 0., z=-1.0*0.0)
    # inclusion  = true
    # symmetric  = false
    # adapt_mesh = true
    # mesh_def   = ( swiss_x=true, swiss_y=true, topo=false)
    # # Boundary conditions
    # BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    # BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    ##############################
    solve      = true
    comp       = false
    PS         = true
    # Numerics
    nc         = (x=71,     y=71   )  # numerical grid resolution
    nv         = (x=nc.x+1, y=nc.y+1) # numerical grid resolution
    solver     = :DI
    # fact       = :Cholesky
    fact       = :LU
    ϵ          = 1e-8          # nonlinear tolerance
    iterMax    = 20            # max number of iters
    penalty    = 1e5
    # Preprocessing
    Δ          = (x=(x.max-x.min)/nc.x, y=(y.max-y.min)/nc.y, t=1.0)
    # Array initialisation
    V        = ( x   = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2)),  # v: vertices --- c: centroids
                 y   = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2)) ) 
    ε̇        = ( xx  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),  # ex: x edges --- ey: y edges
                 yy  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 xy  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)) )
    τ        = ( xx  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),  # ex: x edges --- ey: y edges
                 yy  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 xy  = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)) )
    ∇v       = (        ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)  )
    P        = (        ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)  )
    P0       = (        ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)  )
    D        = ( v11 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v12 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),  
                 v13 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v21 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v22 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v23 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v31 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v32 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
                 v33 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)), )
    ∂ξ       = ( ∂x  = (v  =  ones(nv.x+2, nv.y+2), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y+2), ey =  ones(nv.x+2,   nc.y+2)), 
                 ∂y  = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)) )
    ∂η       = ( ∂x  = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)), 
                 ∂y  = (v  =  ones(nv.x+2, nv.y+2), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y+2), ey =  ones(nv.x+2,   nc.y+2)) ) 
    R        = ( x   = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2)), 
                 y   = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2)),
                 p   = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2, nc.y+2)) ) 
    ρ        = ( (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2)) )
    η        = (  ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2, nc.y+2)  )
    K        = (  ex =  ones(nc.x+2, nv.y+2), ey =  ones(nv.x+2, nc.y+2)  )
    β        = (  ex = zeros(Bool, nc.x+2, nv.y+2), ey = zeros(Bool, nv.x+2, nc.y+2)  )
    BC       = (x = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                y = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                p = (ex = -1*ones(Int, nc.x+2, nv.y+2),   ey = -1*ones(Int, nv.x+2, nc.y+2)), 
                ε̇ = (ex = -1*ones(Int, nc.x+2, nv.y+2),   ey = -1*ones(Int, nv.x+2, nc.y+2)),
                C0=zeros(nc.x+2), C1=zeros(nc.x+2), C2=zeros(nc.x+2),
                D0=zeros(nc.x+2), D1=zeros(nc.x+2), D2=zeros(nc.x+2), ∂h∂x=zeros(nc.x+2) )
    Num      = ( x   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                 y   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                 p   = (ex = -1*ones(Int, nc.x+2, nv.y), ey = -1*ones(Int, nv.x,   nc.y+2)) )
    # Fine mesh
    xxv, yyv    = LinRange(x.min-Δ.x, x.max+Δ.x, 2nc.x+5), LinRange(y.min-Δ.y, y.max+Δ.y, 2nc.y+5)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])    
    ∂ξ∂x =  ones(2nc.x+5, 2nc.y+5)
    ∂ξ∂y = zeros(2nc.x+5, 2nc.y+5)
    ∂η∂x = zeros(2nc.x+5, 2nc.y+5)
    ∂η∂y =  ones(2nc.x+5, 2nc.y+5)
    hx   = zeros(2nc.x+5, 2nc.y+5)

    if adapt_mesh
        x0     = (x.max + x.min)/2
        m      = y.min
        Amp    = 1.0
        σ      = 0.5
        σx     = 0.5
        σy     = 0.5
        ϵ      = 1e-7
        # copy initial y
        x_ini  = copy(xv4)
        y_ini  = copy(yv4)
        X_msh  = zeros(2)
        # Compute slope
        hx .= -dhdx.(x_ini, Amp, σ, y.max, x0)
        # Deform mesh
        for i in eachindex(x_ini)          
            X_msh[1] = x_ini[i]
            X_msh[2] = y_ini[i]     
            xv4[i]   = Mesh_x( X_msh,  Amp, x0, σ, y.max, m, x.min, x.max, σx, mesh_def )
            yv4[i]   = Mesh_y( X_msh,  Amp, x0, σ, y.max, m, y.min, y.max, σy, mesh_def )
        end
        BC.∂h∂x  .=   hx[2:2:end-1,end-1]
        # # Compute forward transformation
        # params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=x.min, xmax=x.max, ymin=y.min, ymax=y.max, σx=σx, σy=σy, ϵ=ϵ)
        # ∂x4    = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        # ∂y4    = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        # ComputeForwardTransformation!( Mesh_x, Mesh_y, ∂x4, ∂y4, x_ini, y_ini, X_msh, Amp, x0, σ, m, x.min, x.max, y.min, y.max, σx, σy, ϵ, mesh_def)
        # # Solve for inverse transformation
        # ∂ξ4 = (∂x=∂ξ∂x, ∂y=∂ξ∂y); ∂η4 = (∂x=∂η∂x, ∂y=∂η∂y)
        # InverseJacobian!(∂ξ4,∂η4,∂x4,∂y4)
        # ∂ξ∂x .= ∂ξ4.∂x; ∂ξ∂y .= ∂ξ4.∂y
        # ∂η∂x .= ∂η4.∂x; ∂η∂y .= ∂η4.∂y
    end

    xv2_1, yv2_1 = xv4[3:2:end-2,3:2:end-2  ], yv4[3:2:end-2,3:2:end-2  ]
    xv2_2, yv2_2 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,1:2:end-1  ]
    x = merge(x, (v=xv4[1:2:end-0,1:2:end-0], c=xv4[2:2:end-1,2:2:end-1], ex=xv4[2:2:end-1,1:2:end-0  ], ey=xv4[1:2:end-0,2:2:end-1]) ) 
    y = merge(y, (v=yv4[1:2:end-0,1:2:end-0], c=yv4[2:2:end-1,2:2:end-1], ex=yv4[2:2:end-1,1:2:end-0  ], ey=yv4[1:2:end-0,2:2:end-1]) ) 

    # Grid subsets
    xv2_1, yv2_1 = xv4[3:2:end-2,3:2:end-2  ], yv4[3:2:end-2,3:2:end-2  ]
    xv2_2, yv2_2 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,1:2:end-1  ]
    xve2 = xv4[1:2:end-0,1:2:end-0]; yve2 = yv4[1:2:end-0,1:2:end-0]
    xce2 = xv4[2:2:end-1,2:2:end-1]; yce2 = yv4[2:2:end-1,2:2:end-1]
    xex2 = xv4[2:2:end-1,1:2:end-0]; yex2 = yv4[2:2:end-1,1:2:end-0]
    xey2 = xv4[1:2:end-0,2:2:end-1]; yey2 = yv4[1:2:end-0,2:2:end-1]

    ∂x       = ( ∂ξ   = (v  =  ones(nv.x+2, nv.y+2), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y+2), ey =  ones(nv.x+2,   nc.y+2)), 
                 ∂η  = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)) )
    ∂y       = ( ∂ξ  = (v  = zeros(nv.x+2, nv.y+2), c  = zeros(nc.x+2, nc.y+2), ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)), 
                 ∂η  = (v  =  ones(nv.x+2, nv.y+2), c  =  ones(nc.x+2, nc.y+2), ex =  ones(nc.x+2, nv.y+2), ey =  ones(nv.x+2,   nc.y+2)) ) 

    ∂x.∂ξ.ex            .= (xve2[2:end,:] .- xve2[1:end-1,:]) ./ Δ.x
    ∂x.∂ξ.ey[2:end-1,:] .= (xce2[2:end,:] .- xce2[1:end-1,:]) ./ Δ.x
    ∂x.∂ξ.v[2:end-1,:]  .= (xex2[2:end,:] .- xex2[1:end-1,:]) ./ Δ.x
    ∂x.∂ξ.c             .= (xey2[2:end,:] .- xey2[1:end-1,:]) ./ Δ.x
    ∂y.∂ξ.ex            .= (yve2[2:end,:] .- yve2[1:end-1,:]) ./ Δ.x
    ∂y.∂ξ.ey[2:end-1,:] .= (yce2[2:end,:] .- yce2[1:end-1,:]) ./ Δ.x
    ∂y.∂ξ.v[2:end-1,:]  .= (yex2[2:end,:] .- yex2[1:end-1,:]) ./ Δ.x
    ∂y.∂ξ.c             .= (yey2[2:end,:] .- yey2[1:end-1,:]) ./ Δ.x

    ∂x.∂η.ex[:,2:end-1] .= (xce2[:,2:end] .- xce2[:,1:end-1]) ./ Δ.y
    ∂x.∂η.ey            .= (xve2[:,2:end] .- xve2[:,1:end-1]) ./ Δ.y
    ∂x.∂η.v[:,2:end-1]  .= (xey2[:,2:end] .- xey2[:,1:end-1]) ./ Δ.y
    ∂x.∂η.c             .= (xex2[:,2:end] .- xex2[:,1:end-1]) ./ Δ.y
    ∂y.∂η.ex[:,2:end-1] .= (yce2[:,2:end] .- yce2[:,1:end-1]) ./ Δ.y
    ∂y.∂η.ey            .= (yve2[:,2:end] .- yve2[:,1:end-1]) ./ Δ.y
    ∂y.∂η.v[:,2:end-1]  .= (yey2[:,2:end] .- yey2[:,1:end-1]) ./ Δ.y
    ∂y.∂η.c             .= (yex2[:,2:end] .- yex2[:,1:end-1]) ./ Δ.y 

    InverseJacobian2!(∂ξ.∂x.ex,∂η.∂x.ex,∂ξ.∂y.ex,∂η.∂y.ex, ∂x.∂ξ.ex,∂x.∂η.ex,∂y.∂ξ.ex,∂y.∂η.ex)
    InverseJacobian2!(∂ξ.∂x.ey,∂η.∂x.ey,∂ξ.∂y.ey,∂η.∂y.ey, ∂x.∂ξ.ey,∂x.∂η.ey,∂y.∂ξ.ey,∂y.∂η.ey)
    InverseJacobian2!(∂ξ.∂x.v ,∂η.∂x.v ,∂ξ.∂y.v ,∂η.∂y.v , ∂x.∂ξ.v ,∂x.∂η.v ,∂y.∂ξ.v ,∂y.∂η.v )
    InverseJacobian2!(∂ξ.∂x.c ,∂η.∂x.c ,∂ξ.∂y.c ,∂η.∂y.c , ∂x.∂ξ.c ,∂x.∂η.c ,∂y.∂ξ.c ,∂y.∂η.c )

    # ∂ξ.∂x.ex .= ∂ξ∂x[2:2:end-1,1:2:end-0]; ∂ξ.∂x.v .= ∂ξ∂x[1:2:end-0,1:2:end-0]
    # ∂ξ.∂x.ey .= ∂ξ∂x[1:2:end-0,2:2:end-1]; ∂ξ.∂x.c .= ∂ξ∂x[2:2:end-1,2:2:end-1]
    # ∂ξ.∂y.ex .= ∂ξ∂y[2:2:end-1,1:2:end-0]; ∂ξ.∂y.v .= ∂ξ∂y[1:2:end-0,1:2:end-0]
    # ∂ξ.∂y.ey .= ∂ξ∂y[1:2:end-0,2:2:end-1]; ∂ξ.∂y.c .= ∂ξ∂y[2:2:end-1,2:2:end-1]
    # ∂η.∂x.ex .= ∂η∂x[2:2:end-1,1:2:end-0]; ∂η.∂x.v .= ∂η∂x[1:2:end-0,1:2:end-0]
    # ∂η.∂x.ey .= ∂η∂x[1:2:end-0,2:2:end-1]; ∂η.∂x.c .= ∂η∂x[2:2:end-1,2:2:end-1]
    # ∂η.∂y.ex .= ∂η∂y[2:2:end-1,1:2:end-0]; ∂η.∂y.v .= ∂η∂y[1:2:end-0,1:2:end-0]
    # ∂η.∂y.ey .= ∂η∂y[1:2:end-0,2:2:end-1]; ∂η.∂y.c .= ∂η∂y[2:2:end-1,2:2:end-1]
    
    @printf("__________\n")
    @printf("min(∂ξ∂x) = %1.6f --- max(∂ξ∂x) = %1.6f\n", minimum(∂ξ.∂x.c), maximum(∂ξ.∂x.c))
    @printf("min(∂ξ∂y) = %1.6f --- max(∂ξ∂y) = %1.6f\n", minimum(∂ξ.∂y.c), maximum(∂ξ.∂y.c))
    @printf("min(∂η∂x) = %1.6f --- max(∂η∂x) = %1.6f\n", minimum(∂η.∂x.c), maximum(∂η.∂x.c))
    @printf("min(∂η∂y) = %1.6f --- max(∂η∂y) = %1.6f\n", minimum(∂η.∂y.c), maximum(∂η.∂y.c))

    # Velocity
    V.x.v .= -PS*ε̇bg.*x.v .+ (1-PS)*ε̇bg.*y.v; V.x.c .= -PS*ε̇bg.*x.c .+ (1-PS)*ε̇bg.*y.c
    V.y.v .=  PS*ε̇bg.*y.v;                    V.y.c .=  PS*ε̇bg.*y.c
    # Viscosity
    η.ex  .=  1.0; η.ey  .=  1.0
    β.ex  .= true; β.ey  .= true
    K.ex  .=  1e5; K.ey  .=  1e5
    if inclusion
        η.ex[x.ex.^2 .+ (y.ex.-y0).^2 .< rad] .= 100.
        η.ey[x.ey.^2 .+ (y.ey.-y0).^2 .< rad] .= 100.
    end
    D.v11.ex .= 2 .* η.ex;      D.v11.ey .= 2 .* η.ey
    D.v22.ex .= 2 .* η.ex;      D.v22.ey .= 2 .* η.ey
    D.v33.ex .= 2 .* η.ex;      D.v33.ey .= 2 .* η.ey
    # Density
    ρ.v  .= 1.0; ρ.c  .= 1.0
    if inclusion
        ρ.v[x.v.^2 .+ (y.v.-y0).^2 .< rad] .= 2.
        ρ.c[x.c.^2 .+ (y.c.-y0).^2 .< rad] .= 2.
    end
    BC.x.v[2:end-1,2:end-1]  .= 0 # inner points
    BC.y.v[2:end-1,2:end-1]  .= 0 # inner points
    BC.x.c[2:end-1,2:end-1]  .= 0 # inner points
    BC.y.c[2:end-1,2:end-1]  .= 0 # inner points
    BC.p.ex[2:end-1,2:end-1] .= 0 # inner points
    BC.p.ey[2:end-1,2:end-1] .= 0 # inner points
    BC.ε̇.ey[2:end-1,2:end-1] .= 0 # inner points
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
    elseif BCVx.North == :FreeSurface 
        BC.x.v[3:end-2,end-1] .= 2
        BC.x.c[2:end-1,end-1] .= 2
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
    elseif BCVy.North == :FreeSurface 
        BC.y.v[3:end-2,end-1] .= 2
        BC.y.c[2:end-1,end-1] .= 2
    end
    # Pressure
    if BCVx.North == :FreeSurface || BCVy.North == :FreeSurface 
        BC.p.ex[2:end-1,end-1] .= 2 # modified equation
        BC.ε̇.ey[2:end-1,end  ] .= 2 # label points just outside - check if used (?)
    end
    # Free surface coefficients
    UpdateFreeSurfaceCoefficients!( BC, D, ∂ξ, ∂η )
    DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    LinearMomentumResidual!( R, ∇v, τ, P, P0, β, K, ρ, g, ∂ξ, ∂η, Δ, BC, comp, symmetric )
    # Display residuals
    err_x = max(norm(R.x.v )/length(R.x.v ), norm(R.x.c )/length(R.x.c ))
    err_y = max(norm(R.y.v )/length(R.y.v ), norm(R.y.c )/length(R.y.c ))
    err_p = max(norm(R.p.ex)/length(R.p.ex), norm(R.p.ey)/length(R.p.ey))
    @printf("Rx = %2.9e\n", err_x )
    @printf("Ry = %2.9e\n", err_y )
    @printf("Rp = %2.9e\n", err_p )
    # if err_x<ϵ && err_y<ϵ && err_p<ϵ
    #     @printf("Converged!\n")
    #     break
    # end
    # Numbering
    Num      = ( x   = (v  = -1*ones(Int, nv.x+2, nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                 y   = (v  = -1*ones(Int, nv.x+2, nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                 p   = (ex = -1*ones(Int, nc.x+2, nv.y+2), ey = -1*ones(Int, nv.x+2, nc.y+2)) )
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
    Kpu   = ExtendableSparseMatrix(ndofP, ndofV)
    Kpp   = ExtendableSparseMatrix(ndofP, ndofP)
    Kppi  = ExtendableSparseMatrix(ndofP, ndofP)
    @time AssembleKuuKupKpu!(Kuu, Kup, Kpu, Kpp, Kppi, Num, BC, D, β, K, ∂ξ, ∂η, Δ, nc, nv, penalty, comp, symmetric)

    Kuu_PC   = ExtendableSparseMatrix(ndofV, ndofV)
    Kup_PC   = ExtendableSparseMatrix(ndofV, ndofP)
    Kpu_PC   = ExtendableSparseMatrix(ndofP, ndofV)
    Kpp_PC   = ExtendableSparseMatrix(ndofP, ndofP)
    Kppi_PC  = ExtendableSparseMatrix(ndofP, ndofP)
    @time AssembleKuuKupKpu!(Kuu, Kup, Kpu, Kpp, Kppi, Num, BC, D, β, K, ∂ξ, ∂η, Δ, nc, nv, penalty, comp, true)

    KuuJ  = dropzeros!(Kuu.cscmatrix)
    KupJ  = dropzeros!(Kup.cscmatrix)
    KpuJ  = dropzeros!(Kpu.cscmatrix)
    KppJ  = dropzeros!(Kpp.cscmatrix)
    KppiJ = dropzeros!(Kppi.cscmatrix)

    KuuPC  = dropzeros!(Kuu_PC.cscmatrix)

    nV   = maximum(Num.y.c)
    nP   = maximum(Num.p.ey)
    fu   = zeros(nV)
    fp   = zeros(nP)
    fu[Num.x.v[2:end-1,2:end-1]]  .= R.x.v[2:end-1,2:end-1]
    fu[Num.y.v[2:end-1,2:end-1]]  .= R.y.v[2:end-1,2:end-1]
    fu[Num.x.c[2:end-1,2:end-1]]  .= R.x.c[2:end-1,2:end-1]
    fu[Num.y.c[2:end-1,2:end-1]]  .= R.y.c[2:end-1,2:end-1]
    fp[Num.p.ex[2:end-1,2:end-1]] .= R.p.ex[2:end-1,2:end-1]
    fp[Num.p.ey[2:end-1,2:end-1]] .= R.p.ey[2:end-1,2:end-1]

    if solver==:DI
        # Decoupled solve
        t = @elapsed Kuu_SC = KuuJ .- KupJ*(KppiJ*KpuJ)
        @printf("Build Kuu_SC took = %02.2e s\n", t)
        if fact == :Cholesky 
            # t = @elapsed Kf    = cholesky(Hermitian( (Kuu_SC) ))
            Kuu_SC1 = KuuPC .- KupJ*(KppiJ*KpuJ)
            t = @elapsed Kf    = cholesky(Hermitian( Kuu_SC1 ))
            # t = @elapsed Kf    = cholesky(Hermitian( 0.5.*(Kuu_SC.+Kuu_SC') ))
            @printf("Cholesky took = %02.2e s\n", t)
        elseif fact == :LU 
            t = @elapsed Kf    = lu(Kuu_SC)
            @printf("LU took = %02.2e s\n", t)
        end
        δu    = zeros(nV, 1)
        ru    = zeros(nV, 1)
        fusc  = zeros(nV, 1)
        δp    = zeros(nP, 1)
        rp    = zeros(nP, 1)
        ######################################
        # Iterations
        for rit=1:20
            ru   .= fu .- KuuJ*δu .- KupJ*δp
            rp   .= fp .- KpuJ*δu .- KppJ*δp
            nrmu, nrmp = norm(ru), norm(rp)
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
            if nrmu/sqrt(length(ru)) < ϵ && nrmp/sqrt(length(ru)) < ϵ
                break
            end
            fusc .= fu .- KupJ*(KppiJ*fp .+ δp)
            δu   .= Kf\fusc
            δp  .+= KppiJ*(fp .- KpuJ*δu .- KppJ*δp)
        end

    # nV   = maximum(Num.y.c)
    # nP   = maximum(Num.p.ey)
    # fu   = zeros(nV)
    # fp   = zeros(nP)
    # fu[Num.x.v[2:end-1,2:end-1]]  .= R.x.v[2:end-1,2:end-1]
    # fu[Num.y.v[2:end-1,2:end-1]]  .= R.y.v[2:end-1,2:end-1]
    # fu[Num.x.c[2:end-1,2:end-1]]  .= R.x.c[2:end-1,2:end-1]
    # fu[Num.y.c[2:end-1,2:end-1]]  .= R.y.c[2:end-1,2:end-1]
    # fp[Num.p.ex[2:end-1,2:end-1]] .= R.p.ex[2:end-1,2:end-1]
    # fp[Num.p.ey[2:end-1,2:end-1]] .= R.p.ey[2:end-1,2:end-1]

    # @printf("Solving linear system\n")

    # # KSP_GCR_Jacobian!( u, Kuuscj, fusc, 5e-6, 2, Kf, f, v, s, val, VV, SS, restart  )

    # Kpp = spdiagm( zeros(nP) )
    # # Decoupled solve
    # if comp==false 
    #     npdof = maximum(Num.p.ey)
    #     coef  = zeros(npdof)
    #     for i in eachindex(coef)
    #         coef[i] = penalty#.*mesh.ke./mesh.Ω
    #     end
    #     Kppi  = spdiagm(coef)
    # else
    #     Kppi  = spdiagm( 1.0 ./ diag(Kpp) )
    # end
    # @printf("Done with Kppi\n")
    # Kuu_SC = Kuuj - Kupj*(Kppi*Kpuj)
    # @show typeof(Kuu_SC)
    # @printf("Done with Kuu_SC\n")
    # K_PC   = 0.5.*(Kuu_SC' .+ Kuu_SC)
    # t = @elapsed Kf    = cholesky(Hermitian(K_PC), check=false)
    # # t = @elapsed Kf    = cholesky((Kuu_SC))
    # # t = @elapsed Kf = lu(Kuu_SC)
    # @printf("Cholesky took = %02.2e s\n", t)


    # if solver == :PH_Cholesky
    #     t = @elapsed Kf    = cholesky((Kuu_SC))
    #     @printf("Cholesky took = %02.2e s\n", t)
    # elseif solver == :PH_LU 
    #     t = @elapsed Kf    = lu(Kuu_SC)
    #     @printf("LU took = %02.2e s\n", t)
    # end

    # restart = 20
    # f       = zeros(Float64, length(fu))
    # v       = zeros(Float64, length(fu))
    # s       = zeros(Float64, length(fu))
    # val     = zeros(Float64, restart)
    # VV      = zeros(Float64, (length(fu), restart) )  # !!!!!!!!!! allocate in the right sense :D
    # SS      = zeros(Float64, (length(fu), restart) )

    # δu    = zeros(nV, 1)
    # ru    = zeros(nV, 1)
    # fusc  = zeros(nV, 1)
    # δp    = zeros(nP, 1)
    # rp    = zeros(nP, 1)
    # ######################################
    # # Iterations
    # for rit=1:20
    #     ru   .= fu .- Kuuj*δu .- Kupj*δp
    #     rp   .= fp .- Kpuj*δu .- Kppj*δp
    #     nrmu, nrmp = norm(ru), norm(rp)
    #     @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
    #     if nrmu/sqrt(length(ru)) < ϵ && nrmp/sqrt(length(ru)) < ϵ
    #         break
    #     end
    #     fusc .= fu  .- Kupj*(Kppi*fp .+ δp)
    #     # δu   .= Kf\fusc
    #     KSP_GCR_Jacobian!( δu, Kuu_SC, fusc, 5e-3, 2, Kf, f, v, s, val, VV, SS, restart  )
    #     δp  .+= Kppi*(fp .- Kpuj*δu .- Kppj*δp)
    # end


        # @time if symmetric
        # Kpp = spdiagm( zeros(nP) )
        # # Decoupled solve
        # if comp==false 
        #     npdof = maximum(Num.p.ey)
        #     coef  = zeros(npdof)
        #     for i in eachindex(coef)
        #         coef[i] = penalty#.*mesh.ke./mesh.Ω
        #     end
        #     Kppi  = spdiagm(coef)
        # else
        #     Kppi  = spdiagm( 1.0 ./ diag(Kpp) )
        # end
        # @printf("Done with Kppi\n")
        # Kuu_SC = Kuuj - Kupj*(Kppi*Kpuj)
        # @printf("Done with Kuu_SC\n")
        # if solver == :PH_Cholesky
        #     t = @elapsed Kf    = cholesky((Kuu_SC))
        #     @printf("Cholesky took = %02.2e s\n", t)
        # elseif solver == :PH_LU 
        #     t = @elapsed Kf    = lu(Kuu_SC)
        #     @printf("LU took = %02.2e s\n", t)
        # end
        # δu    = zeros(nV, 1)
        # ru    = zeros(nV, 1)
        # fusc  = zeros(nV, 1)
        # δp    = zeros(nP, 1)
        # rp    = zeros(nP, 1)
        # ######################################
        # # Iterations
        # for rit=1:20
        #     ru   .= fu .- Kuuj*δu .- Kupj*δp
        #     rp   .= fp .- Kpuj*δu .- Kppj*δp
        #     nrmu, nrmp = norm(ru), norm(rp)
        #     @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
        #     if nrmu/sqrt(length(ru)) < ϵ && nrmp/sqrt(length(ru)) < ϵ
        #         break
        #     end
        #     fusc .= fu  .- Kupj*(Kppi*fp .+ δp)
        #     δu   .= Kf\fusc
        #     δp  .+= Kppi*(fp .- Kpuj*δu .- Kppj*δp)
        # end
        # Global Newton update
        V.x.v[2:end-1, 2:end-1] .-= δu[Num.x.v[2:end-1, 2:end-1]]
        V.y.v[2:end-1, 2:end-1] .-= δu[Num.y.v[2:end-1, 2:end-1]]
        V.x.c[2:end-1, 2:end-1] .-= δu[Num.x.c[2:end-1, 2:end-1]]
        V.y.c[2:end-1, 2:end-1] .-= δu[Num.y.c[2:end-1, 2:end-1]]
        P.ex[2:end-1, 2:end-1]  .-= δp[Num.p.ex[2:end-1, 2:end-1]]
        P.ey[2:end-1, 2:end-1]  .-= δp[Num.p.ey[2:end-1, 2:end-1]]
    else
        #################
        # Kppj[1,:] .= 0.
        # Kppj[1,1] = 1.0
        # Kppj[end,:] .= 0.
        # Kppj[end,end] = 1.0

        J  = [KuuJ KupJ; KpuJ KppJ]
        f  = [fu; fp]
        # Solve
        δx = J\f
        # Extract
        V.x.v[2:end-1, 2:end-1] .-= δx[Num.x.v[2:end-1, 2:end-1]]
        V.y.v[2:end-1, 2:end-1] .-= δx[Num.y.v[2:end-1, 2:end-1]]
        V.x.c[2:end-1, 2:end-1] .-= δx[Num.x.c[2:end-1, 2:end-1]]
        V.y.c[2:end-1, 2:end-1] .-= δx[Num.y.c[2:end-1, 2:end-1]]
        P.ex[2:end-1, 2:end-1]  .-= δx[Num.p.ex[2:end-1, 2:end-1].+maximum(Num.y.c)]
        P.ey[2:end-1, 2:end-1]  .-= δx[Num.p.ey[2:end-1, 2:end-1].+maximum(Num.y.c)]
    end
  #--------------------


    DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    LinearMomentumResidual!( R, ∇v, τ, P, P0, β, K, ρ, g, ∂ξ, ∂η, Δ, BC, comp, symmetric )
    # Display residuals
    err_x = max(norm(R.x.v )/length(R.x.v ), norm(R.x.c )/length(R.x.c ))
    err_y = max(norm(R.y.v )/length(R.y.v ), norm(R.y.c )/length(R.y.c ))
    err_p = max(norm(R.p.ex)/length(R.p.ex), norm(R.p.ey)/length(R.p.ey))
    @printf("Rx = %2.9e\n", err_x )
    @printf("Ry = %2.9e\n", err_y )
    @printf("Rp = %2.9e\n", err_p )


    @printf("%2.2e --- %2.2e\n",  minimum(R.x.v),  maximum(R.x.v))
    @printf("%2.2e --- %2.2e\n",  minimum(R.x.c),  maximum(R.x.c))
    @printf("%2.2e --- %2.2e\n",  minimum(R.y.v),  maximum(R.y.v))
    @printf("%2.2e --- %2.2e\n",  minimum(R.y.c),  maximum(R.y.c))
    @printf("%2.2e --- %2.2e\n",  minimum(R.p.ex),  maximum(R.p.ex))
    @printf("%2.2e --- %2.2e\n",  minimum(R.p.ey),  maximum(R.p.ey))
    #-----------

    # Visualise
    vertx = [  xv2_1[1:end-1,1:end-1][:]  xv2_1[2:end-0,1:end-1][:]  xv2_1[2:end-0,2:end-0][:]  xv2_1[1:end-1,2:end-0][:] ] 
    verty = [  yv2_1[1:end-1,1:end-1][:]  yv2_1[2:end-0,1:end-1][:]  yv2_1[2:end-0,2:end-0][:]  yv2_1[1:end-1,2:end-0][:] ] 
    sol   = ( vx=V.x.c[2:end-1,2:end-1][:], vy=V.y.c[2:end-1,2:end-1][:], p= avWESN(P.ex[2:end-1,2:end-1], P.ey[2:end-1,2:end-1])[:])
    PatchPlotMakieBasic(vertx, verty, sol, x.min, x.max, y.min, y.max, write_fig=false)
 
    ####################
    # if swiss
    #     file = matopen(string(@__DIR__,"/../scripts_2D_PT/output_FS_topo_swiss.mat"))
    # else
    #     file = matopen(string(@__DIR__,"/../scripts_2D_PT/output_FS_topo.mat"))
    # end
    # Vx_1 = read(file, "Vx_1") 
    # Vx_2 = read(file, "Vx_2")
    # Vy_1 = read(file, "Vy_1") 
    # Vy_2 = read(file, "Vy_2")
    # P_1  = read(file, "P_1" ) 
    # P_2  = read(file, "P_2" )
    # duNddudx = read(file, "duNddudx")
    # duNddvdx = read(file, "duNddvdx")
    # duNdP    = read(file, "duNdP"   )   
    # dvNddudx = read(file, "dvNddudx")
    # dvNddvdx = read(file, "dvNddvdx")
    # dvNdP    = read(file, "dvNdP"   )  
    # dkdx = read(file,  "dkdx")
    # dkdy = read(file,  "dkdy")
    # dedx = read(file,  "dedx")
    # dedy = read(file,  "dedy")
    # h_x  = read(file,  "hx" )

    # # display(dkdx')
    # # display(dkdy')
    # # for i=1:length(h_x)
    # #     print(h_x[i], ' ' )
    # # end
    # # print("\n")
    # # for i=1:length(h_x)
    # #     print(dedx[i], ' ' )
    # # end
    # # print("\n")

    # close(file)

    # V.x.v[2:end-1,2:end-1] .= Vx_1
    # V.x.c .= Vx_2
    # V.y.v[2:end-1,2:end-1] .= Vy_1
    # V.y.c .= Vy_2
    # P.ex[2:end-1,2:end-1] .= P_1
    # P.ey[2:end-1,2:end-0] .= P_2


    # # display(norm(BC.C0[2:end-1].-duNdP./Δ.y))
    # # display(norm(BC.C1[2:end-1].-duNddudx./Δ.y))
    # # display(norm(BC.C2[2:end-1].-duNddvdx./Δ.y))

    # # display(norm(BC.D0[2:end-1].-dvNdP./Δ.y))
    # # display(norm(BC.D1[2:end-1].-dvNddudx./Δ.y))
    # # display(norm(BC.D2[2:end-1].-dvNddvdx./Δ.y))

    # # BC.C0[2:end-1] .= duNdP./Δ.y
    # # BC.C1[2:end-1] .= duNddudx./Δ.y
    # # BC.C2[2:end-1] .= duNddvdx./Δ.y
    # # BC.D0[2:end-1] .= dvNdP./Δ.y
    # # BC.D1[2:end-1] .= dvNddudx./Δ.y
    # # BC.D2[2:end-1] .= dvNddvdx./Δ.y

    # DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    # LinearMomentumResidual!( R, ∇v, τ, P, ρ, g, ∂ξ, ∂η, Δ, BC )
    # # Display residuals
    # err_x = max(norm(R.x.v )/length(R.x.v ), norm(R.x.c )/length(R.x.c ))
    # err_y = max(norm(R.y.v )/length(R.y.v ), norm(R.y.c )/length(R.y.c ))
    # err_p = max(norm(R.p.ex)/length(R.p.ex), norm(R.p.ey)/length(R.p.ey))
    # @printf("Rx = %2.9e\n", err_x )
    # @printf("Ry = %2.9e\n", err_y )
    # @printf("Rp = %2.9e\n", err_p )

    # @printf("%2.2e --- %2.2e\n",  minimum(R.x.v),  maximum(R.x.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(R.x.c),  maximum(R.x.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(R.y.v),  maximum(R.y.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(R.y.c),  maximum(R.y.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(R.p.ex),  maximum(R.p.ex))
    # @printf("%2.2e --- %2.2e\n",  minimum(R.p.ey),  maximum(R.p.ey))
end

Main_2D_DI()
