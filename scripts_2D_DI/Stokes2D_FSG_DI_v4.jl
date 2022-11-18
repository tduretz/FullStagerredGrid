# Deformed free surface
using FullStaggeredGrid
using Printf, ExtendableSparse, LinearAlgebra, MAT
import SparseArrays:spdiagm
import Plots:spy
@views   avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
@views h(x,A,σ,b,x0)    = A*exp(-(x-x0)^2/σ^2) + b
@views dhdx(x,A,σ,b,x0) = -2*x/σ^2*A*exp(-(x-x0).^2/σ^2)
@views y_coord(y,ymin,z0,m)   = (y/ymin)*((z0+m))-z0
function Mesh_y( X, A, x0, σ, b, m, ymin0, ymax0, σy, mesh_def )
    y0    = ymax0
    if mesh_def.swiss_y
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    else
        y     = X[2]
    end
    if mesh_def.topo
        z0    = -(A*exp(-(X[1]-x0)^2/σ^2) + b) # topography height
        y     = (y/ymin0)*((z0+m))-z0        # shift grid vertically
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
    # x          = (min=-3.0, max=3.0)  
    # y          = (min=-5.0, max=0.0)
    x          = (min=-3.0, max=3.0)  
    y          = (min=-3.0, max=3.0)
    ε̇bg        = -0.0
    rad        = 0.5
    y0         = -2.0*1.0
    g          = (x = 0., z=-1.0)
    inclusion  = true
    adapt_mesh = true
    mesh_def   = ( swiss_x=true, swiss_y=true, topo=false)
    solve      = true
    comp       = false
    PS         = true
    # Numerics
    nc         = (x=6,     y=6   )  # numerical grid resolution
    nv         = (x=nc.x+1, y=nc.y+1) # numerical grid resolution
    solver     = :PH_Cholesky
    ϵ          = 1e-8          # nonlinear tolerance
    iterMax    = 20            # max number of iters
    penalty    = 1e4
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
    BC       = (x = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                y = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                p = (ex = -1*ones(Int, nc.x+2, nv.y+2),     ey = -1*ones(Int, nv.x+2,   nc.y+2)), 
                ε̇ = (ex = -1*ones(Int, nc.x+2, nv.y+2),     ey = -1*ones(Int, nv.x+2,   nc.y+2)),
                C0=zeros(nc.x+2), C1=zeros(nc.x+2), C2=zeros(nc.x+2),
                D0=zeros(nc.x+2), D1=zeros(nc.x+2), D2=zeros(nc.x+2), ∂h∂x=zeros(nc.x+2) )
    Num      = ( x   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                 y   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                 p   = (ex = -1*ones(Int, nc.x+2, nv.y), ey = -1*ones(Int, nv.x,   nc.y+2)) )
    # Fine mesh
    xxv, yyv    = LinRange(x.min-Δ.x, x.max+Δ.x, 2nc.x+5), LinRange(y.min-Δ.y, y.max+Δ.y, 2nc.y+5)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    xv2_1, yv2_1 = xv4[3:2:end-2,3:2:end-2  ], yv4[3:2:end-2,3:2:end-2  ]
    xv2_2, yv2_2 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,1:2:end-1  ]
    # xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    # xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    x = merge(x, (v=xv4[1:2:end-0,1:2:end-0], c=xv4[2:2:end-1,2:2:end-1], ex=xv4[2:2:end-1,1:2:end-0  ], ey=xv4[1:2:end-0,2:2:end-1]) ) 
    y = merge(y, (v=yv4[1:2:end-0,1:2:end-0], c=yv4[2:2:end-1,2:2:end-1], ex=yv4[2:2:end-1,1:2:end-0  ], ey=yv4[1:2:end-0,2:2:end-1]) ) 
    
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
        # Compute forward transformation
        params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=x.min, xmax=x.max, ymin=y.min, ymax=y.max, σx=σx, σy=σy, ϵ=ϵ)
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation!( Mesh_x, Mesh_y, ∂x, ∂y, x_ini, y_ini, X_msh, Amp, x0, σ, m, x.min, x.max, y.min, y.max, σx, σy, ϵ, mesh_def)
        # Solve for inverse transformation
        ∂ξ4 = (∂x=∂ξ∂x, ∂y=∂ξ∂y); ∂η4 = (∂x=∂η∂x, ∂y=∂η∂y)
        InverseJacobian!(∂ξ4,∂η4,∂x,∂y)
        ∂ξ∂x .= ∂ξ4.∂x; ∂ξ∂y .= ∂ξ4.∂y
        ∂η∂x .= ∂η4.∂x; ∂η∂y .= ∂η4.∂y
    end
    # Grid subsets
    xv2_1, yv2_1 = xv4[3:2:end-2,3:2:end-2  ], yv4[3:2:end-2,3:2:end-2  ]
    xv2_2, yv2_2 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,1:2:end-1  ]
    # xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    # xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    ∂ξ.∂x.ex .= ∂ξ∂x[2:2:end-1,1:2:end-0]; ∂ξ.∂x.v .= ∂ξ∂x[1:2:end-0,1:2:end-0]
    ∂ξ.∂x.ey .= ∂ξ∂x[1:2:end-0,2:2:end-1]; ∂ξ.∂x.c .= ∂ξ∂x[2:2:end-1,2:2:end-1]
    ∂ξ.∂y.ex .= ∂ξ∂y[2:2:end-1,1:2:end-0]; ∂ξ.∂y.v .= ∂ξ∂y[1:2:end-0,1:2:end-0]
    ∂ξ.∂y.ey .= ∂ξ∂y[1:2:end-0,2:2:end-1]; ∂ξ.∂y.c .= ∂ξ∂y[2:2:end-1,2:2:end-1]
    ∂η.∂x.ex .= ∂η∂x[2:2:end-1,1:2:end-0]; ∂η.∂x.v .= ∂η∂x[1:2:end-0,1:2:end-0]
    ∂η.∂x.ey .= ∂η∂x[1:2:end-0,2:2:end-1]; ∂η.∂x.c .= ∂η∂x[2:2:end-1,2:2:end-1]
    ∂η.∂y.ex .= ∂η∂y[2:2:end-1,1:2:end-0]; ∂η.∂y.v .= ∂η∂y[1:2:end-0,1:2:end-0]
    ∂η.∂y.ey .= ∂η∂y[1:2:end-0,2:2:end-1]; ∂η.∂y.c .= ∂η∂y[2:2:end-1,2:2:end-1]
    BC.∂h∂x  .=   hx[2:2:end-1,end-1]
    # Velocity
    V.x.v .= -PS*ε̇bg.*x.v .+ (1-PS)*ε̇bg.*y.v; V.x.c .= -PS*ε̇bg.*x.c .+ (1-PS)*ε̇bg.*y.c
    V.y.v .=  PS*ε̇bg.*y.v;                    V.y.c .=  PS*ε̇bg.*y.c
    # Viscosity
    η.ex  .= 1.0; η.ey  .= 1.0
    if inclusion
        # η.ex[x.ex.^2 .+ (y.ex.-y0).^2 .< rad] .= 100.
        # η.ey[x.ey.^2 .+ (y.ey.-y0).^2 .< rad] .= 100.
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
    # Boundary conditions
    BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    # BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)
    # BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)
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
    LinearMomentumResidual!( R, ∇v, τ, P, ρ, g, ∂ξ, ∂η, Δ, BC )
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
    @time AssembleKuuKupKpu!(Kuu, Kup, Kpu, Kpp, Num, BC, D, ∂ξ, ∂η, Δ, nc, nv)

    Kuuj = Kuu.cscmatrix
    Kupj = Kup.cscmatrix
    Kpuj = Kpu.cscmatrix
    Kppj = Kpp.cscmatrix

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

    Kpp = spdiagm( zeros(nP) )

    # Decoupled solve
    if comp==false 
        npdof = maximum(Num.p.ey)
        coef  = zeros(npdof)
        for i in eachindex(coef)
            coef[i] = penalty#.*mesh.ke./mesh.Ω
        end
        Kppi  = spdiagm(coef)
    else
        Kppi  = spdiagm( 1.0 ./ diag(Kpp) )
    end
    Kuu_SC    = Kuuj .- Kupj*(Kppi*Kpuj)
    Kuu_PC = 0.5.*(Kuu .+ Kuu')

    #################
    K  = [Kuuj Kupj; Kpuj Kppj]
    f  = [fu; fp]


    Kdiff =  Kuuj.-Kuuj'
    # display( Kdiff.nzval )

    # display( R.x.v[2:end-1,2:end-1])
    # # if err_x<ϵ && err_y<ϵ && err_p<ϵ
    # #     @printf("Converged!\n")
    # #     break
    # # end
    display(Δ )
    p=spy(Kuuj.-Kuuj', c=:RdBu)
    # # p=spy(Kuuj, c=:RdBu)
    # # p=spy(Kpu, c=:RdBu)
    # # p=spy(Kuu.-Kuu', c=:RdBu,  size=(600,600))
    # # @show dropzeros!(Kuuj.-Kuuj')
    # # @show dropzeros!(Kupj.+Kpuj')
    display(p)
    cholesky(Kuu_PC)


    δx = K\f

    # @show (Kppj)

    V.x.v[2:end-1, 2:end-1] .-= δx[Num.x.v[2:end-1, 2:end-1]]
    V.y.v[2:end-1, 2:end-1] .-= δx[Num.y.v[2:end-1, 2:end-1]]
    V.x.c[2:end-1, 2:end-1] .-= δx[Num.x.c[2:end-1, 2:end-1]]
    V.y.c[2:end-1, 2:end-1] .-= δx[Num.y.c[2:end-1, 2:end-1]]
    P.ex[2:end-1, 2:end-1]  .-= δx[Num.p.ex[2:end-1, 2:end-1].+maximum(Num.y.c)]
    P.ey[2:end-1, 2:end-1]  .-= δx[Num.p.ey[2:end-1, 2:end-1].+maximum(Num.y.c)]

    DevStrainRateStressTensor!( ε̇, τ, P, D, ∇v, V, ∂ξ, ∂η, Δ, BC )
    LinearMomentumResidual!( R, ∇v, τ, P, ρ, g, ∂ξ, ∂η, Δ, BC )
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
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂x.c),  maximum(∂ξ.∂x.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂x.v),  maximum(∂ξ.∂x.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂x.ex),  maximum(∂ξ.∂x.ex))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂x.ey),  maximum(∂ξ.∂x.ey))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂y.c),  maximum(∂ξ.∂y.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂y.v),  maximum(∂ξ.∂y.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂y.ex),  maximum(∂ξ.∂y.ex))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂ξ.∂y.ey),  maximum(∂ξ.∂y.ey))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂x.c),  maximum(∂η.∂x.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂x.v),  maximum(∂η.∂x.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂x.ex),  maximum(∂η.∂x.ex))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂x.ey),  maximum(∂η.∂x.ey))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂y.c),  maximum(∂η.∂y.c))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂y.v),  maximum(∂η.∂y.v))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂y.ex),  maximum(∂η.∂y.ex))
    # @printf("%2.2e --- %2.2e\n",  minimum(∂η.∂y.ey),  maximum(∂η.∂y.ey))

    # Visualise
    # vertx = [  xv2_1[1:end-1,1:end-1][:]  xv2_1[2:end-0,1:end-1][:]  xv2_1[2:end-0,2:end-0][:]  xv2_1[1:end-1,2:end-0][:] ] 
    # verty = [  yv2_1[1:end-1,1:end-1][:]  yv2_1[2:end-0,1:end-1][:]  yv2_1[2:end-0,2:end-0][:]  yv2_1[1:end-1,2:end-0][:] ] 
    # sol   = ( vx=R.x.c[2:end-1,2:end-1][:], vy=V.y.c[2:end-1,2:end-1][:], p= avWESN(P.ex[2:end-1,2:end-1], P.ey[2:end-1,2:end-1])[:])
    # PatchPlotMakieBasic(vertx, verty, sol, x.min, x.max, y.min, y.max, write_fig=false)
 
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
