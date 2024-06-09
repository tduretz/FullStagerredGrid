# TRY EXPONENTIAL MESH IN BOTH X AND Y
# MAKE IT COMPRESSIBLE
# RELABELED: v, c, i, j 
# ADD OLD PRESSURES/STRESS/VELOCITY
# Initialisation
using FullStaggeredGrid
using Plots, Printf, LinearAlgebra, SpecialFunctions
import CairoMakie
using Makie.GeometryBasics, ForwardDiff, MAT
# Macros
@views    ∂_∂x(f1,f2,Δξ,Δη,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δξ .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δη
@views    ∂_∂y(f1,f2,Δξ,Δη,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δξ .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δη
@views    ∂η∂xv(A)       = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views ∂η∂xv_xa(A)       =  0.5*(A[1:end-1,:].+A[2:end,:])
@views ∂η∂xv_ya(A)       =  0.5*(A[:,1:end-1].+A[:,2:end]) 
@views   ∂η∂xvh(A)       = ( 0.25./A[1:end-1,1:end-1] .+ 0.25./A[1:end-1,2:end-0] .+ 0.25./A[2:end-0,1:end-1] .+ 0.25./A[2:end-0,2:end-0]).^(-1)
@views   ∂η∂xvWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
Dat = Float64  # Precision (double=Float64 or single=Float32)
function PatchPlotMakie(vertx, verty, sol, xmin, xmax, ymin, ymax, x1, y1, x2, y2; cmap = :turbo, write_fig=false )
    f   = CairoMakie.Figure(resolution = (1200, 1000))

    ar = (xmax - xmin) / (ymax - ymin)

    CairoMakie.Axis(f[1,1]) #, aspect = ar
    min_v = minimum( sol.p ); max_v = maximum( sol.p )

    # PRESSURE
    # min_v = .0; max_v = 5.
    # # min_v = minimum( sol.p ); max_v = maximum( sol.p )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    # CairoMakie.poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # CairoMakie.Axis(f[2,1], aspect = ar)
    # min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # CairoMakie.poly!(p, color = sol.vx, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # CairoMakie.Axis(f[2,2], aspect = ar)
    # min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # CairoMakie.poly!(p, color = sol.vy, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
    min_v = minimum( sol.τII ); max_v = maximum( sol.τII )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    CairoMakie.poly!(p, color = sol.τII, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # CairoMakie.scatter!(x1,y1, color=:white)
    # CairoMakie.scatter!(x2,y2, color=:white, marker=:xcross)
    CairoMakie.Colorbar(f[1, 2], colormap = cmap, limits=limits, flipaxis = true, size = 25 )

    display(f)
    if write_fig==true 
        FileIO.s∂η∂xve( string(@__DIR__, "/plot.png"), f)
    end
    return nothing
end
@views h(x,A,σ,b,x0)    = A*exp(-(x-x0)^2/σ^2) + b
@views dhdx(x,A,σ,b,x0) = -2*x/σ^2*A*exp(-(x-x0).^2/σ^2)
@views y_coord(y,ymin,z0,m)   = (y/ymin)*((z0+m))-z0
function Mesh_y( X, A, x0, σ, b, m, ymin0, ymax0, σy, swiss )
    if swiss
        y0    = ymax0 
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    else
        y     = X[2]
    end
    z0    = -(A*exp(-(X[1]-x0)^2/σ^2) + b) # topography height
    y     = (y/ymin0)*((z0+m))-z0        # shift grid vertically
    return y
end
function Mesh_x( X, A, x0, σ, b, m, xmin0, xmax0, σx, swiss )
    if swiss
        xmin1 = (sinh.( σx.*(xmin0.-x0) ))
        xmax1 = (sinh.( σx.*(xmax0.-x0) ))
        sx    = (xmax0-xmin0)/(xmax1-xmin1)
        x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0  
    else
        x   = X[1]  
    end    
    return x
end

# 2D Poisson routine
@views function Wave2D_FSG()

    dynamic    = false
    inclusion  = false
    adapt_mesh = true
    swiss      = false
    solve      = true

    # Physics
    xmin, xmax = -3.0, 3.0  
    ymin, ymax = -5.0, 0.0
    εbg      = -1*0
    rad      = 0.5
    y0       = -2.
    g        = -1.
    K0       = 1e1
    G0       = 1e0
    ρ0       = 1.0
    η0       = 1e-2
    # Numerics
    ncx, ncy = 41, 41    # numerical grid resolution
    Δξ, Δη  = (xmax-xmin)/ncx, (ymax-ymin)/ncy
    if dynamic 
        c_eff    = sqrt((K0+4/3*G0)/ρ0) 
        Δt       = min(1e10, 0.1*Δξ/c_eff, 0.1*Δη/c_eff) /2
        iterMax  = 1       # max number of iters
        nout     = 1       # residual check frequency
        nout_viz = 10
        nt       = 200
    else
        Δt       = 1e0
        iterMax  = 3e4     # max number of iters
        nout     = 1000    # residual check frequency
        nout_viz = 1
        nt       = 10
    end
    ε        = 1e-6      # nonlinear tolerance
    # Iterative parameters -------------------------------------------
    if adapt_mesh
        Reopt    = 0.5*π*1.75
        cfl      = 1.1/2.0
        nsm      = 4
    else
        Reopt    = 0.625*π*2
        cfl      = 1.1
        nsm      = 4
    end
    ρ        = cfl*Reopt/ncx
    tol      = 1e-8
    # Array initialisation
    Vx_v        =   zeros(Dat, ncx+1, ncy+1)
    Vy_v        =   zeros(Dat, ncx+1, ncy+1)
    Vx_c        =   zeros(Dat, ncx+2, ncy+2) # extended
    Vy_c        =   zeros(Dat, ncx+2, ncy+2) # extended
    ε̇xx_j       =   zeros(Dat, ncx-0, ncy+1)
    ε̇yy_j       =   zeros(Dat, ncx-0, ncy+1)
    ε̇xy_j       =   zeros(Dat, ncx-0, ncy+1)
    ∇v_j        =   zeros(Dat, ncx-0, ncy+1)
    ε̇xx_i       =   zeros(Dat, ncx+1, ncy-0+1)
    ε̇yy_i       =   zeros(Dat, ncx+1, ncy-0+1) #FS
    ε̇xy_i       =   zeros(Dat, ncx+1, ncy-0+1)
    ∇v_i        =   zeros(Dat, ncx+1, ncy-0+1)
    τxx_j       =   zeros(Dat, ncx-0, ncy+1)
    τyy_j       =   zeros(Dat, ncx-0, ncy+1)
    τxy_j       =   zeros(Dat, ncx-0, ncy+1)
    P_j         =   zeros(Dat, ncx-0, ncy+1)
    τxx0_j      =   zeros(Dat, ncx-0, ncy+1)
    τyy0_j      =   zeros(Dat, ncx-0, ncy+1)
    τxy0_j      =   zeros(Dat, ncx-0, ncy+1)
    P0_j        =   zeros(Dat, ncx-0, ncy+1)
    ηs_j        =   zeros(Dat, ncx-0, ncy+1)
    ηb_j        =   zeros(Dat, ncx-0, ncy+1)
    τxx_i       =   zeros(Dat, ncx+1, ncy-0+1)
    τyy_i       =   zeros(Dat, ncx+1, ncy-0+1)
    τxy_i       =   zeros(Dat, ncx+1, ncy-0+1)
    P_i         =   zeros(Dat, ncx+1, ncy-0+1)
    τxx0_i      =   zeros(Dat, ncx+1, ncy-0+1)
    τyy0_i      =   zeros(Dat, ncx+1, ncy-0+1)
    τxy0_i      =   zeros(Dat, ncx+1, ncy-0+1)
    P0_i        =   zeros(Dat, ncx+1, ncy-0+1)
    ηs_i        =   zeros(Dat, ncx+1, ncy-0+1)
    ηb_i        =   zeros(Dat, ncx+1, ncy-0+1)
    Rx_v        =   zeros(Dat, ncx+1, ncy+1)
    Ry_v        =   zeros(Dat, ncx+1, ncy+1)
    ρ_v         =   zeros(Dat, ncx+1, ncy+1)
    Rx_c        =   zeros(Dat, ncx+0, ncy+0) # not extended
    Ry_c        =   zeros(Dat, ncx+0, ncy+0)
    ρ_c         =   zeros(Dat, ncx+0, ncy-0)
    RP_j        =   zeros(Dat, ncx-0, ncy+1)
    RP_i        =   zeros(Dat, ncx+1, ncy-0)
    Δτv_v       =   zeros(Dat, ncx-1, ncy-0)
    Δτv_c       =   zeros(Dat, ncx-0, ncy-0)
    κΔτP_j      =   zeros(Dat, ncx-0, ncy+1)
    κΔτP_i      =   zeros(Dat, ncx+1, ncy-0)
    dVxdτ_v     =   zeros(Dat, ncx+1, ncy+1)
    dVydτ_v     =   zeros(Dat, ncx+1, ncy+1)
    dVxdτ_c     =   zeros(Dat, ncx+0, ncy+0)
    dVydτ_c     =   zeros(Dat, ncx+0, ncy+0)
    τII_j       =   zeros(Dat, ncx-0, ncy+1)        
    τII_i       =   zeros(Dat, ncx+1, ncy-0+1) # FS
    # Initialisation
    xxv, yyv    = LinRange(xmin-Δξ/2, xmax+Δξ/2, 2ncx+3), LinRange(ymin-Δη/2, ymax+Δη/2, 2ncy+3)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    ∂ξ∂x =  ones(2ncx+3, 2ncy+3)
    ∂ξ∂y = zeros(2ncx+3, 2ncy+3)
    ∂η∂x = zeros(2ncx+3, 2ncy+3)
    ∂η∂y =  ones(2ncx+3, 2ncy+3)
    hx   = zeros(2ncx+3, 2ncy+3)
    if adapt_mesh
        x0     = (xmax + xmin)/2
        m      = ymin
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
        hx     = -dhdx.(x_ini, Amp, σ, ymax, x0)
        # Deform mesh
        for i in eachindex(x_ini)          
            X_msh[1] = x_ini[i]
            X_msh[2] = y_ini[i]     
            xv4[i]   =  Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx, swiss )
            yv4[i]   =  Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, swiss )
        end
        # Compute forward transformation
        params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, σx=σx, σy=σy, ϵ=ϵ, swiss=swiss)
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation!( Mesh_x, Mesh_y, ∂x, ∂y, x_ini, y_ini, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵ, swiss)
        # Solve for inverse transformation
        ∂ξ = (∂x=∂ξ∂x, ∂y=∂ξ∂y); ∂η = (∂x=∂η∂x, ∂y=∂η∂y)
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
        ∂ξ∂x .= ∂ξ.∂x; ∂ξ∂y .= ∂ξ.∂y
        ∂η∂x .= ∂η.∂x; ∂η∂y .= ∂η.∂y
    end
    # Grid subsets
    xv2_v, yv2_v = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
    xv2_c, yv2_c = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xc2_v, yc2_v = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    xc2_c, yc2_c = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    ∂ξ∂xc_v = ∂ξ∂x[3:2:end-2,2:2:end-1]; ∂ξ∂xv_v = ∂ξ∂x[2:2:end-1,2:2:end-1]
    ∂ξ∂xc_c = ∂ξ∂x[2:2:end-1,3:2:end-2]; ∂ξ∂xv_c = ∂ξ∂x[1:2:end-0,1:2:end-0]
    ∂ξ∂yc_v = ∂ξ∂y[3:2:end-2,2:2:end-1]; ∂ξ∂yv_v = ∂ξ∂y[2:2:end-1,2:2:end-1]
    ∂ξ∂yc_c = ∂ξ∂y[2:2:end-1,3:2:end-2]; ∂ξ∂yv_c = ∂ξ∂y[1:2:end-0,1:2:end-0]
    ∂η∂xc_v = ∂η∂x[3:2:end-2,2:2:end-1]; ∂η∂xv_v = ∂η∂x[2:2:end-1,2:2:end-1]
    ∂η∂xc_c = ∂η∂x[2:2:end-1,3:2:end-2]; ∂η∂xv_c = ∂η∂x[1:2:end-0,1:2:end-0]
    ∂η∂yc_v = ∂η∂y[3:2:end-2,2:2:end-1]; ∂η∂yv_v = ∂η∂y[2:2:end-1,2:2:end-1]
    ∂η∂yc_c = ∂η∂y[2:2:end-1,3:2:end-2]; ∂η∂yv_c = ∂η∂y[1:2:end-0,1:2:end-0]
    # Velocity
    Vx_v .= -εbg.*xv2_v; Vx_c .= -εbg.*xv2_c
    Vy_v .=  εbg.*yv2_v; Vy_c .=  εbg.*yv2_c
    # Viscosity
    ηv  = η0
    ηe  = G0*Δt
    ηve = 1/(1/ηv + 1/ηe)
    ν   = (3*K0 - 2*G0)/2/(3*K0 + G0)
    @show ηv, ηe, ηve, ν
    ηs_j  .= ηve;    ηs_i .= ηve
    ηb_j  .= K0*Δt;  ηb_i .= K0*Δt
    if inclusion
        ηs_j[xc2_v.^2 .+ (yc2_v.-y0).^2 .< rad] .= 100.
        ηs_i[xc2_c.^2 .+ (yc2_c.-y0).^2 .< rad] .= 100.
    end
    # Density
    ρ_v  .= ρ0; ρ_c  .= ρ0
    if inclusion
        ρ_v[xv2_v.^2 .+ (yv2_v.-y0).^2 .< rad] .= 1.
        ρ_c[xv2_c[2:end-1,2:end-1].^2 .+ (yv2_c[2:end-1,2:end-1].-y0).^2 .< rad] .= 1.
    end
    if dynamic
        P_j[xc2_v.^2 .+ (yc2_v.-y0).^2 .< rad] .= 5.
        P_i[xc2_c.^2 .+ (yc2_c.-y0).^2 .< rad] .= 5.
    end
    # Smooth Viscosity
    ηs_j_sm    = zeros(size(ηs_j))
    ηs_i_sm    = zeros(size(ηs_i,1), size(ηs_i,2)-1)
    ηs_j_sm   .= ηs_j
    ηs_i_sm   .= ηs_i[:,1:end-1]
    ηs_j_sm_c = ∂η∂xv(ηs_j_sm)
    ηs_i_sm_c = ∂η∂xv(ηs_i_sm)
    for it = 1:nsm
        ηs_j_sm[2:end-1,2:end-1] .= ∂η∂xv(ηs_j_sm_c)
        ηs_j_sm_c = ∂η∂xv(ηs_j_sm)
        ηs_i_sm[2:end-1,2:end-1] .= ∂η∂xv(ηs_i_sm_c)
        ηs_i_sm_c = ∂η∂xv(ηs_i_sm)
    end
    # PT steps
    @show Δξ1 = minimum(xv4[2:end,:].-xv4[1:end-1,:])
    @show Δη1 = minimum(yv4[:,2:end].-yv4[:,1:end-1])
    @show Δξ
    @show Δη
    Δτv_v  .= cfl .*ρ .*min(Δξ1,Δη1)^2 ./ (0.50*(ηs_j_sm[1:end-1,2:end-0].+ηs_j_sm[2:end-0,2:end-0])) ./ 4.1 
    Δτv_c  .= cfl .*ρ .*min(Δξ1,Δη1)^2 ./ (0.25*(ηs_j_sm[:,1:end-1].+ηs_j_sm[:,2:end-0].+ηs_i_sm[1:end-1,:].+ηs_i_sm[2:end-0,:])) ./ 4.1 
    κΔτP_j .= cfl .* ηs_j            .* Δξ ./ (xmax-xmin)
    κΔτP_i .= cfl .* ηs_i[:,1:end-1] .* Δξ ./ (xmax-xmin)
    if solve
        hx_surf =   hx[3:2:end-2, end-1]
        η_surf  = ηs_j[:,end]
        dx      = Δξ
        dz      = Δη
        dkdx    = ∂ξ∂x[3:2:end-2, end-1]
        dkdy    = ∂ξ∂y[3:2:end-2, end-1]
        dedx    = ∂η∂x[3:2:end-2, end-1]
        dedy    = ∂η∂y[3:2:end-2, end-1]
        h_x   = hx[3:2:end-2, end]
        eta   = η_surf
        eta_e = ηe
        # duNddudx = dz .* (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        # duNddvdx = dz .* (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        # duNdP    = (3 // 2) .* dz .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2))
        # dvNddudx = dz .* (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        # dvNddvdx = dz .* (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        # dvNdP    = (3 // 2) .* dz .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2))
        duNddudx = similar(η_surf); duNddvdx = similar(η_surf);  duNdP = similar(η_surf); duNdTxx0 = similar(η_surf); duNdTyy0 = similar(η_surf); duNdTxy0 = similar(η_surf);
        @. duNddudx = dz .* (-2 * dedx .* dkdx .* h_x .^ 2 - dedx .* dkdx - 2 * dedy .* dkdx .* h_x - dedy .* dkdy .* h_x .^ 2 - 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2)
        @. duNddvdx = dz .* (dedx .* dkdy .* h_x .^ 2 + 2 * dedx .* dkdy - dedy .* dkdx .* h_x .^ 2 - 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2)
        @. duNdP    = (3 // 2) * dz .* (dedx .* h_x .^ 2 - dedx + 2 * dedy .* h_x) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. duNdTxx0 = dz .* h_x .* (-3 * dedx .* h_x - 4 * dedy) ./ (2 * eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. duNdTyy0 = dz .* (3 * dedx - 2 * dedy .* h_x) ./ (2 * eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. duNdTxy0 = dedy .* dz .* (-h_x .^ 2 - 2) ./ (eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        #--------------------
        dvNddudx = similar(η_surf); dvNddvdx = similar(η_surf);  dvNdP = similar(η_surf); dvNdTxx0 = similar(η_surf); dvNdTyy0 = similar(η_surf); dvNdTxy0 = similar(η_surf);
        @. dvNddudx = dz .* (-2 * dedx .* dkdy .* h_x .^ 2 - dedx .* dkdy + 2 * dedy .* dkdx .* h_x .^ 2 + dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2)
        @. dvNddvdx = dz .* (-2 * dedx .* dkdx .* h_x .^ 2 - dedx .* dkdx - 2 * dedx .* dkdy .* h_x - dedy .* dkdy .* h_x .^ 2 - 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2)
        @. dvNdP    = (3 // 2) * dz .* (2 * dedx .* h_x - dedy .* h_x .^ 2 + dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. dvNdTxx0 = dz .* h_x .* (-2 * dedx + 3 * dedy .* h_x) ./ (2 * eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. dvNdTyy0 = dz .* (-4 * dedx .* h_x - 3 * dedy) ./ (2 * eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        @. dvNdTxy0 = dedx .* dz .* (-2 * h_x .^ 2 - 1) ./ (eta_e .* (2 * dedx .^ 2 .* h_x .^ 2 + dedx .^ 2 + 2 * dedx .* dedy .* h_x + dedy .^ 2 .* h_x .^ 2 + 2 * dedy .^ 2))
        # Time loop
        for it=1:nt
            @printf("it = %03d\n", it)
            # Old guys
            τxx0_j .= τxx_j
            τxx0_i .= τxx_i
            τyy0_j .= τyy_j
            τyy0_i .= τyy_i
            τxy0_j .= τxy_j
            τxy0_i .= τxy_i
            P0_j   .= P_j
            P0_i   .= P_i
            # PT loop
            iter=1; err=2*ε; err_evo1=[]; err_evo2=[];
            while (err>ε && iter<=iterMax)
                #
                # Vx_v[[1 end],:] .= Vx_v[[2 end-1],:]
                # Vx_c[[1 end],:] .= Vx_c[[2 end-1],:]
                # Vx_c[:,    [1]] .= Vx_c[:,      [2]]
                # Vx_v[:,    [1]] .= Vx_v[:,      [2]]
                # Surface values
                dVxdx  = (Vx_v[2:end-0,end] - Vx_v[1:end-1,end])/Δξ   
                dVydx  = (Vy_v[2:end-0,end] - Vy_v[1:end-1,end])/Δξ
                P_surf = P_j[:,end]
                if dynamic
                    # See python notebook v5
                    Vx_c[2:end-1,end] .= Vx_c[2:end-1,end-1] .+ duNddudx.*dVxdx .+ duNddvdx.*dVydx .+ duNdP.*P_surf .+ duNdTxx0.*τxx0_j[:,end] .+ duNdTyy0.*τyy0_j[:,end]  .+ duNdTxy0.*τxy0_j[:,end]
                    Vy_c[2:end-1,end] .= Vy_c[2:end-1,end-1] .+ dvNddudx.*dVxdx .+ dvNddvdx.*dVydx .+ dvNdP.*P_surf .+ dvNdTxx0.*τxx0_j[:,end] .+ dvNdTyy0.*τyy0_j[:,end]  .+ dvNdTxy0.*τxy0_j[:,end]
                    ∇v_j              .= ∂_∂x(Vx_v,Vx_c[2:end-1,:],Δξ,Δη,∂ξ∂xc_v,∂η∂xc_v) .+ ∂_∂y(Vy_c[2:end-1,:],Vy_v,Δξ,Δη,∂ξ∂yc_v,∂η∂yc_v) 
                    ∇v_i[:,1:end-1]   .= ∂_∂x(Vx_c[:,2:end-1],Vx_v,Δξ,Δη,∂ξ∂xc_c,∂η∂xc_c) .+ ∂_∂y(Vy_v,Vy_c[:,2:end-1],Δξ,Δη,∂ξ∂yc_c,∂η∂yc_c) 
                    P_j                    .= P0_j            .- ηb_j           .*∇v_j           
                    P_i[:,1:end-1]         .= P0_i[:,1:end-1] .- ηb_i[:,1:end-1].*∇v_i[:,1:end-1]
                end
                P_surf = P_j[:,end]
                # See python notebook v5
                Vx_c[2:end-1,end] .= Vx_c[2:end-1,end-1] .+ duNddudx.*dVxdx .+ duNddvdx.*dVydx .+ duNdP.*P_surf .+ duNdTxx0.*τxx0_j[:,end] .+ duNdTyy0.*τyy0_j[:,end]  .+ duNdTxy0.*τxy0_j[:,end]
                Vy_c[2:end-1,end] .= Vy_c[2:end-1,end-1] .+ dvNddudx.*dVxdx .+ dvNddvdx.*dVydx .+ dvNdP.*P_surf .+ dvNdTxx0.*τxx0_j[:,end] .+ dvNdTyy0.*τyy0_j[:,end]  .+ dvNdTxy0.*τxy0_j[:,end]
                ∇v_j              .= ∂_∂x(Vx_v,Vx_c[2:end-1,:],Δξ,Δη,∂ξ∂xc_v,∂η∂xc_v) .+ ∂_∂y(Vy_c[2:end-1,:],Vy_v,Δξ,Δη,∂ξ∂yc_v,∂η∂yc_v) 
                ∇v_i[:,1:end-1]   .= ∂_∂x(Vx_c[:,2:end-1],Vx_v,Δξ,Δη,∂ξ∂xc_c,∂η∂xc_c) .+ ∂_∂y(Vy_v,Vy_c[:,2:end-1],Δξ,Δη,∂ξ∂yc_c,∂η∂yc_c) 
                ε̇xx_j             .= ∂_∂x(Vx_v,Vx_c[2:end-1,:],Δξ,Δη,∂ξ∂xc_v,∂η∂xc_v) .- 1.0/3.0*∇v_j
                ε̇yy_j             .= ∂_∂y(Vy_c[2:end-1,:],Vy_v,Δξ,Δη,∂ξ∂yc_v,∂η∂yc_v) .- 1.0/3.0*∇v_j
                ε̇xy_j             .=(∂_∂y(Vx_c[2:end-1,:],Vx_v,Δξ,Δη,∂ξ∂yc_v,∂η∂yc_v) .+ ∂_∂x(Vy_v,Vy_c[2:end-1,:], Δξ,Δη,∂ξ∂xc_v,∂η∂xc_v) ) / 2.
                ε̇xx_i[:,1:end-1]  .= ∂_∂x(Vx_c[:,2:end-1],Vx_v,Δξ,Δη,∂ξ∂xc_c,∂η∂xc_c) .- 1.0/3.0*∇v_i[:,1:end-1]
                ε̇yy_i[:,1:end-1]  .= ∂_∂y(Vy_v,Vy_c[:,2:end-1],Δξ,Δη,∂ξ∂yc_c,∂η∂yc_c) .- 1.0/3.0*∇v_i[:,1:end-1]
                ε̇xy_i[:,1:end-1]  .=(∂_∂y(Vx_v,Vx_c[:,2:end-1],Δξ,Δη,∂ξ∂yc_c,∂η∂yc_c) .+ ∂_∂x(Vy_c[:,2:end-1],Vy_v, Δξ,Δη,∂ξ∂xc_c,∂η∂xc_c) ) / 2.
                # @show ε̇xx_i[:, end]
                τxx_j .= 2.0 .* ηs_j .* (ε̇xx_j + τxx0_j./(2.0.*ηe))
                τxx_i .= 2.0 .* ηs_i .* (ε̇xx_i + τxx0_i./(2.0.*ηe))
                τyy_j .= 2.0 .* ηs_j .* (ε̇yy_j + τyy0_j./(2.0.*ηe))
                τyy_i .= 2.0 .* ηs_i .* (ε̇yy_i + τyy0_i./(2.0.*ηe))
                τxy_j .= 2.0 .* ηs_j .* (ε̇xy_j + τxy0_j./(2.0.*ηe))
                τxy_i .= 2.0 .* ηs_i .* (ε̇xy_i + τxy0_i./(2.0.*ηe))
                # if dynamic 
                #     P_j                    .= P0_j            .- ηb_j           .*∇v_j           
                #     P_i[:,1:end-1]         .= P0_i[:,1:end-1] .- ηb_i[:,1:end-1].*∇v_i[:,1:end-1]
                # end
                Rx_v[2:end-1,2:end-0] .= ∂_∂x(τxx_j[:,2:end-0],τxx_i[2:end-1,:],Δξ,Δη,∂ξ∂xv_v[2:end-1,2:end-0],∂η∂xv_v[2:end-1,2:end-0]) .+ ∂_∂y(τxy_i[2:end-1,:],τxy_j[:,2:end-0],Δξ,Δη,∂ξ∂yv_v[2:end-1,2:end-0],∂η∂yv_v[2:end-1,2:end-0]) .-  ∂_∂x(P_j[:,2:end-0],P_i[2:end-1,:],Δξ,Δη,∂ξ∂xv_v[2:end-1,2:end-0],∂η∂xv_v[2:end-1,2:end-0])
                Rx_c                  .= ∂_∂x(τxx_i[:,1:end-1],τxx_j,           Δξ,Δη,∂ξ∂xv_c[2:end-1,2:end-1],∂η∂xv_c[2:end-1,2:end-1]) .+ ∂_∂y(τxy_j,τxy_i[:,1:end-1],           Δξ,Δη,∂ξ∂yv_c[2:end-1,2:end-1],∂η∂yv_c[2:end-1,2:end-1]) .-  ∂_∂x(P_i[:,1:end-1],P_j,           Δξ,Δη,∂ξ∂xv_c[2:end-1,2:end-1],∂η∂xv_c[2:end-1,2:end-1]) 
                Ry_v[2:end-1,2:end-0] .= ∂_∂y(τyy_i[2:end-1,:],τyy_j[:,2:end-0],Δξ,Δη,∂ξ∂yv_v[2:end-1,2:end-0],∂η∂yv_v[2:end-1,2:end-0]) .+ ∂_∂x(τxy_j[:,2:end-0],τxy_i[2:end-1,:],Δξ,Δη,∂ξ∂xv_v[2:end-1,2:end-0],∂η∂xv_v[2:end-1,2:end-0]) .-  ∂_∂y(P_i[2:end-1,:],P_j[:,2:end-0],Δξ,Δη,∂ξ∂yv_v[2:end-1,2:end-0],∂η∂yv_v[2:end-1,2:end-0]) .+ ρ_v[2:end-1,2:end-0].*g
                Ry_c                  .= ∂_∂y(τyy_j,τyy_i[:,1:end-1],           Δξ,Δη,∂ξ∂yv_c[2:end-1,2:end-1],∂η∂yv_c[2:end-1,2:end-1]) .+ ∂_∂x(τxy_i[:,1:end-1],τxy_j,           Δξ,Δη,∂ξ∂xv_c[2:end-1,2:end-1],∂η∂xv_c[2:end-1,2:end-1]) .-  ∂_∂y(P_j,P_i[:,1:end-1],           Δξ,Δη,∂ξ∂yv_c[2:end-1,2:end-1],∂η∂yv_c[2:end-1,2:end-1]) .+ ρ_c.*g
                RP_j                  .= .-∇v_j            .- (P_j            .- P0_j)./ηb_j
                RP_i                  .= .-∇v_i[:,1:end-1] .- (P_i[:,1:end-1] .- P0_i[:,1:end-1])./ηb_i[:,1:end-1]
                if dynamic==false
                    # Calculate rate update --------------------------------------
                    dVxdτ_v .= (1-ρ) .* dVxdτ_v .+ Rx_v
                    dVydτ_v .= (1-ρ) .* dVydτ_v .+ Ry_v
                    dVxdτ_c .= (1-ρ) .* dVxdτ_c .+ Rx_c
                    dVydτ_c .= (1-ρ) .* dVydτ_c .+ Ry_c
                    # Update velocity and pressure -------------------------------
                    Vx_v[2:end-1,2:end-0] .+= Δτv_v ./ ρ .* dVxdτ_v[2:end-1,2:end-0]
                    Vy_v[2:end-1,2:end-0] .+= Δτv_v ./ ρ .* dVydτ_v[2:end-1,2:end-0]
                    Vx_c[2:end-1,2:end-1] .+= Δτv_c ./ ρ .* dVxdτ_c
                    Vy_c[2:end-1,2:end-1] .+= Δτv_c ./ ρ .* dVydτ_c
                    P_j                   .+= κΔτP_j .* RP_j
                    P_i[:,1:end-1]        .+= κΔτP_i .* RP_i
                    # Convergence check
                    if mod(iter, nout)==0 || iter==1
                        norm_Rx = 0.5*( norm(Rx_v)/sqrt(length(Rx_v)) + norm(Rx_c)/sqrt(length(Rx_c)) )
                        norm_Ry = 0.5*( norm(Ry_v)/sqrt(length(Ry_v)) + norm(Ry_c)/sqrt(length(Ry_c)) )
                        norm_Rp = 0.5*( norm(RP_j)/sqrt(length(RP_j)) + norm(RP_i)/sqrt(length(RP_i)) )
                        @printf("it = %03d, iter = %05d, nRx=%1.3e nRy=%1.3e nRp=%1.3e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
                        if (isnan(norm_Rx) || isnan(norm_Ry) || isnan(norm_Rp)) error("NaN"); end
                        if (norm_Rx<tol && norm_Ry<tol && norm_Rp<tol) break; end
                    end
                else
                    Vx_v[2:end-1,2:end-0] .+= Δt ./ ρ .* Rx_v[2:end-1,2:end-0]
                    Vy_v[2:end-1,2:end-0] .+= Δt ./ ρ .* Ry_v[2:end-1,2:end-0]
                    Vx_c[2:end-1,2:end-1] .+= Δt ./ ρ .* Rx_c
                    Vy_c[2:end-1,2:end-1] .+= Δt ./ ρ .* Ry_c
                end
                iter+=1; global itg=iter
            end
            if mod(it, nout_viz)==0 || it==1
                # Plotting
                # Generate data
                @. τII_j = sqrt(0.5*(τxx_j^2 + τyy_j^2 + (τxx_j+τyy_j)^2) + τxy_j^2)
                @. τII_i = sqrt(0.5*(τxx_i^2 + τyy_i^2 + (τxx_i+τyy_i)^2) + τxy_i^2)
                vertx = [  xv2_v[1:end-1,1:end-1][:]  xv2_v[2:end-0,1:end-1][:]  xv2_v[2:end-0,2:end-0][:]  xv2_v[1:end-1,2:end-0][:] ] 
                verty = [  yv2_v[1:end-1,1:end-1][:]  yv2_v[2:end-0,1:end-1][:]  yv2_v[2:end-0,2:end-0][:]  yv2_v[1:end-1,2:end-0][:] ] 
                sol   = ( vx=Vx_c[2:end-1,2:end-1][:], vy=Vy_c[2:end-1,2:end-1][:], p=∂η∂xvWESN(P_j, P_i[:,1:end-1])[:], η=∂η∂xvWESN(ηs_j, ηs_i[:,1:end-1])[:], τII=∂η∂xvWESN(τII_j, τII_i[:,1:end-1])[:])
                xc2   = ∂η∂xvWESN(xc2_v, xc2_c[:,1:end-1])[:] 
                yc2   = ∂η∂xvWESN(yc2_v, yc2_c[:,1:end-1])[:]
                PatchPlotMakie(vertx, verty, sol, minimum(xv2_v), maximum(xv2_v), minimum(yv2_v), maximum(yv2_v), xv2_v[:], yv2_v[:], xc2[:], yc2[:], write_fig=false)
                
                # file = matopen(string(@__DIR__,"/output_FS_topo.mat"), "w")
                # write(file, "Vx_v", Vx_v)
                # write(file, "Vx_c", Vx_c)
                # write(file, "Vy_v", Vy_v)
                # write(file, "Vy_c", Vy_c)
                # write(file, "P_j", P_j)
                # write(file, "P_i", P_i)
                # write(file, "duNddudx",duNddudx)
                # write(file, "duNddvdx",duNddvdx)
                # write(file, "duNdP"   ,duNdP   )   
                # write(file, "dvNddudx",dvNddudx)
                # write(file, "dvNddvdx",dvNddvdx)
                # write(file, "dvNdP"   ,dvNdP   )    
                # write(file, "dkdx", Array(dkdx))
                # write(file, "dkdy", Array(dkdy))
                # write(file, "dedx", Array(dedx))
                # write(file, "dedy", Array(dedy))
                # write(file, "hx",   Array(h_x) )
                # close(file)
            end
            @show norm(P_i .- P0_i)./length(P0_i)
        end
    end

    return
end

Wave2D_FSG()