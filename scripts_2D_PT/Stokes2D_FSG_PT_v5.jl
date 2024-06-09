# TRY EXPONENTIAL MESH IN BOTH X AND Y
# Initialisation
using FullStaggeredGrid
using Plots, Printf, LinearAlgebra, SpecialFunctions
import CairoMakie
using Makie.GeometryBasics, ForwardDiff, MAT
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
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

    # min_v = .0; max_v = 5.

    # min_v = minimum( sol.p ); max_v = maximum( sol.p )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    # CairoMakie.poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # CairoMakie.Axis(f[2,1], aspect = ar)
    # min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # CairoMakie.poly!(p, color = sol.vx, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # CairoMakie.Axis(f[2,2], aspect = ar)
    min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    CairoMakie.poly!(p, color = sol.vy, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
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
@views function Stokes2S_FSG()
    # Physics
    xmin, xmax = -3.0, 3.0  
    ymin, ymax = -5.0, 0.0
    εbg      = -1*0
    rad      = 0.5
    y0       = -2.
    g        = -1
    inclusion  = false
    adapt_mesh = true
    swiss      = false
    solve      = true
    # Numerics
    ncx, ncy = 61, 61    # numerical grid resolution
    ε        = 1e-6      # nonlinear tolerance
    iterMax  = 3e4       # max number of iters
    nout     = 1000      # residual check frequency
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
    # Preprocessing
    Δx, Δy  = (xmax-xmin)/ncx, (ymax-ymin)/ncy
    # Array initialisation
    Vx_1        =   zeros(Dat, ncx+1, ncy+1)
    Vy_1        =   zeros(Dat, ncx+1, ncy+1)
    Vx_2        =   zeros(Dat, ncx+2, ncy+2) # extended
    Vy_2        =   zeros(Dat, ncx+2, ncy+2) # extended
    ε̇xx_1       =   zeros(Dat, ncx-0, ncy+1)
    ε̇yy_1       =   zeros(Dat, ncx-0, ncy+1)
    ε̇xy_1       =   zeros(Dat, ncx-0, ncy+1)
    ∇v_1        =   zeros(Dat, ncx-0, ncy+1)
    ε̇xx_2       =   zeros(Dat, ncx+1, ncy-0+1)
    ε̇yy_2       =   zeros(Dat, ncx+1, ncy-0+1) #FS
    ε̇xy_2       =   zeros(Dat, ncx+1, ncy-0+1)
    ∇v_2        =   zeros(Dat, ncx+1, ncy-0+1)
    τxx_1       =   zeros(Dat, ncx-0, ncy+1)
    τyy_1       =   zeros(Dat, ncx-0, ncy+1)
    τxy_1       =   zeros(Dat, ncx-0, ncy+1)
    P_1         =   zeros(Dat, ncx-0, ncy+1)
    η_1         =   zeros(Dat, ncx-0, ncy+1)
    τxx_2       =   zeros(Dat, ncx+1, ncy-0+1)
    τyy_2       =   zeros(Dat, ncx+1, ncy-0+1)
    τxy_2       =   zeros(Dat, ncx+1, ncy-0+1)
    P_2         =   zeros(Dat, ncx+1, ncy-0+1)
    η_2         =   zeros(Dat, ncx+1, ncy-0+1)
    Rx_1        =   zeros(Dat, ncx+1, ncy+1)
    Ry_1        =   zeros(Dat, ncx+1, ncy+1)
    ρ_1         =   zeros(Dat, ncx+1, ncy+1)
    Rx_2        =   zeros(Dat, ncx+0, ncy+0) # not extended
    Ry_2        =   zeros(Dat, ncx+0, ncy+0)
    ρ_2         =   zeros(Dat, ncx+0, ncy-0)
    Rp_1        =   zeros(Dat, ncx-0, ncy+1)
    Rp_2        =   zeros(Dat, ncx+1, ncy-0)
    Δτv_1       =   zeros(Dat, ncx-1, ncy-0)
    Δτv_2       =   zeros(Dat, ncx-0, ncy-0)
    κΔτp_1      =   zeros(Dat, ncx-0, ncy+1)
    κΔτp_2      =   zeros(Dat, ncx+1, ncy-0)
    dVxdτ_1     =   zeros(Dat, ncx+1, ncy+1)
    dVydτ_1     =   zeros(Dat, ncx+1, ncy+1)
    dVxdτ_2     =   zeros(Dat, ncx+0, ncy+0)
    dVydτ_2     =   zeros(Dat, ncx+0, ncy+0)
    # Initialisation
    xxv, yyv    = LinRange(xmin-Δx/2, xmax+Δx/2, 2ncx+3), LinRange(ymin-Δy/2, ymax+Δy/2, 2ncy+3)
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
    xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
    xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    ∂ξ∂xc_1 = ∂ξ∂x[3:2:end-2,2:2:end-1]; ∂ξ∂xv_1 = ∂ξ∂x[2:2:end-1,2:2:end-1]
    ∂ξ∂xc_2 = ∂ξ∂x[2:2:end-1,3:2:end-2]; ∂ξ∂xv_2 = ∂ξ∂x[1:2:end-0,1:2:end-0]
    ∂ξ∂yc_1 = ∂ξ∂y[3:2:end-2,2:2:end-1]; ∂ξ∂yv_1 = ∂ξ∂y[2:2:end-1,2:2:end-1]
    ∂ξ∂yc_2 = ∂ξ∂y[2:2:end-1,3:2:end-2]; ∂ξ∂yv_2 = ∂ξ∂y[1:2:end-0,1:2:end-0]
    ∂η∂xc_1 = ∂η∂x[3:2:end-2,2:2:end-1]; ∂η∂xv_1 = ∂η∂x[2:2:end-1,2:2:end-1]
    ∂η∂xc_2 = ∂η∂x[2:2:end-1,3:2:end-2]; ∂η∂xv_2 = ∂η∂x[1:2:end-0,1:2:end-0]
    ∂η∂yc_1 = ∂η∂y[3:2:end-2,2:2:end-1]; ∂η∂yv_1 = ∂η∂y[2:2:end-1,2:2:end-1]
    ∂η∂yc_2 = ∂η∂y[2:2:end-1,3:2:end-2]; ∂η∂yv_2 = ∂η∂y[1:2:end-0,1:2:end-0]
    
    # Velocity
    Vx_1 .= -εbg.*xv2_1; Vx_2 .= -εbg.*xv2_2
    Vy_1 .=  εbg.*yv2_1; Vy_2 .=  εbg.*yv2_2
    # Viscosity
    η_1  .= 1.0; η_2  .= 1.0
    if inclusion
        η_1[xc2_1.^2 .+ (yc2_1.-y0).^2 .< rad] .= 100.
        η_2[xc2_2.^2 .+ (yc2_2.-y0).^2 .< rad] .= 100.
    end
    # Density
    ρ_1  .= 1.0; ρ_2  .= 1.0
    if inclusion
        ρ_1[xv2_1.^2 .+ (yv2_1.-y0).^2 .< rad] .= 1.
        ρ_2[xv2_2[2:end-1,2:end-1].^2 .+ (yv2_2[2:end-1,2:end-1].-y0).^2 .< rad] .= 1.
    end
    # Smooth Viscosity
    η_1_sm    = zeros(size(η_1))
    η_2_sm    = zeros(size(η_2,1), size(η_2,2)-1)
    η_1_sm   .= η_1
    η_2_sm   .= η_2[:,1:end-1]
    η_1_sm_c = ∂η∂xv(η_1_sm)
    η_2_sm_c = ∂η∂xv(η_2_sm)
    for it = 1:nsm
        η_1_sm[2:end-1,2:end-1] .= ∂η∂xv(η_1_sm_c)
        η_1_sm_c = ∂η∂xv(η_1_sm)
        η_2_sm[2:end-1,2:end-1] .= ∂η∂xv(η_2_sm_c)
        η_2_sm_c = ∂η∂xv(η_2_sm)
    end
    # PT steps
    @show Δx1 = minimum(xv4[2:end,:].-xv4[1:end-1,:])
    @show Δy1 = minimum(yv4[:,2:end].-yv4[:,1:end-1])
    @show Δx
    @show Δy
    Δτv_1  .= cfl .*ρ .*min(Δx1,Δy1)^2 ./ (0.50*(η_1_sm[1:end-1,2:end-0].+η_1_sm[2:end-0,2:end-0])) ./ 4.1 
    Δτv_2  .= cfl .*ρ .*min(Δx1,Δy1)^2 ./ (0.25*(η_1_sm[:,1:end-1].+η_1_sm[:,2:end-0].+η_2_sm[1:end-1,:].+η_2_sm[2:end-0,:])) ./ 4.1 
    κΔτp_1 .= cfl .* η_1 .* Δx ./ (xmax-xmin)
    κΔτp_2 .= cfl .* η_2[:,1:end-1] .* Δx ./ (xmax-xmin)
    if solve
        hx_surf =   hx[3:2:end-2, end-1]
        η_surf  = η_1[:,end]
        dx      = Δx
        dz      = Δy
        dkdx    = ∂ξ∂x[3:2:end-2, end-1]
        dkdy    = ∂ξ∂y[3:2:end-2, end-1]
        dedx    = ∂η∂x[3:2:end-2, end-1]
        dedy    = ∂η∂y[3:2:end-2, end-1]
        h_x = hx[3:2:end-2, end]
        eta = η_surf
        duNddudx = dz .* (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        duNddvdx = dz .* (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        duNdP    = (3 // 2) .* dz .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2))
        dvNddudx = dz .* (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        dvNddvdx = dz .* (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
        dvNdP    = (3 // 2) .* dz .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2))
        # PT loop
        it=1; iter=1; err=2*ε; err_evo1=[]; err_evo2=[];
        while (err>ε && iter<=iterMax)
            # !!!!!!!!! Change derivatives !!!!!
            dVxdx  = (Vx_1[2:end-0,end] - Vx_1[1:end-1,end])/Δx
            dVydx  = (Vy_1[2:end-0,end] - Vy_1[1:end-1,end])/Δx
            P_surf = P_1[:,end]
            # See python notebook v5
            Vx_2[2:end-1,end] = Vx_2[2:end-1,end-1] .+ duNddudx.*dVxdx .+ duNddvdx.*dVydx .+ duNdP.*P_surf
            Vy_2[2:end-1,end] = Vy_2[2:end-1,end-1] .+ dvNddudx.*dVxdx .+ dvNddvdx.*dVydx .+ dvNdP.*P_surf
            ∇v_1            .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) .+ ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) 
            ∇v_2[:,1:end-1] .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) .+ ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) 
            ε̇xx_1 .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) .- 1.0/3.0*∇v_1
            ε̇yy_1 .=  ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) .- 1.0/3.0*∇v_1
            ε̇xy_1 .= (∂_∂y(Vx_2[2:end-1,:],Vx_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) .+ ∂_∂x(Vy_1,Vy_2[2:end-1,:], Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) ) / 2.
            ε̇xx_2[:,1:end-1] .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) .- 1.0/3.0*∇v_2[:,1:end-1]
            ε̇yy_2[:,1:end-1] .=  ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) .- 1.0/3.0*∇v_2[:,1:end-1]
            ε̇xy_2[:,1:end-1] .= (∂_∂y(Vx_1,Vx_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) .+ ∂_∂x(Vy_2[:,2:end-1],Vy_1, Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) ) / 2.
            # @show ε̇xx_2[:, end]
            τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
            τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
            τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
            τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
            τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
            τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
            Rx_1[2:end-1,2:end-0] .= ∂_∂x(τxx_1[:,2:end-0],τxx_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .+ ∂_∂y(τxy_2[2:end-1,:],τxy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .-  ∂_∂x(P_1[:,2:end-0],P_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0])
            Rx_2                  .= ∂_∂x(τxx_2[:,1:end-1],τxx_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .+ ∂_∂y(τxy_1,τxy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .-  ∂_∂x(P_2[:,1:end-1],P_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) 
            Ry_1[2:end-1,2:end-0] .= ∂_∂y(τyy_2[2:end-1,:],τyy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ∂_∂x(τxy_1[:,2:end-0],τxy_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .-  ∂_∂y(P_2[2:end-1,:],P_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ρ_1[2:end-1,2:end-0].*g
            Ry_2                  .= ∂_∂y(τyy_1,τyy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ∂_∂x(τxy_2[:,1:end-1],τxy_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .-  ∂_∂y(P_1,P_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ρ_2.*g
            Rp_1                  .= .-∇v_1
            Rp_2                  .= .-∇v_2[:,1:end-1]
            # Calculate rate update --------------------------------------
            dVxdτ_1 .= (1-ρ) .* dVxdτ_1 .+ Rx_1
            dVydτ_1 .= (1-ρ) .* dVydτ_1 .+ Ry_1
            dVxdτ_2 .= (1-ρ) .* dVxdτ_2 .+ Rx_2
            dVydτ_2 .= (1-ρ) .* dVydτ_2 .+ Ry_2
            # Update velocity and pressure -------------------------------
            Vx_1[2:end-1,2:end-0] .+= Δτv_1 ./ ρ .* dVxdτ_1[2:end-1,2:end-0]
            Vy_1[2:end-1,2:end-0] .+= Δτv_1 ./ ρ .* dVydτ_1[2:end-1,2:end-0]
            Vx_2[2:end-1,2:end-1] .+= Δτv_2 ./ ρ .* dVxdτ_2
            Vy_2[2:end-1,2:end-1] .+= Δτv_2 ./ ρ .* dVydτ_2
            P_1                   .+= κΔτp_1 .* Rp_1
            P_2[:,1:end-1]        .+= κΔτp_2 .* Rp_2
            # Convergence check
            if mod(iter, nout)==0 || iter==1
                norm_Rx = 0.5*( norm(Rx_1)/sqrt(length(Rx_1)) + norm(Rx_2)/sqrt(length(Rx_2)) )
                norm_Ry = 0.5*( norm(Ry_1)/sqrt(length(Ry_1)) + norm(Ry_2)/sqrt(length(Ry_2)) )
                norm_Rp = 0.5*( norm(Rp_1)/sqrt(length(Rp_1)) + norm(Rp_2)/sqrt(length(Rp_2)) )
                @printf("it = %03d, iter = %05d, nRx=%1.3e nRy=%1.3e nRp=%1.3e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
                if (isnan(norm_Rx) || isnan(norm_Ry) || isnan(norm_Rp)) error("NaN"); end
                if (norm_Rx<tol && norm_Ry<tol && norm_Rp<tol) break; end
            end
            iter+=1; global itg=iter
        end
    end
    # # Plotting
    # p1 = heatmap( xv_1, yv_1, Vx_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="Vx_1")
    # p2 = heatmap( xv_1, yv_1, Vy_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="Vy_1")
    # p3 = heatmap( xc_1, yc_1, P_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="P_1")
    # p4 = heatmap( xc_2, yc_2, P_2', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="P_2")
    # display(plot(p1, p2, p3, p4))
    # Generate data
    vertx = [  xv2_1[1:end-1,1:end-1][:]  xv2_1[2:end-0,1:end-1][:]  xv2_1[2:end-0,2:end-0][:]  xv2_1[1:end-1,2:end-0][:] ] 
    verty = [  yv2_1[1:end-1,1:end-1][:]  yv2_1[2:end-0,1:end-1][:]  yv2_1[2:end-0,2:end-0][:]  yv2_1[1:end-1,2:end-0][:] ] 
    sol   = ( vx=Vx_2[2:end-1,2:end-1][:], vy=Vy_2[2:end-1,2:end-1][:], p=∂η∂xvWESN(P_1, P_2[:,1:end-1])[:], η=∂η∂xvWESN(η_1, η_2[:,1:end-1])[:])
    # sol   = ( vx=Rx_2[:,:][:], vy=Ry_2[:,:][:], p=∂η∂xvWESN(P_1, P_2[:,1:end-1])[:], η=∂η∂xvWESN(η_1, η_2[:,1:end-1])[:])

    xc2   = ∂η∂xvWESN(xc2_1, xc2_2[:,1:end-1])[:] 
    yc2   = ∂η∂xvWESN(yc2_1, yc2_2[:,1:end-1])[:]
    PatchPlotMakie(vertx, verty, sol, minimum(xv2_1), maximum(xv2_1), minimum(yv2_1), maximum(yv2_1), xv2_1[:], yv2_1[:], xc2[:], yc2[:], write_fig=false)
    
    file = matopen(string(@__DIR__,"/output_FS_topo.mat"), "w")
    write(file, "Vx_1", Vx_1)
    write(file, "Vx_2", Vx_2)
    write(file, "Vy_1", Vy_1)
    write(file, "Vy_2", Vy_2)
    write(file, "P_1", P_1)
    write(file, "P_2", P_2)

    write(file, "duNddudx",duNddudx)
    write(file, "duNddvdx",duNddvdx)
    write(file, "duNdP"   ,duNdP   )   
    write(file, "dvNddudx",dvNddudx)
    write(file, "dvNddvdx",dvNddvdx)
    write(file, "dvNdP"   ,dvNdP   )    
    write(file, "dkdx", Array(dkdx))
    write(file, "dkdy", Array(dkdy))
    write(file, "dedx", Array(dedx))
    write(file, "dedy", Array(dedy))
    write(file, "hx",   Array(h_x) )
    close(file)

    @show  ρ
    @show Δτv_1[1] ./ ρ
    @show κΔτp_2[1]

    return
end

Stokes2S_FSG()
# Stokes2S_FSG()
# Stokes2S_FSG()

