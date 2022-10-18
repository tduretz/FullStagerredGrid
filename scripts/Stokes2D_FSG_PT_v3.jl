# VISCOUS SINKER WITH TOPOGRAPHY AND FREE SURFACE
# Initialisation
using Plots, Printf, LinearAlgebra
import CairoMakie
using Makie.GeometryBasics
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,A) = (f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ A.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,C) = C.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy# .+ C.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx
@views    av(A)       = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A)       =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A)       =  0.5*(A[:,1:end-1].+A[:,2:end]) 
@views   avh(A)       = ( 0.25./A[1:end-1,1:end-1] .+ 0.25./A[1:end-1,2:end-0] .+ 0.25./A[2:end-0,1:end-1] .+ 0.25./A[2:end-0,2:end-0]).^(-1)
@views   avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
Dat = Float64  # Precision (double=Float64 or single=Float32)
function PatchPlotMakie(vertx, verty, sol, xmin, xmax, ymin, ymax, x1, y1, x2, y2; cmap = :turbo )
    f   = CairoMakie.Figure(resolution = (1200, 1000))

    ar = (xmax - xmin) / (ymax - ymin)

    CairoMakie.Axis(f[1,1], aspect = ar)
    min_v = minimum( sol.p ); max_v = maximum( sol.p )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    CairoMakie.poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    CairoMakie.Axis(f[2,1], aspect = ar)
    min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    CairoMakie.poly!(p, color = sol.vx, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    CairoMakie.Axis(f[2,2], aspect = ar)
    min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    CairoMakie.poly!(p, color = sol.vy, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
    # CairoMakie.scatter!(x1,y1, color=:white)
    # CairoMakie.scatter!(x2,y2, color=:white, marker=:xcross)
    CairoMakie.Colorbar(f, colormap = cmap, limits=limits, flipaxis = true, size = 25 )

    display(f)
    return nothing
end
@views h(x,A,σ,b)    = A*exp(-x^2/σ^2) + b
@views dhdx(x,A,σ,b) = -2*x/σ^2*A*exp(-x.^2/σ^2)
# 2D Poisson routine
@views function Stokes2S_FSG()
    # Physics
    xmin, xmax = -3.0, 3.0  
    ymin, ymax = -5.0, 0.0
    εbg      = -1*0
    rad      = 0.5
    y0       = -2.
    g        = -1
    adapt_mesh = false
    solve      = true
    # Numerics
    ncx, ncy = 51, 51    # numerical grid resolution
    ε        = 1e-6      # nonlinear tolerance
    iterMax  = 1e4       # max number of iters
    nout     = 1000      # check frequency
    # Iterative parameters -------------------------------------------
    if adapt_mesh
        Reopt    = 0.5*π*2
        cfl      = 1.1/2.5
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
    A = zeros(2ncx+3, 2ncy+3)
    C =  ones(2ncx+3, 2ncy+3)
    if adapt_mesh
        m      = ymin
        z0     =    -h.(xv4, 1., 0.5, ymax) 
        hx     = -dhdx.(xv4, 1., 0.5, ymax) # negative to conform with De La Puente paper
        A     .= (ymin .- yv4)./(z0.+m) .* hx
        C     .= ymin./(z0.+m)
        yv4    = (yv4/ymin).*((z0.+m)).-z0     
    end
    # Grid subsets
    xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1], yv4[2:2:end-1,2:2:end-1]
    xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0], yv4[1:2:end-0,1:2:end-0]
    xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1], yv4[3:2:end-2,2:2:end-1]
    xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    Ac_1 = A[3:2:end-2,2:2:end-1]; Av_1 = A[2:2:end-1,2:2:end-1]
    Ac_2 = A[2:2:end-1,3:2:end-2]; Av_2 = A[1:2:end-0,1:2:end-0]
    Cc_1 = C[3:2:end-2,2:2:end-1]; Cv_1 = C[2:2:end-1,2:2:end-1]
    Cc_2 = C[2:2:end-1,3:2:end-2]; Cv_2 = C[1:2:end-0,1:2:end-0]
    # Velocity
    Vx_1 .= -εbg.*xv2_1; Vx_2 .= -εbg.*xv2_2
    Vy_1 .=  εbg.*yv2_1; Vy_2 .=  εbg.*yv2_2
    # Viscosity
    η_1  .= 1.0; η_2  .= 1.0
    η_1[xc2_1.^2 .+ (yc2_1.-y0).^2 .< rad] .= 100.
    η_2[xc2_2.^2 .+ (yc2_2.-y0).^2 .< rad] .= 100.
    # Density
    ρ_1  .= 1.0; ρ_2  .= 1.0
    ρ_1[xv2_1.^2 .+ (yv2_1.-y0).^2 .< rad] .= 2.
    ρ_2[xv2_2[2:end-1,2:end-1].^2 .+ (yv2_2[2:end-1,2:end-1].-y0)^2 .< rad] .= 2.
    # Smooth Viscosity
    η_1_sm    = zeros(size(η_1))
    η_2_sm    = zeros(size(η_2,1), size(η_2,2)-1)
    η_1_sm   .= η_1
    η_2_sm   .= η_2[:,1:end-1]
    η_1_sm_c = av(η_1_sm)
    η_2_sm_c = av(η_2_sm)
    for it = 1:nsm
        η_1_sm[2:end-1,2:end-1] .= av(η_1_sm_c)
        η_1_sm_c = av(η_1_sm)
        η_2_sm[2:end-1,2:end-1] .= av(η_2_sm_c)
        η_2_sm_c = av(η_2_sm)
    end
    # PT steps
    Δτv_1  .= cfl .*ρ .*min(Δx,Δy)^2 ./ (0.50*(η_1_sm[1:end-1,2:end-0].+η_1_sm[2:end-0,2:end-0])) ./ 4.1 
    Δτv_2  .= cfl .*ρ .*min(Δx,Δy)^2 ./ (0.25*(η_1_sm[:,1:end-1].+η_1_sm[:,2:end-0].+η_2_sm[1:end-1,:].+η_2_sm[2:end-0,:])) ./ 4.1 
    κΔτp_1 .= cfl .* η_1 .* Δx ./ (xmax-xmin)
    κΔτp_2 .= cfl .* η_2[:,1:end-1] .* Δx ./ (xmax-xmin)
    if solve
    # PT loop
    it=1; iter=1; err=2*ε; err_evo1=[]; err_evo2=[];
    while (err>ε && iter<=iterMax)
        # dvydy = 1/2*dv(dvxdx) + 3/4 * P/eta
        Vy_2[2:end-1,end] = Vy_2[2:end-1,end-1] - Δy*1/2*(Vx_1[2:end-0,end] - Vx_1[1:end-1,end])/Δx
        # dvxdy = -(dvydx)
        Vx_2[2:end-1,end] = Vx_2[2:end-1,end-1] - Δy*1/1*(Vy_1[2:end-0,end] - Vy_1[1:end-1,end])/Δx
        ∇v_1  .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,Ac_1) .+ ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,Cc_1) 
        ∇v_2[:,1:end-1]  .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,Ac_2) .+ ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,Cc_2) 
        ε̇xx_1 .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,Ac_1) .- 1.0/3.0*∇v_1
        ε̇yy_1 .=  ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,Cc_1) .- 1.0/3.0*∇v_1
        ε̇xy_1 .= (∂_∂y(Vx_2[2:end-1,:],Vx_1,Δx,Δy,Cc_1) .+ ∂_∂x(Vy_1,Vy_2[2:end-1,:], Δx,Δy,Ac_1) ) / 2.
        ε̇xx_2[:,1:end-1] .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,Ac_2) .- 1.0/3.0*∇v_2[:,1:end-1]
        ε̇yy_2[:,1:end-1] .=  ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,Cc_2) .- 1.0/3.0*∇v_2[:,1:end-1]
        ε̇xy_2[:,1:end-1] .= (∂_∂y(Vx_1,Vx_2[:,2:end-1],Δx,Δy,Cc_2) .+ ∂_∂x(Vy_2[:,2:end-1],Vy_1, Δx,Δy,Ac_2) ) / 2.
        τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
        τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
        τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
        τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
        τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
        τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
        Rx_1[2:end-1,2:end-0] .= ∂_∂x(τxx_1[:,2:end-0],τxx_2[2:end-1,:],Δx,Δy,Av_1[2:end-1,2:end-0]) .+ ∂_∂y(τxy_2[2:end-1,:],τxy_1[:,2:end-0],Δx,Δy,Cv_1[2:end-1,2:end-0]) .-  ∂_∂x(P_1[:,2:end-0],P_2[2:end-1,:],Δx,Δy,Av_1[2:end-1,2:end-0])
        Rx_2                  .= ∂_∂x(τxx_2[:,1:end-1],τxx_1,           Δx,Δy,Av_2[2:end-1,2:end-1]) .+ ∂_∂y(τxy_1,τxy_2[:,1:end-1],           Δx,Δy,Cv_2[2:end-1,2:end-1]) .-  ∂_∂x(P_2[:,1:end-1],P_1,           Δx,Δy,Av_2[2:end-1,2:end-1]) 
        Ry_1[2:end-1,2:end-0] .= ∂_∂y(τyy_2[2:end-1,:],τyy_1[:,2:end-0],Δx,Δy,Cv_1[2:end-1,2:end-0]) .+ ∂_∂x(τxy_1[:,2:end-0],τxy_2[2:end-1,:],Δx,Δy,Av_1[2:end-1,2:end-0]) .-  ∂_∂y(P_2[2:end-1,:],P_1[:,2:end-0],Δx,Δy,Cv_1[2:end-1,2:end-0]) .+ ρ_1[2:end-1,2:end-0].*g
        Ry_2                  .= ∂_∂y(τyy_1,τyy_2[:,1:end-1],           Δx,Δy,Cv_2[2:end-1,2:end-1]) .+ ∂_∂x(τxy_2[:,1:end-1],τxy_1,           Δx,Δy,Av_2[2:end-1,2:end-1]) .-  ∂_∂y(P_1,P_2[:,1:end-1],           Δx,Δy,Cv_2[2:end-1,2:end-1]) .+ ρ_2.*g
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
            @printf("it = %03d, iter = %04d, nRx=%1.3e nRy=%1.3e nRp=%1.3e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
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
    sol   = ( vx=Vx_2[2:end-1,2:end-1][:], vy=Vy_2[2:end-1,2:end-1][:], p=avWESN(P_1, P_2[:,1:end-1])[:])
    xc2   = avWESN(xc2_1, xc2_2[:,1:end-1])[:] 
    yc2   = avWESN(yc2_1, yc2_2[:,1:end-1])[:]
    PatchPlotMakie(vertx, verty, sol, minimum(xv2_1), maximum(xv2_1), minimum(yv2_1), maximum(yv2_1), xv2_1[:], yv2_1[:], xc2[:], yc2[:])
    return
end

Stokes2S_FSG()
