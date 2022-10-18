# VISCOUS SINKER
# Initialisation
using Plots, Printf, LinearAlgebra
import CairoMakie
using Makie.GeometryBasics
Dat = Float64  # Precision (double=Float64 or single=Float32)
# Macros
@views    ∂_∂x1(f1,f2,Δx,Δy,A) = (f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ A.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y1(f1,f2,Δx,Δy,C) = (f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy .+ C.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx

@views    ∂_∂x(f,Δx,Δy,A) = (f[2:size(f,1),:] .- f[1:size(f,1)-1,:]) ./ Δx
@views    ∂_∂y(f,Δx,Δy,C) = (f[:,2:size(f,2)] .- f[:,1:size(f,2)-1]) ./ Δy
@views    av(A)      = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A)      =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A)      =  0.5*(A[:,1:end-1].+A[:,2:end]) 
@views   avh(A)      = ( 0.25./A[1:end-1,1:end-1] .+ 0.25./A[1:end-1,2:end-0] .+ 0.25./A[2:end-0,1:end-1] .+ 0.25./A[2:end-0,2:end-0]).^(-1)
@views   avWESN(A,B) = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
function PatchPlotMakie(vertx, verty, msh, v, xmin, xmax, ymin, ymax, x1, y1, x2, y2; cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    f = CairoMakie.Figure()
    ar = (maximum(msh.xv) - minimum(msh.xv)) / (maximum(msh.xv) - minimum(msh.yv))
    CairoMakie.Axis(f[1, 1], aspect = ar)
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (msh.xv[msh.e2v[i,j]], msh.yv[msh.e2v[i,j]]) for j=1:msh.nf_el] ) for i in 1:msh.nel]
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:msh.nf_el] ) for i in 1:msh.nel]
    CairoMakie.poly!(p, color = v, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    # CairoMakie.scatter!(x1,y1, color=:white)
    # CairoMakie.scatter!(x2,y2, color=:white, marker=:xcross)
    CairoMakie.Colorbar(f[1, 2], colormap = cmap, limits=limits, flipaxis = true, size = 25 )
    display(f)
    return nothing
end
@views h(x,s,b) = x*tand(s) + b + 0*x^2*tand(s)/10
@views h_exp(x,A,σ,b) = A*exp(-x^2/σ^2) + b
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
    iterMax  = 1e4     # max number of iters
    nout     = 1000       # check frequency
    # Iterative parameters -------------------------------------------
    Reopt    = 0.625*π*2
    cfl      = 1.1
    ρ        = cfl*Reopt/ncx
    nsm      = 4
    tol      = 1e-8
    # Reopt    = 0.625*π /3
    # cfl      = 0.3*4
    # ρ        = cfl*Reopt/ncx
    # nsm      = 10
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
    ε̇xx_2       =   zeros(Dat, ncx+1, ncy-0)
    ε̇yy_2       =   zeros(Dat, ncx+1, ncy-0)
    ε̇xy_2       =   zeros(Dat, ncx+1, ncy-0)
    ∇v_2        =   zeros(Dat, ncx+1, ncy-0)
    τxx_1       =   zeros(Dat, ncx-0, ncy+1)
    τyy_1       =   zeros(Dat, ncx-0, ncy+1)
    τxy_1       =   zeros(Dat, ncx-0, ncy+1)
    P_1         =   zeros(Dat, ncx-0, ncy+1)
    η_1         =   zeros(Dat, ncx-0, ncy+1)
    τxx_2       =   zeros(Dat, ncx+1, ncy-0)
    τyy_2       =   zeros(Dat, ncx+1, ncy-0)
    τxy_2       =   zeros(Dat, ncx+1, ncy-0)
    P_2         =   zeros(Dat, ncx+1, ncy-0)
    η_2         =   zeros(Dat, ncx+1, ncy-0)
    Rx_1        =   zeros(Dat, ncx+1, ncy+1)
    Ry_1        =   zeros(Dat, ncx+1, ncy+1)
    ρ_1         =   zeros(Dat, ncx+1, ncy+1)
    Rx_2        =   zeros(Dat, ncx+0, ncy+0) # not extended
    Ry_2        =   zeros(Dat, ncx+0, ncy+0)
    ρ_2         =   zeros(Dat, ncx+0, ncy-0)
    Rp_1        =   zeros(Dat, ncx-0, ncy+1)
    Rp_2        =   zeros(Dat, ncx+1, ncy-0)
    Δτv_1       =   zeros(Dat, ncx-1, ncy-1)
    Δτv_2       =   zeros(Dat, ncx-0, ncy-0)
    κΔτp_1      =   zeros(Dat, ncx-0, ncy+1)
    κΔτp_2      =   zeros(Dat, ncx+1, ncy-0)
    dVxdτ_1     =   zeros(Dat, ncx+1, ncy+1)
    dVydτ_1     =   zeros(Dat, ncx+1, ncy+1)
    dVxdτ_2     =   zeros(Dat, ncx+0, ncy+0)
    dVydτ_2     =   zeros(Dat, ncx+0, ncy+0)
    # Initialisation
    xv_1, yv_1  = LinRange(xmin, xmax, ncx+1), LinRange(ymin, ymax, ncy+1)
    xv_2, yv_2  = LinRange(xmin-Δx/2, xmax+Δx/2, ncx+2), LinRange(ymin-Δy/2, ymax+Δy/2, ncy+2)
    xc_1, yc_1  = LinRange(xmin+Δx/2, xmax-Δx/2, ncx+0), LinRange(ymin, ymax, ncy+1)
    xc_2, yc_2  = LinRange(xmin, xmax, ncx+1), LinRange(ymin+Δy/2, ymax-Δy/2, ncy+0)
    (xv2_1, yv2_1) = ([x for x=xv_1,y=yv_1], [y for x=xv_1,y=yv_1])
    (xv2_2, yv2_2) = ([x for x=xv_2,y=yv_2], [y for x=xv_2,y=yv_2])
    (xc2_1, yc2_1) = ([x for x=xc_1,y=yc_1], [y for x=xc_1,y=yc_1])
    (xc2_2, yc2_2) = ([x for x=xc_2,y=yc_2], [y for x=xc_2,y=yc_2])
    numv  = reshape(1:(ncx+1)*(ncy+1), ncx+1, ncy+1)
    xv, yv    = LinRange(xmin, xmax, ncx+1), LinRange(ymin, ymax, ncy+1)
    xc, yc    = LinRange(xmin+Δx/2, xmax-Δx/2, ncx+1), LinRange(ymin+Δy/2, ymax-Δy/2, ncy+1)
    (xv2,yv2) = ([x for x=xv,y=yv], [y for x=xv,y=yv])
    (xc2,yc2) = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    if adapt_mesh
        # z0    = h.(xv2, slope, ord)
        m      = ymin
        z0     = h_exp.(xv2, 1., 0.5, ymax)
        yv2    = (yv2/ymin).*((-z0.+m)).+z0
        z0     = h_exp.(xc2, 1., 0.5, ymax)
        yc2    = (yc2/ymin).*((-z0.+m)).+z0
        z0     = h_exp.(xv2_1, 1., 0.5, ymax)
        yv2_1  = (yv2_1/ymin).*((-z0.+m)).+z0
        z0     = h_exp.(xv2_2, 1., 0.5, ymax)
        yv2_2  = (yv2_2/ymin).*((-z0.+m)).+z0
        z0     = h_exp.(xc2_1, 1., 0.5, ymax)
        yc2_1  = (yc2_1/ymin).*((-z0.+m)).+z0
        z0     = h_exp.(xc2_2, 1., 0.5, ymax)
        yc2_2  = (yc2_2/ymin).*((-z0.+m)).+z0
    end

    A = 0.
    C = 0.

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
    η_2_sm    = zeros(size(η_2))
    η_1_sm   .= η_1
    η_2_sm   .= η_2
    η_1_sm_c = av(η_1_sm)
    η_2_sm_c = av(η_2_sm)
    for it = 1:nsm
        η_1_sm[2:end-1,2:end-1] .= av(η_1_sm_c)
        η_1_sm_c = av(η_1_sm)
        η_2_sm[2:end-1,2:end-1] .= av(η_2_sm_c)
        η_2_sm_c = av(η_2_sm)
    end
    # PT steps
    Δτv_1  .= cfl .*ρ .*min(Δx,Δy)^2 ./ (0.50*(η_1_sm[1:end-1,2:end-1].+η_1_sm[2:end-0,2:end-1])) ./ 4.1 
    Δτv_2  .= cfl .*ρ .*min(Δx,Δy)^2 ./ (0.25*(η_1_sm[:,1:end-1].+η_1_sm[:,2:end-0].+η_2_sm[1:end-1,:].+η_2_sm[2:end-0,:])) ./ 4.1 
    κΔτp_1 .= cfl .* η_1 .* Δx ./ (xmax-xmin)
    κΔτp_2 .= cfl .* η_2 .* Δx ./ (xmax-xmin)
    if solve
    # PT loop
    it=1; iter=1; err=2*ε; err_evo1=[]; err_evo2=[];
    while (err>ε && iter<=iterMax)
        ∇v_1  .=  ∂_∂x1(Vx_1,Vx_2[2:end-1,:],Δx,Δy,A) .+ ∂_∂y1(Vy_2[2:end-1,:],Vy_1,Δx,Δy,C) 
        ∇v_2  .=  ∂_∂x1(Vx_2[:,2:end-1],Vx_1,Δx,Δy,A) .+ ∂_∂y1(Vy_1,Vy_2[:,2:end-1],Δx,Δy,C) 
        ε̇xx_1 .=  ∂_∂x1(Vx_1,Vx_2[2:end-1,:],Δx,Δy,A) .- 1.0/3.0*∇v_1
        ε̇yy_1 .=  ∂_∂y1(Vy_2[2:end-1,:],Vy_1,Δx,Δy,C) .- 1.0/3.0*∇v_1
        ε̇xy_1 .= (∂_∂y1(Vx_2[2:end-1,:],Vx_1,Δx,Δy,C) .+ ∂_∂x1(Vy_1,Vy_2[2:end-1,:], Δx,Δy,A) ) / 2.
        ε̇xx_2 .=  ∂_∂x1(Vx_2[:,2:end-1],Vx_1,Δx,Δy,A) .- 1.0/3.0*∇v_2
        ε̇yy_2 .=  ∂_∂y1(Vy_1,Vy_2[:,2:end-1],Δx,Δy,C) .- 1.0/3.0*∇v_2
        ε̇xy_2 .= (∂_∂y1(Vx_1,Vx_2[:,2:end-1]           ,Δx,Δy,C) .+ ∂_∂x1(Vy_2[:,2:end-1],Vy_1, Δx,Δy,A) ) / 2.
        τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
        τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
        τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
        τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
        τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
        τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
        Rx_1[2:end-1,2:end-1] .= ∂_∂x1(τxx_1[:,2:end-1],τxx_2[2:end-1,:],Δx,Δy,A) .+ ∂_∂y1(τxy_2[2:end-1,:],τxy_1[:,2:end-1],Δx,Δy,C) .-  ∂_∂x1(P_1[:,2:end-1],P_2[2:end-1,:],Δx,Δy,A)
        Rx_2                  .= ∂_∂x1(τxx_2,τxx_1,                      Δx,Δy,A) .+ ∂_∂y1(τxy_1,τxy_2,                      Δx,Δy,C) .-  ∂_∂x1(P_2,P_1,                      Δx,Δy,A) 
        Ry_1[2:end-1,2:end-1] .= ∂_∂y1(τyy_2[2:end-1,:],τyy_1[:,2:end-1],Δx,Δy,C) .+ ∂_∂x1(τxy_1[:,2:end-1],τxy_2[2:end-1,:],Δx,Δy,A) .-  ∂_∂y1(P_2[2:end-1,:],P_1[:,2:end-1],Δx,Δy,C) .+ ρ_1[2:end-1,2:end-1].*g
        Ry_2                  .= ∂_∂y1(τyy_1,τyy_2,                      Δx,Δy,C) .+ ∂_∂x1(τxy_2,τxy_1,                      Δx,Δy,A) .-  ∂_∂y1(P_1,P_2,                      Δx,Δy,C) .+ ρ_2.*g
        Rp_1                  .= .-∇v_1
        Rp_2                  .= .-∇v_2 
        # Calculate rate update --------------------------------------
        dVxdτ_1 .= (1-ρ) .* dVxdτ_1 .+ Rx_1
        dVydτ_1 .= (1-ρ) .* dVydτ_1 .+ Ry_1
        dVxdτ_2 .= (1-ρ) .* dVxdτ_2 .+ Rx_2
        dVydτ_2 .= (1-ρ) .* dVydτ_2 .+ Ry_2
        # Update velocity and pressure -------------------------------
        Vx_1[2:end-1,2:end-1] .+= Δτv_1 ./ ρ .* dVxdτ_1[2:end-1,2:end-1]
        Vy_1[2:end-1,2:end-1] .+= Δτv_1 ./ ρ .* dVydτ_1[2:end-1,2:end-1]
        Vx_2[2:end-1,2:end-1] .+= Δτv_2 ./ ρ .* dVxdτ_2
        Vy_2[2:end-1,2:end-1] .+= Δτv_2 ./ ρ .* dVydτ_2
        P_1                   .+= κΔτp_1 .* Rp_1
        P_2                   .+= κΔτp_2 .* Rp_2
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
    verts = [ numv[1:end-1,1:end-1][:] numv[2:end-0,1:end-1][:] numv[2:end-0,2:end-0][:] numv[1:end-1,2:end-0][:] ] 
    vertx = [  xv2[1:end-1,1:end-1][:]  xv2[2:end-0,1:end-1][:]  xv2[2:end-0,2:end-0][:]  xv2[1:end-1,2:end-0][:] ] 
    verty = [  yv2[1:end-1,1:end-1][:]  yv2[2:end-0,1:end-1][:]  yv2[2:end-0,2:end-0][:]  yv2[1:end-1,2:end-0][:] ] 

    nel  = 4
    nc   = ncx*ncy
    v    = avWESN(P_1, P_2)[:]
    msh  = (xv=xv2[:], yv=yv2[:], e2v=verts, nf_el=4, nel=nc )
    PatchPlotMakie(vertx, verty, msh, v, minimum(xv), maximum(xv), minimum(yv), maximum(yv), xv2[:], yv2[:], xc2[:], yc2[:])

    return
end

Stokes2S_FSG()
