# VISCOUS INCLUSION PURE SHEAR TEST
# Initialisation
using FullStaggeredGrid
using Plots, Printf, LinearAlgebra
import CairoMakie
using Makie.GeometryBasics
using ForwardDiff
Dat = Float64  # Precision (double=Float64 or single=Float32)
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
function Mesh_y( X, A, x0, σ, b, m, ymin0, ymax0, σy, swiss )
    y     = X[2]
    return y
end
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end]) 
@views   avh(A) = ( 0.25./A[1:end-1,1:end-1] .+ 0.25./A[1:end-1,2:end-0] .+ 0.25./A[2:end-0,1:end-1] .+ 0.25./A[2:end-0,2:end-0]).^(-1)
function PatchPlotMakie(vertx, verty, msh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    f = CairoMakie.Figure()
    ar = (maximum(msh.xv) - minimum(msh.xv)) / (maximum(msh.xv) - minimum(msh.yv))
    CairoMakie.Axis(f[1, 1], aspect = ar)
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (msh.xv[msh.e2v[i,j]], msh.yv[msh.e2v[i,j]]) for j=1:msh.nf_el] ) for i in 1:msh.nel]
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:msh.nf_el] ) for i in 1:msh.nel]
    CairoMakie.poly!(p, color = v, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio=:image, colorrange=limits)
    CairoMakie.Colorbar(f[1, 2], colormap = cmap, limits=limits, flipaxis = true, size = 25 )
    display(f)
    return nothing
end
@views     h(x,s,b) = x*tand(s) + b + 0*x^2*tand(s)/10
# 2D Poisson routine
@views function Stokes2S_FSG()
    # Physics
    adapt_mesh=true
    swiss=true
    Lx, Ly   = 6.0, 6.0  # domain size
    xmin = -Lx/2
    xmax =  Lx/2
    ymin = -Ly/2
    ymax =  Ly/2
    εbg      = -1
    rad      = 0.5
    # Numerics
    ncx, ncy = 61, 61    # numerical grid resolution
    ε        = 1e-6      # nonlinear tolerence
    iterMax  = 1e4     # max number of iters
    nout     = 1000       # check frequency
    # Iterative parameters -------------------------------------------
    Reopt    = 0.625*π*2
    cfl      = 1.1/2
    ρ        = cfl*Reopt/ncx
    nsm      = 4
    tol      = 1e-8
    # Reopt    = 0.625*π /3
    # cfl      = 0.3*4
    # ρ        = cfl*Reopt/ncx
    # nsm      = 10
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
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
    Rx_2        =   zeros(Dat, ncx+0, ncy+0) # not extended
    Ry_2        =   zeros(Dat, ncx+0, ncy+0)
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
    xv_1, yv_1  = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    xv_2, yv_2  = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc_1, yc_1  = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx+0), LinRange(-Ly/2, Ly/2, ncy+1)
    xc_2, yc_2  = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy+0)
    (xv2_1, yv2_1) = ([x for x=xv_1,y=yv_1], [y for x=xv_1,y=yv_1])
    (xv2_2, yv2_2) = ([x for x=xv_2,y=yv_2], [y for x=xv_2,y=yv_2])
    (xc2_1, yc2_1) = ([x for x=xc_1,y=yc_1], [y for x=xc_1,y=yc_1])
    (xc2_2, yc2_2) = ([x for x=xc_2,y=yc_2], [y for x=xc_2,y=yc_2])
    xxv, yyv    = LinRange(xmin-Δx/2, xmax+Δx/2, 2ncx+3), LinRange(ymin-Δy/2, ymax+Δy/2, 2ncy+3)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    ∂ξ∂x =  ones(2ncx+3, 2ncy+3)
    ∂ξ∂y = zeros(2ncx+3, 2ncy+3)
    ∂η∂x = zeros(2ncx+3, 2ncy+3)
    ∂η∂y =  ones(2ncx+3, 2ncy+3)
    # hx   = zeros(2ncx+3, 2ncy+3)
    if adapt_mesh
        x0     = (xmax + xmin)/2
        m      = ymin
        Amp    = 1.0
        σ      = 0.5
        σx     = 0.5
        σy     = 0.5
        ϵ      = 1e-6
        # copy initial y
        x_ini  = copy(xv4)
        y_ini  = copy(yv4)
        X_msh  = zeros(2)
        # Compute slope
        # hx     = -dhdx.(x_ini, Amp, σ, ymax, x0)
        # Deform mesh
        for i in eachindex(x_ini)          
            X_msh[1] = x_ini[i]
            X_msh[2] = y_ini[i]     
            xv4[i]   =  Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx, swiss )
            yv4[i]   =  Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, swiss )
        end
    end
    # Grid subsets
    xce_1, yce_1 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
    xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+1], yv4[2:2:end-1,3:2:end-2+1]
    ∂ξ∂xc_1 = Δx ./ (xv2_1[2:end,:].-xv2_1[1:end-1,:])
    ∂ξ∂xc_2 = Δx ./ (xce_1[2:end,2:end-1].-xce_1[1:end-1,2:end-1])
    ∂ξ∂xv_1 = Δx ./ (xce_1[2:end,1:end-1].-xce_1[1:end-1,1:end-1])
    ∂ξ∂xv_2 = Δx ./ (xv2_1[2:end,1:end-1].-xv2_1[1:end-1,1:end-1])
    # Velocity
    Vx_1 .= -εbg.*xv2_1; Vx_2 .= -εbg.*xv2_2
    Vy_1 .=  εbg.*yv2_1; Vy_2 .=  εbg.*yv2_2
    # Viscosity
    η_1  .= 1.0; η_2  .= 1.0
    η_1[xc2_1.^2 .+ yc2_1.^2 .< rad] .= 100.
    η_2[xc2_2.^2 .+ yc2_2.^2 .< rad] .= 100.
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
    κΔτp_1 .= cfl .* η_1 .* Δx ./ Lx
    κΔτp_2 .= cfl .* η_2 .* Δx ./ Lx
    # PT loop
    it=1; iter=1; err=2*ε; err_evo1=[]; err_evo2=[];
    while (err>ε && iter<=iterMax)
        ∇v_1  .=  ∂ξ∂xc_1.*diff(Vx_1,            dims=1)./Δx .+ diff(Vy_2[2:end-1,:], dims=2)./Δy
        ∇v_2  .=  ∂ξ∂xc_2.*diff(Vx_2[:,2:end-1], dims=1)./Δx .+ diff(Vy_1,            dims=2)./Δy
        # ∇v_1  .=  (∂ξ∂xv_1[2:end,:].*Vx_1[2:end,:].-∂ξ∂xv_1[1:end-1,:].*Vx_1[1:end-1,:])./Δx .+ diff(Vy_2[2:end-1,:], dims=2)./Δy
        # ∇v_2  .=  (∂ξ∂xv_2[2:end,2:end-1].*Vx_2[2:end,2:end-1].-∂ξ∂xv_2[1:end-1,2:end-1].*Vx_2[1:end-1,2:end-1])./Δx .+ diff(Vy_1,            dims=2)./Δy
        ε̇xx_1 .=  ∂ξ∂xc_1.*diff(Vx_1,            dims=1)./Δx .- 1.0/3.0*∇v_1
        ε̇yy_1 .=  diff(Vy_2[2:end-1,:], dims=2)./Δy .- 1.0/3.0*∇v_1
        ε̇xy_1 .= (diff(Vx_2[2:end-1,:], dims=2)./Δy .+ ∂ξ∂xc_1.*diff(Vy_1, dims=1)./Δx) / 2.
        ε̇xx_2 .=  ∂ξ∂xc_2.*diff(Vx_2[:,2:end-1], dims=1)./Δx .- 1.0/3.0*∇v_2
        ε̇yy_2 .=  diff(Vy_1,            dims=2)./Δy .- 1.0/3.0*∇v_2
        ε̇xy_2 .= (diff(Vx_1,            dims=2)./Δy .+ ∂ξ∂xc_2.*diff(Vy_2[:,2:end-1], dims=1)./Δx) / 2.
        τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
        τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
        τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
        τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
        τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
        τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
        Rx_1[2:end-1,2:end-1] .= ∂ξ∂xv_1[2:end-1,2:end-1].*diff(τxx_1[:,2:end-1], dims=1)./Δx .+ diff(τxy_2[2:end-1,:], dims=2)./Δy .-  ∂ξ∂xv_1[2:end-1,2:end-1].*diff(P_1[:,2:end-1], dims=1)./Δx 
        Rx_2                  .= ∂ξ∂xv_2.*diff(τxx_2,            dims=1)./Δx .+ diff(τxy_1,            dims=2)./Δy .-  ∂ξ∂xv_2.*diff(P_2           , dims=1)./Δx 
        Ry_1[2:end-1,2:end-1] .= diff(τyy_2[2:end-1,:], dims=2)./Δy .+ ∂ξ∂xv_1[2:end-1,2:end-1].*diff(τxy_1[:,2:end-1], dims=1)./Δx .-  diff(P_2[2:end-1,:], dims=2)./Δy
        Ry_2                  .= diff(τyy_1,            dims=2)./Δy .+ ∂ξ∂xv_2.*diff(τxy_2,            dims=1)./Δx .-  diff(P_1           , dims=2)./Δy 
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
    @printf("%2.2e %2.2e\n", minimum(P_1) , maximum(P_1))

    # Plotting
    p1 = heatmap( xv_1, yv_1, Vx_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="Vx_1")
    p2 = heatmap( xv_1, yv_1, Vy_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="Vy_1")
    p3 = heatmap( xc_1, yc_1, P_1', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="P_1")
    p4 = heatmap( xc_2, yc_2, P_2', aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), c=:turbo, title="P_2")

#     y_int = h.(xv, slope, ord)
#     p4 = plot(xv, y_int, aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), color=:black, label=:none)
#     ied1  = [ 1 2 3 4] 
#     ied2  = [ 2 3 4 1] 
#     for ic=1:length(XC)
#         for ied=1:4
#             edgex = [vertx[ic, ied1[ied]]; vertx[ic, ied2[ied]]]
#             edgey = [verty[ic, ied1[ied]]; verty[ic, ied2[ied]]]
#             p4 = plot!( edgex, edgey, color=:red, label=:none)
#         end
#     end
#     p4 = plot!( XC, YC , marker=:circle, linewidth=0, color=:green, label=:none)
#     p4 = plot!( xqx3, yqx3, marker=:cross, linewidth=0, color=:green, label=:none)
#     p4 = plot!( xqy3, yqy3, marker=:cross, linewidth=0, color=:blue, label=:none)
#     # lines
#     # for j=1:ncy+1
#     #     y_line = yv[j]*ones(size(xv))
#     #     p4 = plot!(xv, y_line, aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), color=:gray, label=:none)
#     # end
#     # for i=1:ncx+1
#     #     x_line = xv[i]*ones(size(yv))
#     #     p4 = plot!(x_line, yv, aspect_ratio=1, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2), color=:gray, label=:none)
#     # end
#     # p4 = plot!( xc2, yc2, marker=:circle, linewidth=0, color=:green, label=:none)
    display(plot(p1, p2, p3, p4))


#     # Generate data
#     nel  = 4
#     v    = log10.(kc[:])
#     msh  = (xv=xv3[:], yv=yv3[:], e2v=verts, nf_el=4, nel=nc )

#     # Call function
#     PatchPlotMakie(vertx, verty, msh, v, minimum(xv), maximum(xv), minimum(yv), maximum(yv))

    return
end

Stokes2S_FSG()
