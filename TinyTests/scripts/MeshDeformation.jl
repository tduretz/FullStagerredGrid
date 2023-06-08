    
@views function FreeSurfaceDiscretisation(ηy, ∂ξ, ∂η, hx )
    # Topography # See python notebook v5
    η_surf   = ηy[:,end]
    dkdx     = ∂ξ.∂x[1:2:end, end-1]
    dkdy     = ∂ξ.∂y[1:2:end, end-1]
    dedx     = ∂η.∂x[1:2:end, end-1]
    dedy     = ∂η.∂y[1:2:end, end-1]
    h_x      = hx[1:2:end]
    eta      = η_surf
    free_surface_params = (; # removed factor dz since we apply it directly to strain rates
        ∂Vx∂∂Vx∂x = (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂∂Vy∂x = (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂P     = (3 // 2) .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2)),
        ∂Vy∂∂Vx∂x = (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂∂Vy∂x = (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂P     = (3 // 2) .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)),
    )
    return free_surface_params
end

###############################


@views function CopyJacobianToDevice!(∂, ∂ξ, ∂η)
    ∂.ξ.∂x.x .= to_device( ∂ξ.∂x[2:2:end-1,1:2:end-0])
    ∂.ξ.∂x.y .= to_device( ∂ξ.∂x[1:2:end-0,2:2:end-1])
    ∂.ξ.∂x.c .= to_device( ∂ξ.∂x[1:2:end-0,1:2:end-0])
    ∂.ξ.∂x.v .= to_device( ∂ξ.∂x[2:2:end-1,2:2:end-1])
    ∂.ξ.∂y.x .= to_device( ∂ξ.∂y[2:2:end-1,1:2:end-0])
    ∂.ξ.∂y.y .= to_device( ∂ξ.∂y[1:2:end-0,2:2:end-1])
    ∂.ξ.∂y.c .= to_device( ∂ξ.∂y[1:2:end-0,1:2:end-0])
    ∂.ξ.∂y.v .= to_device( ∂ξ.∂y[2:2:end-1,2:2:end-1])
    ∂.η.∂x.x .= to_device( ∂η.∂x[2:2:end-1,1:2:end-0])
    ∂.η.∂x.y .= to_device( ∂η.∂x[1:2:end-0,2:2:end-1])
    ∂.η.∂x.c .= to_device( ∂η.∂x[1:2:end-0,1:2:end-0])
    ∂.η.∂x.v .= to_device( ∂η.∂x[2:2:end-1,2:2:end-1])
    ∂.η.∂y.x .= to_device( ∂η.∂y[2:2:end-1,1:2:end-0])
    ∂.η.∂y.y .= to_device( ∂η.∂y[1:2:end-0,2:2:end-1])
    ∂.η.∂y.c .= to_device( ∂η.∂y[1:2:end-0,1:2:end-0])
    ∂.η.∂y.v .= to_device( ∂η.∂y[2:2:end-1,2:2:end-1])
    return nothing
end

###############################

function Mesh_x( X, h, x0, m, xmin0, xmax0, σx, options )
if options.swiss_x
        xmin1 = (sinh.( σx.*(xmin0.-x0) ))
        xmax1 = (sinh.( σx.*(xmax0.-x0) ))
        sx    = (xmax0-xmin0)/(xmax1-xmin1)
        x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0
    else
        x = X[1]
    end
    return x
end

###############################

function Mesh_y( X, h, y0, m, ymin0, ymax0, σy, options )
    # y0    = ymax0
    y     = X[2]
    if options.swiss_y
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    end
    if options.topo
        z0    = -h                     # topography height
        y     = (y/ymin0)*((z0+m))-z0  # shift grid vertically
    end   
    return y
end

###############################

@views function dhdx_num(xv4, yv4, Δ)
    #  ∂h∂ξ * ∂ξ∂x + ∂h∂η * ∂η∂x
    ∂h∂x = zero(xv4[:,end])
    ∂h∂x[2:2:end-1] .= (yv4[3:2:end-0,end-1] .- yv4[1:2:end-2,end-1])./Δ.ξ
    ∂h∂x[3:2:end-2] .= (yv4[4:2:end-1,end-1] .- yv4[2:2:end-3,end-1])./Δ.ξ
    ∂h∂x[[1 end]]   .= ∂h∂x[[2 end-1]]   
    return ∂h∂x
end

 ###############################

 function ComputeForwardTransformation!(∂x, ∂y, xv4, yv4, Δ)
    # ----------------------
    ∂x.∂η[:,2:2:end-1] .= (xv4[:,3:2:end-0] .- xv4[:,1:2:end-2])./Δ.η
    ∂x.∂η[:,3:2:end-2] .= (xv4[:,4:2:end-1] .- xv4[:,2:2:end-3])./Δ.η
    ∂x.∂η[:,[1 end]]   .= ∂x.∂η[:,[2 end-1]]
    # ----------------------
    ∂y.∂ξ[2:2:end-1,:] .= (yv4[3:2:end-0,:] .- yv4[1:2:end-2,:])./Δ.ξ
    ∂y.∂ξ[3:2:end-2,:] .= (yv4[4:2:end-1,:] .- yv4[2:2:end-3,:])./Δ.ξ
    ∂y.∂ξ[[1 end],:]   .= ∂y.∂ξ[[2 end-1],:]  
    # ----------------------
    ∂y.∂η[:,2:2:end-1] .= (yv4[:,3:2:end-0] .- yv4[:,1:2:end-2])./Δ.η
    ∂y.∂η[:,3:2:end-2] .= (yv4[:,4:2:end-1] .- yv4[:,2:2:end-3])./Δ.η
    ∂y.∂η[:,[1 end]]   .= ∂y.∂η[:,[2 end-1]]     
    return nothing
 end

function ComputeForwardTransformation_ini!( ∂x, ∂y, x_ini, y_ini, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵ, options)
 
    @time for i in eachindex(y_ini)          
    
        # compute dxdksi
        X_msh[1] = x_ini[i]-ϵ
        X_msh[2] = y_ini[i] 
        xm       = Mesh_x( X_msh,  Amp, x0, σ, xmax, m, xmin, xmax, σx, options )
        # --------
        X_msh[1] = x_ini[i]+ϵ
        X_msh[2] = y_ini[i]
        xp       = Mesh_x( X_msh,  Amp, x0, σ, xmax, m, xmin, xmax, σx, options )
        # --------
        ∂x.∂ξ[i] = (xp - xm) / (2ϵ)
    
        # compute dydeta
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]-ϵ
        xm     = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]+ϵ
        xp       = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        ∂x.∂η[i] = (xp - xm) / (2ϵ)
    
        # compute dydksi
        X_msh[1] = x_ini[i]-ϵ
        X_msh[2] = y_ini[i] 
        ym       = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        X_msh[1] = x_ini[i]+ϵ
        X_msh[2] = y_ini[i]
        yp       = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        ∂y.∂ξ[i] = (yp - ym) / (2ϵ)
    
        # compute dydeta
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]-ϵ
        ym     = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]+ϵ
        yp     = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        # --------
        ∂y.∂η[i] = (yp - ym) / (2ϵ)
    end
    # #################
    # # ForwardDiff
    # g = zeros(2)
    # Y = zeros(1)
    # dydksi_FD = zeros(size(dydeta))
    # dydeta_FD = zeros(size(dydeta))
    # dxdksi_FD = zeros(size(dydeta))
    # dxdeta_FD = zeros(size(dydeta))
    # @time for i in eachindex(dydeta_FD)
    #     X_msh[1] = x_ini[i]
    #     X_msh[2] = y_ini[i]
    #     Mesh_y_closed = (X_msh) -> Mesh_y( X_msh, Amp, x0, σ, b, m, ymin )
    #     ForwardDiff.gradient!( g, Mesh_y_closed, X_msh )
    #     dydksi_FD[i] = g[1]
    #     dydeta_FD[i] = g[2]
    #     Meshx_surf_closed = (X_msh) -> Mesh_x( X_msh, Amp, x0, σ, b, m, ymin )
    #     ForwardDiff.gradient!( g, Meshx_surf_closed, X_msh )
    #     dxdksi_FD[i] = g[1]
    #     dxdeta_FD[i] = g[2]
    # end
    
    # dxdksi_num = diff(xv4,dims=1)/(Δx/2)
    # dxdeta_num = diff(xv4,dims=2)/(Δy/2)
    # dydksi_num = diff(yv4,dims=1)/(Δx/2)
    # dydeta_num = diff(yv4,dims=2)/(Δy/2)
    
    # @printf("min(dxdksi    ) = %1.6f --- max(dxdksi    ) = %1.6f\n", minimum(dxdksi   ), maximum(dxdksi   ))
    # @printf("min(dxdksi_FD ) = %1.6f --- max(dxdksi_FD ) = %1.6f\n", minimum(dxdksi_FD), maximum(dxdksi_FD))
    # @printf("min(dxdksi_num) = %1.6f --- max(dxdksi_num) = %1.6f\n", minimum(dxdksi_num), maximum(dxdksi_num))
    
    # @printf("min(dxdeta    ) = %1.6f --- max(dxdeta   ) = %1.6f\n", minimum(dxdeta   ), maximum(dxdeta   ))
    # @printf("min(dxdeta_FD ) = %1.6f --- max(dxdeta_FD) = %1.6f\n", minimum(dxdeta_FD), maximum(dxdeta_FD))
    # @printf("min(dxdeta_num) = %1.6f --- max(dxdeta_num) = %1.6f\n", minimum(dxdeta_num), maximum(dxdeta_num))
    
    # @printf("min(dydksi    ) = %1.6f --- max(dydksi    ) = %1.6f\n", minimum(dydksi   ), maximum(dydksi   ))
    # @printf("min(dydksi_FD ) = %1.6f --- max(dydksi_FD ) = %1.6f\n", minimum(dydksi_FD), maximum(dydksi_FD))
    # @printf("min(dydksi_num) = %1.6f --- max(dydksi_num) = %1.6f\n", minimum(dydksi_num), maximum(dydksi_num))
    
    # @printf("min(dydeta    ) = %1.6f --- max(dydeta    ) = %1.6f\n", minimum(dydeta   ), maximum(dydeta   ))
    # @printf("min(dydeta_FD ) = %1.6f --- max(dydeta_FD ) = %1.6f\n", minimum(dydeta_FD), maximum(dydeta_FD))
    # @printf("min(dydeta_num) = %1.6f --- max(dydeta_num) = %1.6f\n", minimum(dydeta_num), maximum(dydeta_num))
    return nothing
end

##########################################
    
function InverseJacobian!(∂ξ,∂η,∂x,∂y)
    M = zeros(2,2)
    @time for i in eachindex(∂ξ.∂x)
        M[1,1]   = ∂x.∂ξ[i]
        M[1,2]   = ∂x.∂η[i]
        M[2,1]   = ∂y.∂ξ[i]
        M[2,2]   = ∂y.∂η[i]
        invJ     = inv(M)
        ∂ξ.∂x[i] = invJ[1,1]
        ∂ξ.∂y[i] = invJ[1,2]
        ∂η.∂x[i] = invJ[2,1]
        ∂η.∂y[i] = invJ[2,2]
    end
    @printf("min(∂ξ∂x) = %1.6f --- max(∂ξ∂x) = %1.6f\n", minimum(∂ξ.∂x), maximum(∂ξ.∂x))
    @printf("min(∂ξ∂y) = %1.6f --- max(∂ξ∂y) = %1.6f\n", minimum(∂ξ.∂y), maximum(∂ξ.∂y))
    @printf("min(∂η∂x) = %1.6f --- max(∂η∂x) = %1.6f\n", minimum(∂η.∂x), maximum(∂η.∂x))
    @printf("min(∂η∂y) = %1.6f --- max(∂η∂y) = %1.6f\n", minimum(∂η.∂y), maximum(∂η.∂y))
    return nothing
end