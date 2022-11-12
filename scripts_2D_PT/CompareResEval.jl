using FullStaggeredGrid, Statistics
include("MeshDeformation.jl")

inclusion = true
adapt_mesh = false

@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
@views    ∂η∂xv(A)       = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views ∂η∂xv_xa(A)       =  0.5*(A[1:end-1,:].+A[2:end,:])
@views ∂η∂xv_ya(A)       =  0.5*(A[:,1:end-1].+A[:,2:end]) 
@views   ∂η∂xvh(A)       = ( 0.25./A[1:end-1,1:end-1] .+ 0.25./A[1:end-1,2:end-0] .+ 0.25./A[2:end-0,1:end-1] .+ 0.25./A[2:end-0,2:end-0]).^(-1)
@views   ∂η∂xvWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
Dat = Float64  # Precision (double=Float64 or single=Float32)

if inclusion
    file = matopen(string(@__DIR__,"/../scripts_2D_PT/output_FS_inc_rho.mat"))
else
    file = matopen(string(@__DIR__,"/../scripts_2D_PT/output_FS.mat"))
end
Vx_1 = read(file, "Vx_1") 
Vx_2 = read(file, "Vx_2")
Vy_1 = read(file, "Vy_1") 
Vy_2 = read(file, "Vy_2")
P_1  = read(file, "P_1" ) 
P_2  = read(file, "P_2" )

close(file)

############################################

# Physics
x          = (min=-3.0, max=3.0)  
y          = (min=-5.0, max=0.0)
xmin, xmax = -3.0, 3.0  
ymin, ymax = -5.0, 0.0
εbg      = -1*0
rad      = 0.5
y0       = -2.
g          = (x = 0., z=-1.0)

ncx, ncy = 21, 21    # numerical grid resolution
nc         = (x=21,    y=21   )  # numerical grid resolution
nv         = (x=nc.x+1, y=nc.y+1)  # numerical grid resolution
# Preprocessing
Δ          = (x=(x.max-x.min)/nc.x, y=(y.max-y.min)/nc.y, t=1.0)
  # Array initialisation
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
  η_1         =   zeros(Dat, ncx-0, ncy+1)
  τxx_2       =   zeros(Dat, ncx+1, ncy-0+1)
  τyy_2       =   zeros(Dat, ncx+1, ncy-0+1)
  τxy_2       =   zeros(Dat, ncx+1, ncy-0+1)
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
          xv4[i]   =  Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx )
          yv4[i]   =  Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
      end
      # Compute forward transformation
      params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, σx=σx, σy=σy, ϵ=ϵ)
      ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
      ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
      ComputeForwardTransformation!( ∂x, ∂y, x_ini, y_ini, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵ)
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

# Preprocessing
Δx, Δy  = (xmax-xmin)/ncx, (ymax-ymin)/ncy

# Viscosity
η_1  .= 1.0; η_2  .= 1.0
if inclusion
    # η_1[xc2_1.^2 .+ (yc2_1.-y0).^2 .< rad] .= 100.
    # η_2[xc2_2.^2 .+ (yc2_2.-y0).^2 .< rad] .= 100.
end
# Density
ρ_1  .= 1.0; ρ_2  .= 1.0
if inclusion
    ρ_1[xv2_1.^2 .+ (yv2_1.-y0).^2 .< rad] .= 2.
    ρ_2[xv2_2[2:end-1,2:end-1].^2 .+ (yv2_2[2:end-1,2:end-1].-y0).^2 .< rad] .= 2.
end

hx_surf =   hx[3:2:end-2, end]
        η_surf  = η_1[:,end]
        dx      = Δx
        dz      = Δy
        dkdx    = ∂ξ∂x[3:2:end-2, end]
        dkdy    = ∂ξ∂y[3:2:end-2, end]
        dedx    = ∂η∂x[3:2:end-2, end]
        dedy    = ∂η∂y[3:2:end-2, end]

        M1 = dedx.*hx_surf.^2 .- dedx .+ 2*dedy.*hx_surf
        M2 = 2*dedx.^2 .*hx_surf.^2 .+ dedx.^2 .+ 2*dedx.*dedy.*hx_surf .+ dedy.^2 .*hx_surf.^2 .+ 2*dedy.^2
        M3 = 2*dedx.*hx_surf.^2 .+ dedx .+ 2*dedy.*hx_surf
        M4 = hx_surf.^2 .+ 2
        M5 = 2*hx_surf.^2 .+ 1
        M6 = 2*dedx.*hx_surf - dedy.*hx_surf.^2 .+ dedy

  # !!!!!!!!! Change derivatives !!!!!
  dVxdx  = (Vx_1[2:end-0,end] - Vx_1[1:end-1,end])/Δx
  dVydx  = (Vy_1[2:end-0,end] - Vy_1[1:end-1,end])/Δx
  P_surf = P_1[:,end]
  # See python notebook v5
#   Vx_2[2:end-1,end] = Vx_2[2:end-1,end-1] - Δy*1/1*dVydx # dvxdy = -(dvydx)
#   Vy_2[2:end-1,end] = Vy_2[2:end-1,end-1] + Δy*1/2*dVxdx + 3/4 * Δy*P_surf./η_surf # dvydy = 1/2*dv(dvxdx) + 3/4 * P/eta
  Vx_2[2:end-1,end] = (3*M1.*P_surf.*dz/2 + M2.*η_surf.*Vx_2[2:end-1,end-1] - M3.*dkdx.*dVxdx.*dz.*η_surf - M4.*dedy.*dkdx.*dVydx.*dz.*η_surf + dedx.*dkdy.*dz.*η_surf.*hx_surf.^2 + 2*dedx.*dkdy.*dz.*η_surf - dedy.*dkdy.*dz.*η_surf.*hx_surf.^2 - 2*dedy.*dkdy.*dz.*η_surf)./(M2.*η_surf)
  Vy_2[2:end-1,end] = (M2.*η_surf.*Vy_2[2:end-1,end-1] - M5.*dedx.*dkdx.*dVydx.*dz.*η_surf + M5.*dedy.*dkdx.*dVxdx.*dz.*η_surf + 3*M6.*P_surf.*dz/2 - 2*dedx.*dkdy.*dz.*η_surf.*hx_surf.^2 - 2*dedx.*dkdy.*dz.*η_surf.*hx_surf - dedx.*dkdy.*dz.*η_surf - dedy.*dkdy.*dz.*η_surf.*hx_surf.^2 - 2*dedy.*dkdy.*dz.*η_surf)./(M2.*η_surf)
  ∇v_1            .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) .+ ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) 
  ∇v_2[:,1:end-1] .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) .+ ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) 
  ε̇xx_1 .=  ∂_∂x(Vx_1,Vx_2[2:end-1,:],Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) .- 1.0/3.0*∇v_1
  ε̇yy_1 .=  ∂_∂y(Vy_2[2:end-1,:],Vy_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) .- 1.0/3.0*∇v_1
  ε̇xy_1 .= (∂_∂y(Vx_2[2:end-1,:],Vx_1,Δx,Δy,∂ξ∂yc_1,∂η∂yc_1) .+ ∂_∂x(Vy_1,Vy_2[2:end-1,:], Δx,Δy,∂ξ∂xc_1,∂η∂xc_1) ) / 2.
  ε̇xx_2[:,1:end-1] .=  ∂_∂x(Vx_2[:,2:end-1],Vx_1,Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) .- 1.0/3.0*∇v_2[:,1:end-1]
  ε̇yy_2[:,1:end-1] .=  ∂_∂y(Vy_1,Vy_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) .- 1.0/3.0*∇v_2[:,1:end-1]
  ε̇xy_2[:,1:end-1] .= (∂_∂y(Vx_1,Vx_2[:,2:end-1],Δx,Δy,∂ξ∂yc_2,∂η∂yc_2) .+ ∂_∂x(Vy_2[:,2:end-1],Vy_1, Δx,Δy,∂ξ∂xc_2,∂η∂xc_2) ) / 2.
  τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
  τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
  τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
  τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
  τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
  τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
  Rx_1[2:end-1,2:end-0] .= ∂_∂x(τxx_1[:,2:end-0],τxx_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .+ ∂_∂y(τxy_2[2:end-1,:],τxy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .-  ∂_∂x(P_1[:,2:end-0],P_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0])
  Rx_2                  .= ∂_∂x(τxx_2[:,1:end-1],τxx_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .+ ∂_∂y(τxy_1,τxy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .-  ∂_∂x(P_2[:,1:end-1],P_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) 
  Ry_1[2:end-1,2:end-0] .= ∂_∂y(τyy_2[2:end-1,:],τyy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ∂_∂x(τxy_1[:,2:end-0],τxy_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .-  ∂_∂y(P_2[2:end-1,:],P_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ρ_1[2:end-1,2:end-0].*g.z
  Ry_2                  .= ∂_∂y(τyy_1,τyy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ∂_∂x(τxy_2[:,1:end-1],τxy_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .-  ∂_∂y(P_1,P_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ρ_2.*g.z
  Rp_1                  .= .-∇v_1
  Rp_2                  .= .-∇v_2[:,1:end-1]
  norm_Rx = 0.5*( norm(Rx_1)/sqrt(length(Rx_1)) + norm(Rx_2)/sqrt(length(Rx_2)) )
  norm_Ry = 0.5*( norm(Ry_1)/sqrt(length(Ry_1)) + norm(Ry_2)/sqrt(length(Ry_2)) )
  norm_Rp = 0.5*( norm(Rp_1)/sqrt(length(Rp_1)) + norm(Rp_2)/sqrt(length(Rp_2)) )
  @printf("nRx=%1.3e nRy=%1.3e nRp=%1.3e\n", norm_Rx, norm_Ry, norm_Rp)

###################################################################

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
ε̇ = (ex = -1*ones(Int, nc.x+2, nv.y+2),     ey = -1*ones(Int, nv.x+2,   nc.y+2)) )

# Fine mesh
xxv, yyv    = LinRange(x.min-Δ.x, x.max+Δ.x, 2nc.x+5), LinRange(y.min-Δ.y, y.max+Δ.y, 2nc.y+5)
(xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
xv2_1, yv2_1 = xv4[3:2:end-2,3:2:end-2  ], yv4[3:2:end-2,3:2:end-2  ]
xv2_2, yv2_2 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,1:2:end-1  ]
# xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
# xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
x = merge(x, (v=xv4[1:2:end-0,1:2:end-0], c=xv4[2:2:end-1,2:2:end-1], ex=xv4[2:2:end-1,1:2:end-0  ], ey=xv4[1:2:end-0,2:2:end-1]) ) 
y = merge(y, (v=yv4[1:2:end-0,1:2:end-0], c=yv4[2:2:end-1,2:2:end-1], ex=yv4[2:2:end-1,1:2:end-0  ], ey=yv4[1:2:end-0,2:2:end-1]) ) 
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
 # BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
 # BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
 BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)
 BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:FreeSurface)
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
 end
 # Pressure
 if BCVx.North == :FreeSurface || BCVy.North == :FreeSurface 
     BC.p.ex[2:end-1,end-1] .= 2 # modified equation
     BC.ε̇.ey[2:end-1,end  ] .= 2
 end

V.x.v[2:end-1,2:end-1] .= Vx_1
V.x.c .= Vx_2
V.y.v[2:end-1,2:end-1] .= Vy_1
V.y.c .= Vy_2
P.ex[2:end-1,2:end-1] .= P_1
P.ey[2:end-1,2:end-0] .= P_2

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

@show mean(ε̇.xx.ex[2:end-1,2:end-1] .- ε̇xx_1)/mean(ε̇xx_1)
@show mean(ε̇.xx.ey[2:end-1,2:end-0] .- ε̇xx_2)/mean(ε̇xx_2)
@show mean(ε̇.yy.ex[2:end-1,2:end-1] .- ε̇yy_1)/mean(ε̇yy_1)
@show mean(ε̇.yy.ey[2:end-1,2:end-0] .- ε̇yy_2)/mean(ε̇yy_2)
@show mean(ε̇.xy.ex[2:end-1,2:end-1] .- ε̇xy_1)/mean(ε̇xy_1)
@show mean(ε̇.xy.ey[2:end-1,2:end-0] .- ε̇xy_2)/mean(ε̇xy_2)

@show mean(τ.xx.ex[2:end-1,2:end-1] .- τxx_1)/mean(τxx_1)
@show mean(τ.xx.ey[2:end-1,2:end-0] .- τxx_2)/mean(τxx_2)
@show mean(τ.yy.ex[2:end-1,2:end-1] .- τyy_1)/mean(τyy_1)
@show mean(τ.yy.ey[2:end-1,2:end-0] .- τyy_2)/mean(τyy_2)
@show mean(τ.xy.ex[2:end-1,2:end-1] .- τxy_1)/mean(τxy_1)
@show mean(τ.xy.ey[2:end-1,2:end-0] .- τxy_2)/mean(τxy_2)

@show mean(P.ex[2:end-1,2:end-1] .- P_1)/mean(P_1)
@show mean(P.ey[2:end-1,2:end-0] .- P_2)/mean(P_2)


display( maximum( ρ.c[2:end-1,2:end-1] .- ρ_2) )
# @show ∇v_1.-∇v.ex[2:end-1,2:end-1]
# display(ε̇.yy.ex[2:end-1,2:end-1] .- ε̇yy_1)
# @show ε̇.xx.ex[2:end-1,2:end-1] .- ε̇xx_1
# BC.y.v
# display( ρ.c[2:end-1,2:end-1] )
# display(ρ_2)
# display(xv2_2[2:end-1,2:end-1] .- x.c[2:end-1,2:end-1] )


