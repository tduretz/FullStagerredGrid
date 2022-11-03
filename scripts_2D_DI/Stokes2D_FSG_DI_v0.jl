using Plots, Printf, LinearAlgebra, SpecialFunctions
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy

# Physics
x          = (min=-3.0, max=3.0)  
y          = (min=-5.0, max=0.0)
ε̇bg        = -1.0
rad        = 0.5
y0         = -2.
g          = -1
inclusion  = true
adapt_mesh = true
solve      = true
# Numerics
nc         = (x=21,     y=21    )  # numerical grid resolution
nv         = (x=nc.x+1, y=nc.y+1)  # numerical grid resolution
ε          = 1e-8          # nonlinear tolerance
iterMax    = 20            # max number of iters
# Preprocessing
Δ          = (x=(x.max-x.min)/nc.x, y=(y.max-y.min)/nc.y)
# Array initialisation
V        = ( x   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)),  # v: vertices --- c: centroids
             y   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) ) 
ε̇        = ( xx  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),  # ex: x edges --- ey: y edges
             yy  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             xy  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)) )
τ        = ( xx  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),  # ex: x edges --- ey: y edges
             yy  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             xy  = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)) )
∇v̇       = (        ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)  )
P        = (        ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)  )
D        = ( v11 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v12 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),  
             v13 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v21 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v22 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v23 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v31 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v32 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)),
             v33 = (ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)), )
∂ξ       = ( ∂x  = (v  =  ones(nv.x,   nv.y), c  =  ones(nc.x+2, nc.y+2), ey =  ones(nc.x+2, nv.y), ex =  ones(nv.x,   nc.y+2)), 
             ∂y  = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2), ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)) )
∂η       = ( ∂x  = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2), ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)), 
             ∂y  = (v  =  ones(nv.x,   nv.y), c  =  ones(nc.x+2, nc.y+2), ey =  ones(nc.x+2, nv.y), ex =  ones(nv.x,   nc.y+2)) )
             
R        = ( x   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)), 
             y   = (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) ) 
ρ        = ( (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) )
η        = (        ey = zeros(nc.x+2, nv.y), ex = zeros(nv.x,   nc.y+2)  )
# Fine mesh
xxv, yyv    = LinRange(x.min-Δ.x/2, x.max+Δ.x/2, 2nc.x+3), LinRange(y.min-Δ.y/2, y.max+Δ.y/2, 2nc.y+3)
(xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
x = merge(x, (v=xv4[2:2:end-1,2:2:end-1  ], c=xv4[1:2:end-0,1:2:end-0  ], ey=xv4[1:2:end,2:2:end-1  ], ex=xv4[2:2:end-1,1:2:end]) ) 
y = merge(x, (v=yv4[2:2:end-1,2:2:end-1  ], c=yv4[1:2:end-0,1:2:end-0  ], ey=yv4[1:2:end,2:2:end-1  ], ex=yv4[2:2:end-1,1:2:end]) ) 
# Velocity
V.x.v .= -ε̇bg.*x.v; V.x.c .= -ε̇bg.*x.c
V.y.v .=  ε̇bg.*y.v; V.y.c .=  ε̇bg.*y.c
# Viscosity
η.ey  .= 1.0; η.ex  .= 1.0
if inclusion
    η.ey[x.ey.^2 .+ (y.ey.-y0).^2 .< rad] .= 100.
    η.ex[x.ex.^2 .+ (y.ex.-y0).^2 .< rad] .= 100.
end
D.v11.ey .= 2 .* η.ey; D.v11.ex .= 2 .* η.ex
D.v22.ey .= 2 .* η.ey; D.v22.ex .= 2 .* η.ex
D.v33.ey .= 2 .* η.ey; D.v33.ex .= 2 .* η.ex
# Density
ρ.v  .= 1.0; ρ.c  .= 1.0
if inclusion
    ρ.v[x.v.^2 .+ (y.v.-y0).^2 .< rad] .= 2.
    ρ.c[x.c.^2 .+ (y.c.-y0)^2 .< rad] .= 2.
end
# Residual
∇v̇.ey[2:end-1,:]   .=  ∂_∂x(V.x.v,V.x.c[2:end-1,:], Δ.x, Δ.y, ∂ξ.∂x.ey[2:end-1,:], ∂η.∂x.ey[2:end-1,:] ) .+ ∂_∂y(V.y.c[2:end-1,:],V.y.v, Δ.x, Δ.y, ∂ξ.∂y.ey[2:end-1,:], ∂η.∂y.ey[2:end-1,:] ) 
∇v̇.ex[:,2:end-1]   .=  ∂_∂x(V.x.c[:,2:end-1],V.x.v, Δ.x, Δ.y, ∂ξ.∂x.ex[:,2:end-1], ∂η.∂x.ex[:,2:end-1] ) .+ ∂_∂y(V.y.v,V.y.c[:,2:end-1], Δ.x, Δ.y, ∂ξ.∂y.ex[:,2:end-1], ∂η.∂y.ex[:,2:end-1] ) 
ε̇.xx.ey[2:end-1,:] .=  ∂_∂x(V.x.v,V.x.c[2:end-1,:], Δ.x, Δ.y, ∂ξ.∂x.ey[2:end-1,:], ∂η.∂x.ey[2:end-1,:] ) .- ∇v̇.ey[2:end-1,:]
ε̇.yy.ey[2:end-1,:] .=  ∂_∂y(V.y.c[2:end-1,:],V.y.v, Δ.x, Δ.y, ∂ξ.∂y.ey[2:end-1,:], ∂η.∂y.ey[2:end-1,:] ) .- ∇v̇.ey[2:end-1,:]
ε̇.xx.ex[:,2:end-1] .=  ∂_∂x(V.x.c[:,2:end-1],V.x.v, Δ.x, Δ.y, ∂ξ.∂x.ex[:,2:end-1], ∂η.∂x.ex[:,2:end-1] ) .- ∇v̇.ex[:,2:end-1]
ε̇.yy.ex[:,2:end-1] .=  ∂_∂y(V.y.v,V.y.c[:,2:end-1], Δ.x, Δ.y, ∂ξ.∂y.ex[:,2:end-1], ∂η.∂y.ex[:,2:end-1] ) .- ∇v̇.ex[:,2:end-1]
ε̇.xy.ey[2:end-1,:] .= (∂_∂y(V.x.c[2:end-1,:],V.x.v, Δ.x, Δ.y, ∂ξ.∂y.ey[2:end-1,:], ∂η.∂y.ey[2:end-1,:]) .+ ∂_∂x(V.y.v,V.y.c[2:end-1,:], Δ.x, Δ.y, ∂ξ.∂x.ey[2:end-1,:], ∂η.∂x.ey[2:end-1,:]) ) / 2.
ε̇.xy.ex[:,2:end-1] .= (∂_∂y(V.x.v,V.x.c[:,2:end-1], Δ.x, Δ.y, ∂ξ.∂y.ex[:,2:end-1], ∂η.∂y.ex[:,2:end-1]) .+ ∂_∂x(V.y.c[:,2:end-1],V.y.v, Δ.x, Δ.y, ∂ξ.∂x.ex[:,2:end-1], ∂η.∂x.ex[:,2:end-1]) ) / 2.
# τxx_1 .= 2.0 .* η_1 .* ε̇xx_1
# τxx_2 .= 2.0 .* η_2 .* ε̇xx_2
# τyy_1 .= 2.0 .* η_1 .* ε̇yy_1
# τyy_2 .= 2.0 .* η_2 .* ε̇yy_2
# τxy_1 .= 2.0 .* η_1 .* ε̇xy_1
# τxy_2 .= 2.0 .* η_2 .* ε̇xy_2
# Rx_1[2:end-1,2:end-0] .= ∂_∂x(τxx_1[:,2:end-0],τxx_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .+ ∂_∂y(τxy_2[2:end-1,:],τxy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .-  ∂_∂x(P_1[:,2:end-0],P_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0])
# Rx_2                  .= ∂_∂x(τxx_2[:,1:end-1],τxx_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .+ ∂_∂y(τxy_1,τxy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .-  ∂_∂x(P_2[:,1:end-1],P_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) 
# Ry_1[2:end-1,2:end-0] .= ∂_∂y(τyy_2[2:end-1,:],τyy_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ∂_∂x(τxy_1[:,2:end-0],τxy_2[2:end-1,:],Δx,Δy,∂ξ∂xv_1[2:end-1,2:end-0],∂η∂xv_1[2:end-1,2:end-0]) .-  ∂_∂y(P_2[2:end-1,:],P_1[:,2:end-0],Δx,Δy,∂ξ∂yv_1[2:end-1,2:end-0],∂η∂yv_1[2:end-1,2:end-0]) .+ ρ_1[2:end-1,2:end-0].*g
# Ry_2                  .= ∂_∂y(τyy_1,τyy_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ∂_∂x(τxy_2[:,1:end-1],τxy_1,           Δx,Δy,∂ξ∂xv_2[2:end-1,2:end-1],∂η∂xv_2[2:end-1,2:end-1]) .-  ∂_∂y(P_1,P_2[:,1:end-1],           Δx,Δy,∂ξ∂yv_2[2:end-1,2:end-1],∂η∂yv_2[2:end-1,2:end-1]) .+ ρ_2.*g
# Rp_1                  .= .-∇v_1
# Rp_2                  .= .-∇v_2[:,1:end-1]