using Plots, Printf, LinearAlgebra, SpecialFunctions
using ExtendableSparse
include("FSG_Rheology.jl")
# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
@views    ∂_∂(fE,fW,fN,fS,Δ,a,b) = a*(fE - fW) / Δ.x .+ b*(fN - fS) / Δ.y
@views   function AddToExtSparse!(K,i,j,Tag, v) 
    if ((j!=-1) || (j==i || Tag==1)) K[i,j]  = v end
end

@views function AssembleKuuKupKpu!(Kuu, Kup, Kpu, Num, BC, nc, nv)
    i_uu    = zeros(Int, 18); t_uu    = zeros(Int, 18); v_uu   = ones(18)
    i_up    = zeros(Int,  4); t_up    = zeros(Int,  4); v_up   = ones( 4)
    i_pu    = zeros(Int,  8); t_pu    = zeros(Int,  8); v_pu   = ones( 8)
    # Loop on vertices
    for j in 2:nv.y+1, i in 2:nv.x+1
        i_uu[1]  = Num.x.v[i,j];     t_uu[1]  = BC.x.v[i,j]      # C
        i_uu[2]  = Num.x.v[i-1,j];   t_uu[2]  = BC.x.v[i-1,j]    # W
        i_uu[3]  = Num.x.v[i+1,j];   t_uu[3]  = BC.x.v[i+1,j]    # E
        i_uu[4]  = Num.x.v[i,j-1];   t_uu[4]  = BC.x.v[i,j-1]    # S
        i_uu[5]  = Num.x.v[i,j+1];   t_uu[5]  = BC.x.v[i,j+1]    # N
        i_uu[6]  = Num.x.c[i-1,j-1]; t_uu[6]  = BC.x.c[i-1,j-1]  # SW
        i_uu[7]  = Num.x.c[i,j-1];   t_uu[7]  = BC.x.c[i,j-1]    # SE
        i_uu[8]  = Num.x.c[i-1,j];   t_uu[8]  = BC.x.c[i-1,j]    # NW
        i_uu[9]  = Num.x.c[i,j];     t_uu[9]  = BC.x.c[i,j]      # NE
        #--------------
        i_uu[10] = Num.y.v[i,j];     t_uu[10] = BC.y.v[i,j]      # C
        i_uu[11] = Num.y.v[i-1,j];   t_uu[11] = BC.y.v[i-1,j]    # W
        i_uu[12] = Num.y.v[i,j];     t_uu[12] = BC.y.v[i,j]      # E
        i_uu[13] = Num.y.v[i,j-1];   t_uu[13] = BC.y.v[i,j-1]    # S
        i_uu[14] = Num.y.v[i,j+1];   t_uu[14] = BC.y.v[i,j+1]    # N
        i_uu[15] = Num.y.c[i-1,j-1]; t_uu[15] = BC.y.c[i-1,j-1]  # SW
        i_uu[16] = Num.y.c[i,j-1];   t_uu[16] = BC.y.c[i,j-1]    # SE
        i_uu[17] = Num.y.c[i-1,j];   t_uu[17] = BC.y.c[i-1,j]    # NW
        i_uu[18] = Num.y.c[i,j];     t_uu[18] = BC.y.c[i,j]      # NE
        #--------------
        i_up[1]  = Num.p.ex[i-1,j];  t_up[1]  = BC.p.ex[i-1,j]   # W
        i_up[2]  = Num.p.ex[i,j-1];  t_up[2]  = BC.p.ex[i,j-1]   # E
        i_up[3]  = Num.p.ey[i,j-1];  t_up[3]  = BC.p.ey[i,j-1]   # S   
        i_up[4]  = Num.p.ey[i,j];    t_up[4]  = BC.p.ey[i,j]     # N
        # Vx
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[1], i_uu[1], t_uu[1], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[1], i_uu[ii],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[1], i_up[ii],  t_up[ii],  v_up[ii])
            end
        end
        # Vx
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[10], i_uu[1], t_uu[1], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[10], i_uu[ii],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[10], i_up[ii],  t_up[ii],  v_up[ii])
            end
        end
    end 
    # Loop on centroids
    for j in 2:nc.y+1, i in 2:nc.x+1
        i_uu[1]  = Num.x.c[i,j];     t_uu[1]  = BC.x.c[i,j]      # C
        i_uu[2]  = Num.x.c[i-1,j];   t_uu[2]  = BC.x.c[i-1,j]    # W
        i_uu[3]  = Num.x.c[i+1,j];   t_uu[3]  = BC.x.c[i+1,j]    # E
        i_uu[4]  = Num.x.c[i,j-1];   t_uu[4]  = BC.x.c[i,j-1]    # S
        i_uu[5]  = Num.x.c[i,j+1];   t_uu[5]  = BC.x.c[i,j+1]    # N
        i_uu[6]  = Num.x.v[i,j];     t_uu[6]  = BC.x.v[i,j]  # SW
        i_uu[7]  = Num.x.v[i+1,j];   t_uu[7]  = BC.x.v[i+1,j]    # SE
        i_uu[8]  = Num.x.v[i,j+1];   t_uu[8]  = BC.x.v[i,j+1]    # NW
        i_uu[9]  = Num.x.v[i+1,j+1]; t_uu[9]  = BC.x.v[i+1,j+1]      # NE
        #--------------
        i_uu[10] = Num.y.c[i,j];     t_uu[10] = BC.y.c[i,j]      # C
        i_uu[11] = Num.y.c[i-1,j];   t_uu[11] = BC.y.c[i-1,j]    # W
        i_uu[12] = Num.y.c[i,j];     t_uu[12] = BC.y.c[i,j]      # E
        i_uu[13] = Num.y.c[i,j-1];   t_uu[13] = BC.y.c[i,j-1]    # S
        i_uu[14] = Num.y.c[i,j+1];   t_uu[14] = BC.y.c[i,j+1]    # N
        i_uu[15] = Num.y.v[i,j];     t_uu[15] = BC.y.v[i,j]  # SW
        i_uu[16] = Num.y.v[i+1,j];   t_uu[16] = BC.y.v[i+1,j]    # SE
        i_uu[17] = Num.y.v[i,j+1];   t_uu[17] = BC.y.v[i,j+1]    # NW
        i_uu[18] = Num.y.v[i+1,j+1]; t_uu[18] = BC.y.v[i+1,j+1]      # NE
        #--------------
        i_up[1]  = Num.p.ey[i,j];    t_up[1]  = BC.p.ey[i,j]   # W
        i_up[2]  = Num.p.ey[i+1,j];  t_up[2]  = BC.p.ey[i+1,j]     # E
        i_up[3]  = Num.p.ex[i,j];    t_up[3]  = BC.p.ex[i,j]   # S   
        i_up[4]  = Num.p.ex[i,j+1];  t_up[4]  = BC.p.ex[i,j+1]     # N
        # Vx
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[1], i_uu[1], t_uu[1], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[1], i_uu[ii],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[1], i_up[ii],  t_up[ii],  v_up[ii])
            end
        end
        # Vx
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[10], i_uu[1], t_uu[1], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[10], i_uu[ii],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[10], i_up[ii],  t_up[ii],  v_up[ii])
            end
        end
    end 

    # Loop on horizontal edges
    for j in 2:nv.y+1, i in 2:nc.x+1
        @show i, j
        @show i_pp    =  Num.p.ex[i,j]
        #--------------
        i_pu[1] =  Num.x.v[i,j];   t_pu[1] =  BC.x.v[i,j]   
        i_pu[2] =  Num.x.v[i+1,j]; t_pu[2] =  BC.x.v[i+1,j] 
        i_pu[3] =  Num.x.c[i,j-1]; t_pu[3] =  BC.x.c[i,j-1] 
        i_pu[4] =  Num.x.c[i,j];   t_pu[4] =  BC.x.c[i,j]   
        #--------------
        i_pu[5] =  Num.y.v[i,j];   t_pu[5] =  BC.y.v[i,j]   
        i_pu[6] =  Num.y.v[i+1,j]; t_pu[6] =  BC.y.v[i+1,j] 
        i_pu[7] =  Num.y.c[i,j-1]; t_pu[7] =  BC.y.c[i,j-1] 
        i_pu[8] =  Num.y.c[i,j];   t_pu[8] =  BC.y.c[i,j]  
        for ii in eachindex(v_pu)
            AddToExtSparse!(Kpu, i_pp, i_pu[ii],  t_pu[ii],  v_pu[ii])
        end
    end
    @show size(Num.p.ex)


    # Loop on vertical edges
    for j in 2:nc.y+1, i in 2:nv.x+1
        i_pp    =  Num.p.ey[i,j]
        #--------------
        i_pu[1] =  Num.x.c[i-1,j]; t_pu[1] =  BC.x.c[i-1,j]   
        i_pu[2] =  Num.x.c[i,j];   t_pu[2] =  BC.x.c[i,j] 
        i_pu[3] =  Num.x.v[i,j];   t_pu[3] =  BC.x.v[i,j] 
        i_pu[4] =  Num.x.v[i,j+1]; t_pu[4] =  BC.x.v[i,j+1]   
        #--------------
        i_pu[5] =  Num.y.c[i-1,j]; t_pu[5] =  BC.y.c[i-1,j]   
        i_pu[6] =  Num.y.c[i,j];   t_pu[6] =  BC.y.c[i,j] 
        i_pu[7] =  Num.y.v[i,j];   t_pu[7] =  BC.y.v[i,j] 
        i_pu[8] =  Num.y.v[i,j+1]; t_pu[8] =  BC.y.v[i,j+1]
        for ii in eachindex(v_pu)
            AddToExtSparse!(Kpu, i_pp, i_pu[ii],  t_pu[ii],  v_pu[ii])
        end
    end
    flush!(Kuu), flush!(Kup), flush!(Kpu)
    end

function Main_2D_DI()
    # Physics
    # x          = (min=-3.0, max=3.0)  
    # y          = (min=-5.0, max=0.0)
    x          = (min=-3.0, max=3.0)  
    y          = (min=-3.0, max=3.0)
    ε̇bg        = -1.0
    rad        = 0.5
    y0         = -2*0.
    g          = (x = 0., z=-1.0*0.0)
    inclusion  = true
    adapt_mesh = true
    solve      = true
    # Numerics
    nc         = (x=10,     y=13    )  # numerical grid resolution
    nv         = (x=nc.x+1, y=nc.y+1)  # numerical grid resolution
    ε          = 1e-8          # nonlinear tolerance
    iterMax    = 20            # max number of iters
    # Preprocessing
    Δ          = (x=(x.max-x.min)/nc.x, y=(y.max-y.min)/nc.y)
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
    Dv       = ( v11 = (ex = zeros(nc.x+2, nv.y+2), ey = zeros(nv.x+2,   nc.y+2)),
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
    ρ        = ( (v  = zeros(nv.x,   nv.y), c  = zeros(nc.x+2, nc.y+2)) )
    η        = (        ex = zeros(nc.x+2, nv.y), ey = zeros(nv.x,   nc.y+2)  )
    BC       = (x = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                y = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2) ),
                p = (ex = -1*ones(Int, nc.x+2, nv.y+2),     ey = -1*ones(Int, nv.x+2,   nc.y+2))  )
    Num      = ( x   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                 y   = (v  = -1*ones(Int, nv.x,   nv.y), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                 p   = (ex = -1*ones(Int, nc.x+2, nv.y), ey = -1*ones(Int, nv.x,   nc.y+2)) )
    # Fine mesh
    xxv, yyv    = LinRange(x.min-Δ.x/2, x.max+Δ.x/2, 2nc.x+3), LinRange(y.min-Δ.y/2, y.max+Δ.y/2, 2nc.y+3)
    (xv4,yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    xv2_1, yv2_1 = xv4[2:2:end-1,2:2:end-1  ], yv4[2:2:end-1,2:2:end-1  ]
    xv2_2, yv2_2 = xv4[1:2:end-0,1:2:end-0  ], yv4[1:2:end-0,1:2:end-0  ]
    xc2_1, yc2_1 = xv4[3:2:end-2,2:2:end-1  ], yv4[3:2:end-2,2:2:end-1  ]
    xc2_2, yc2_2 = xv4[2:2:end-1,3:2:end-2+2], yv4[2:2:end-1,3:2:end-2+2]
    x = merge(x, (v=xv4[2:2:end-1,2:2:end-1  ], c=xv4[1:2:end-0,1:2:end-0  ], ex=xv4[1:2:end,2:2:end-1  ], ey=xv4[2:2:end-1,1:2:end]) ) 
    y = merge(x, (v=yv4[2:2:end-1,2:2:end-1  ], c=yv4[1:2:end-0,1:2:end-0  ], ex=yv4[1:2:end,2:2:end-1  ], ey=yv4[2:2:end-1,1:2:end]) ) 
    # Velocity
    V.x.v[2:end-1,2:end-1] .= -ε̇bg.*x.v; V.x.c .= -ε̇bg.*x.c
    V.y.v[2:end-1,2:end-1] .=  ε̇bg.*y.v; V.y.c .=  ε̇bg.*y.c
    # V.x.v[2:end-1,2:end-1] .=  ε̇bg.*y.v; V.x.c .=  ε̇bg.*y.c
    # V.y.v[2:end-1,2:end-1] .=  ε̇bg.*x.v; V.y.c .=  ε̇bg.*x.c
    # Viscosity
    η.ex  .= 1.0; η.ey  .= 1.0
    if inclusion
        η.ex[x.ex.^2 .+ (y.ex.-y0).^2 .< rad] .= 100.
        η.ey[x.ey.^2 .+ (y.ey.-y0).^2 .< rad] .= 100.
    end
    Dv.v11.ex[:,2:end-1] .= 2 .* η.ex; Dv.v11.ey[2:end-1,:] .= 2 .* η.ey
    Dv.v22.ex[:,2:end-1] .= 2 .* η.ex; Dv.v22.ey[2:end-1,:] .= 2 .* η.ey
    Dv.v33.ex[:,2:end-1] .= 2 .* η.ex; Dv.v33.ey[2:end-1,:] .= 2 .* η.ey
    # Density
    ρ.v  .= 1.0; ρ.c  .= 1.0
    if inclusion
        ρ.v[x.v.^2 .+ (y.v.-y0).^2 .< rad] .= 2.
        ρ.c[x.c.^2 .+ (y.c.-y0).^2 .< rad] .= 2.
    end
    # Boundary conditions
    BCVx = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    BCVy = (West=:Dirichlet, East=:Dirichlet, South=:Dirichlet, North=:Dirichlet)
    BC.x.v[2:end-1,2:end-1] .= 0 # inner points
    BC.y.v[2:end-1,2:end-1] .= 0 # inner points
    BC.x.c[2:end-1,2:end-1] .= 0 # inner points
    BC.y.c[2:end-1,2:end-1] .= 0 # inner points
    BC.p.ex[2:end-1,2:end-1] .= 0 # inner points
    BC.p.ey[2:end-1,2:end-1] .= 0 # inner points
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
    end

    DevStrainRateStressTensor!( ε̇, τ, Dv, ∇v, V, Δ, ∂ξ, ∂η )
    # Loop on vertices
    @time for j in axes(R.x.v, 2), i in axes(R.x.v, 1)
        if i>1 && i<size(R.x.v,1) && j>1 && j<size(R.x.v,2)
            # Stencil
            τxxW = τ.xx.ex[i-1,j]; τyyW = τ.yy.ex[i-1,j]; τxyW = τ.xy.ex[i-1,j]; PW   = P.ex[i-1,j]; 
            τxxE = τ.xx.ex[i,j];   τyyE = τ.yy.ex[i,j];   τxyE = τ.xy.ex[i,j];   PE   = P.ex[i,j];   
            τxxS = τ.xx.ey[i,j-1]; τyyS = τ.yy.ey[i,j-1]; τxyS = τ.xy.ey[i,j-1]; PS   = P.ey[i,j-1]; 
            τxxN = τ.xx.ey[i,j];   τyyN = τ.yy.ey[i,j];   τxyN = τ.xy.ey[i,j];   PN   = P.ey[i,j];      
            ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.v[i,j];   ∂η∂y = ∂η.∂y.v[i,j]
            # x
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                R.x.v[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x )
            end
            # y
            if BC.x.v[i,j] == 1
                R.x.v[i,j] = 0.
            else
                R.y.v[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z) 
            end
        end
    end 
    # Loop on centroids
    @time for j in axes(R.x.c, 2), i in axes(R.x.c, 1)
        if i>1 && i<size(R.x.c,1) && j>1 && j<size(R.x.c,2)
            # Stencil
            τxxW = τ.xx.ey[i,j];   τyyW = τ.yy.ey[i,j];   τxyW = τ.xy.ey[i,j];   PW   = P.ey[i,j]
            τxxE = τ.xx.ey[i+1,j]; τyyE = τ.yy.ey[i+1,j]; τxyE = τ.xy.ey[i+1,j]; PE   = P.ey[i+1,j]
            τxxS = τ.xx.ex[i,j];   τyyS = τ.yy.ex[i,j];   τxyS = τ.xy.ex[i,j];   PS   = P.ex[i,j]
            τxxN = τ.xx.ex[i,j+1]; τyyN = τ.yy.ex[i,j+1]; τxyN = τ.xy.ex[i,j+1]; PN   = P.ex[i,j+1]
            ∂ξ∂x = ∂ξ.∂x.v[i,j];   ∂ξ∂y = ∂ξ.∂y.v[i,j];   ∂η∂x = ∂η.∂x.c[i,j];   ∂η∂y = ∂η.∂y.c[i,j]
            # x
            R.x.c[i,j] = - (∂_∂(τxxE,τxxW,τxxN,τxxS,Δ,∂ξ∂x,∂η∂x) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂y,∂η∂y) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂x,∂η∂x) + ρ.v[i,j]*g.x)
            # y
            R.y.c[i,j] = - (∂_∂(τyyE,τyyW,τyyN,τyyS,Δ,∂ξ∂y,∂η∂y) + ∂_∂(τxyE,τxyW,τxyN,τxyS,Δ,∂ξ∂x,∂η∂x) - ∂_∂(PE,PW,PN,PS,Δ,∂ξ∂y,∂η∂y) + ρ.v[i,j]*g.z) 
        end
    end 
    # Loop on horizontal edges
    @time for j in axes(R.p.ex, 2), i in axes(R.p.ex, 1)
        if i>1 && i<size(R.p.ex, 1)
            R.p.ex[i,j] = -∇v.ex[i,j]
        end
    end
    # Loop on horizontal edges
    @time for j in axes(R.p.ey, 2), i in axes(R.p.ey, 1)
        if j>1 && j<size(R.p.ey, 2)
            R.p.ey[i,j] = -∇v.ey[i,j]
        end
    end
    # Numbering
    Num      = ( x   = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)), 
                    y   = (v  = -1*ones(Int, nv.x+2,   nv.y+2), c  = -1*ones(Int, nc.x+2, nc.y+2)),
                    p   = (ex = -1*ones(Int, nc.x+2, nv.y+2), ey = -1*ones(Int, nv.x+2,   nc.y+2)) )

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
    @time AssembleKuuKupKpu!(Kuu, Kup, Kpu, Num, BC, nc, nv)
    println("Touch Kuu and Kup")
    @time AssembleKuuKupKpu!(Kuu, Kup, Kpu, Num, BC, nc, nv)
    # τ.xy.ey'
    # ε̇.xy.ey'
    # τ.xx.ey'
    # R.y.c'
    Kpu
    # Num.p.ex
end

Main_2D_DI()