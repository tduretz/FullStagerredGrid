using Plots, Printf, SparseArrays, LinearAlgebra, ForwardDiff
using ExtendableSparse

function Mesh_y(y_ini, σy, ymin0, ymax0, y0)
    ymin1 = (sinh.( σy.*(ymin0.-y0) ))
    ymax1 = (sinh.( σy.*(ymax0.-y0) ))
    sy    = (ymax0-ymin0)/(ymax1-ymin1)
    y     = (sinh.( σy.*(y_ini.-y0) )) .* sy  .+ y0
    return y
end

function CreateExtSparseK(nnod, Num, bct)
    K = ExtendableSparseMatrix(nnod, nnod)
    inz = 1
    for inode in 1:nnod # Loop on equations
        iC = Num[inode] 
        if bct[inode] == 1 # BC Dirichlet
            K[iC,iC] = inz; inz+=1
        else
            iS = Num[inode-1] 
            iN = Num[inode+1] 
            if bct[inode-1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                K[iC,iS] = inz; inz+=1
            end
            K[iC,iC] = inz; inz+=1
            if bct[inode+1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                K[iC,iN] = inz; inz+=1
            end
        end
    end 
    flush!(K)
    return K  
end

function UpdateExtSparseK!(K, Num, dy, η, ∂η∂y, nnod, bct)
    # Update non-zeros, K_nz, in CSC format
    # K.nzval .= 0.0
    for inode in 1:nnod  # Loop on equations
        iC = Num[inode] 
        if bct[inode] == 1
            # updateindex!(K,+,1.0,iC,iC) # accumulate, e.g. FEM
            setindex!(K,1.0,iC,iC)
        else
            iS = Num[inode-1] 
            iN = Num[inode+1] 
            eS = η[inode-1]
            eN = η[inode]
            dC = ∂η∂y.v[inode]
            dS = ∂η∂y.c[inode-1]
            dN = ∂η∂y.c[inode]
            cC = -dC .* (-1.0 * dN .* eN ./ dy - 1.0 * dS .* eS ./ dy) ./ dy
            cS = -1.0 * dC .* dS .* eS ./ dy .^ 2
            cN = -1.0 * dC .* dN .* eN ./ dy .^ 2
            coeff = 1.0/dC # make matrix symmetrix despite variable spacing !! 
            cC *= coeff
            cS *= coeff
            cN *= coeff
            if (iC==2 && iN==3 )
                println((cN, dC, dS, dN))
            end
            if (iC==3 && iS==2 )
                println((cS, dC, dS, dN))
            end
            if bct[inode-1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                # updateindex!(K,+,cS,iC,iS)
                setindex!(K,cS,iC,iS)
            end
            # updateindex!(K,+,cC,iC,iC)
            setindex!(K,cC,iC,iC)
            if bct[inode+1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                # updateindex!(K,+,cN,iC,iN)
                setindex!(K,cN,iC,iN)
            end
        end
    end
    return nothing    
end

function Main_1D_Poisson()
    swiss_x  = true
    L        = 1.0              # domain size
    # ncy      = 10000000           # number of cells, ncy+1=number of nodes
    ncy      = 10           # number of cells, ncy+1=number of nodes
    Δy       = L/ncy            # spacing
    Num      = 1:1:(ncy+1)      # dof numbering
    η        = ones(ncy)        # coefficient
    bct      = zeros(Int,ncy+1) # array that identifies dof type (here Dirichlet:1, else: 0)
    bct[1]   = 1                # identify BC nodes as Dirichlets
    bct[end] = 1                # identify BC nodes as Dirichlets 
    ∂η∂y     = ( v=ones(ncy+1), c=ones(ncy))
    yv4      = Array(LinRange(-L/2,L/2,2*ncy+1))
    yv40     = copy(yv4)
    ∂y∂η4    = zeros(2*ncy+1)
    if swiss_x
        ymin0    = -L/2
        ymax0    =  L/2
        σy       = 5.
        y0       = 0.0
        ϵ        = 1e-7
        for i in eachindex(yv4)
            g        = 0.
            y_ini    = yv4[i]
            yv4[i]   = Mesh_y(y_ini, σy, ymin0, ymax0, y0)
            Mesh_y_closed = (Y) -> Mesh_y( Y, σy, ymin0, ymax0, y0 )
            ∂y∂η4[i] =  ForwardDiff.derivative( Mesh_y_closed, y_ini )
            ##
            # ymin1    = (sinh.( σy.*(ymin0.-y0) ))
            # ymax1    = (sinh.( σy.*(ymax0.-y0) ))
            # sy       = (ymax0-ymin0)/(ymax1-ymin1)
            # dm       = (sinh.( σy.*((y_ini-ϵ).-y0) )) .* sy  .+ y0
            # dp       = (sinh.( σy.*((y_ini+ϵ).-y0) )) .* sy  .+ y0
            # ∂y∂η4[i] = (dp-dm)/2/ϵ
        end
        ∂η∂y4   = 1.0 ./ ∂y∂η4
        ∂η∂y.c .= ∂η∂y4[2:2:end-1]
        ∂η∂y.v .= ∂η∂y4[1:2:end-0]
    end

    # println("Create initial extendable sparse matrix")
    # @time Kes = CreateExtSparseK(ncy+1, Num, bct)
    # # Set coefficient to the matrix
    # println("Update non-zeros of extendable sparse")
    # @time UpdateExtSparseK!(Kes, Num, Δy, η, ∂η∂y, ncy+1, bct)
    # # Initial solve
    # println("Initial Cholesky factorisation")
    # Kuu = Kes.cscmatrix

    # #-------------------------#
    # Initial field
    ε  = 1.
    Vx = zeros(ncy+1)
    Vx .= ε .* yv4[1:2:end]

    # That's incorrect 
    ∂Vx∂y = (∂η∂y.c) .* (Vx[2:end].-Vx[1:end-1])/Δy
    @show r   = ∂Vx∂y .- ε
   
    # That's correct 
    yv    = yv4[1:2:end]
    Δyc   = yv[2:end] .- yv[1:end-1]
    ∂Vx∂y = (Vx[2:end].-Vx[1:end-1])./Δyc
    @show r   = ∂Vx∂y .- ε

    # The Jacobian should be computed using the deformed mesh information 
    ∂η∂y  = Δy ./ Δyc
    ∂Vx∂y = ∂η∂y .* (Vx[2:end].-Vx[1:end-1])/Δy
    @show r = ∂Vx∂y .- ε


    # # Rhs (only BC)
    # b    = zeros(ncy+1)
    # b[1] = 1
    # # Initial residual
    # @show F = Kuu*Vx .- b
    # # Cholesky
    # @time Kuu_fact = cholesky(Hermitian(Kuu), check=false)
    # # Defect correction update
    # Vx .-= Kuu_fact\F
    # # Final residual
    # @show F = Kuu*Vx .- b
    display(plot(Vx, yv4[1:2:end]))
end 

for i=1:1
    println("run $i")
    Main_1D_Poisson()
end