using Plots, Printf, SparseArrays, LinearAlgebra

function UpdateCOO!(I,J,V, i,j,v, inz)
    I[inz] = i
    J[inz] = j
    V[inz] = v
    return inz   += 1 
end

function CreateSparseKuu(nnod, NumVx, NumVy, bct)
    # Construct velocity block
    I = ones(Int,3*2*(nnod))
    J = ones(Int,3*2*(nnod))
    V = zeros(3*2*(nnod))
    inz = 1
    # Vx
    for inode in 1:nnod
        iVxC = NumVx[inode] 
        # Connectivity
        if bct[inode] == 1 # Dirichlet
            inz = UpdateCOO!(I,J,V, iVxC,iVxC,inz, inz)
        else
            iVxS = NumVx[inode-1] 
            iVxN = NumVx[inode+1] 
            if bct[inode-1] == 0 # assumes symmetry
                inz = UpdateCOO!(I,J,V, iVxS,iVxC,inz, inz)
            end
            inz = UpdateCOO!(I,J,V, iVxC,iVxC,inz, inz)
            if bct[inode+1] == 0 # assumes symmetry
                inz = UpdateCOO!(I,J,V, iVxN,iVxC,inz, inz)
            end
        end
    end
    # Vy
    for inode in 1:nnod
        iVyC = NumVy[inode] 
        # Connectivity
        if bct[inode] == 1 # Dirichlet
            inz = UpdateCOO!(I,J,V, iVyC,iVyC,inz, inz)
        else
            iVyS = NumVy[inode-1] 
            iVyN = NumVy[inode+1] 
            if bct[inode-1] == 0 # assumes symmetry
                inz = UpdateCOO!(I,J,V, iVyS,iVyC,inz, inz)
            end
            inz = UpdateCOO!(I,J,V, iVyC,iVyC,inz, inz)
            if bct[inode+1] == 0 # assumes symmetry
                inz = UpdateCOO!(I,J,V, iVyN,iVyC,inz, inz)
            end
        end
    end
    # @printf("Number of dofs: %06d\n", inz)
    Kuu = sparse(I, J, V)
    return Kuu, nonzeros(Kuu)   
end

function UpdateSparseKuu!(Kuu_nz, dy, η, nnod, bct)
    # Construct velocity block
    inz = 1
    for inode in 1:nnod
        # Connectivity
        if bct[inode] == 1
            Kuu_nz[inz] = 1.0; inz+=1
        else
            eS   = η[inode-1]
            eN   = η[inode]
            cVxC = (1.0 * eN ./ dy + 1.0 * eS ./ dy) ./ dy
            cVxS = -1.0 * eS ./ dy .^ 2
            cVxN = -1.0 * eN ./ dy .^ 2
            if bct[inode-1] == 0
                Kuu_nz[inz] = cVxS; inz+=1
            end
            Kuu_nz[inz] = cVxC; inz+=1
            if bct[inode+1] == 0
                Kuu_nz[inz] = cVxN; inz+=1
            end
        end
    end
    for inode in 1:nnod
        # Connectivity
        if bct[inode] == 1
            Kuu_nz[inz] = 1.0; inz+=1
        else
            eS   = η[inode-1]
            eN   = η[inode]
            cVyC = ((4 // 3) * eN ./ dy + (4 // 3) * eS ./ dy) ./ dy
            cVyS = -4 // 3 * eS ./ dy .^ 2
            cVyN = -4 // 3 * eN ./ dy .^ 2
            if bct[inode-1] == 0
                Kuu_nz[inz] = cVyS; inz+=1
            end
            Kuu_nz[inz] = cVyC; inz+=1
            if bct[inode+1] == 0
                Kuu_nz[inz] = cVyN; inz+=1
            end
        end
    end
    # @printf("Number of dofs: %06d\n", inz)
    return nothing    
end

function UpdateSparseKuuDummy!(Kuu_nz, nnod, bct)
    # Construct velocity block
    inz = 1
    # Vx
    for inode in 1:nnod
        # Connectivity
        if bct[inode] == 1
            Kuu_nz[num_csc[inz]] = inz; inz+=1
        else
            if bct[inode-1] == 0
                Kuu_nz[num_csc[inz]] = inz; inz+=1
            end
            Kuu_nz[num_csc[inz]] = inz; inz+=1
            if bct[inode+1] == 0
                Kuu_nz[num_csc[inz]] = inz; inz+=1
            end
        end
    end
    # Vy
    for inode in 1:nnod
        # Connectivity
        if bct[inode] == 1
            Kuu_nz[num_csc[inz]] = inz; inz+=1
        else
            if bct[inode-1] == 0
                Kuu_nz[num_csc[inz]] = inz; inz+=1
            end
            Kuu_nz[num_csc[inz]] = inz; inz+=1
            if bct[inode+1] == 0
                Kuu_nz[num_csc[inz]] = inz; inz+=1
            end
        end
    end
    @printf("Number of dofs: %06d\n", inz)
    return nothing    
end

function Main_1D()
    ymin = -1.0
    ymax =  1.0
    ε̇bg  =  1.0

    ncy  = 100
    Δy   = (ymax-ymin)/ncy
    yv   = LinRange(     ymin,      ymax, ncy+1)
    yc   = LinRange(ymin+Δy/2, ymax-Δy/2, ncy  )
    dy = Δy

    NumVx = (1:1:(ncy+1))
    NumVy = (1:1:(ncy+1)) .+ maximum(NumVx)
    NumPt = 1:ncy
    η     = ones(ncy)

    Vx    = yv.*ε̇bg
    Vy    = 0.0.*yv
    Pt    = 0.0.*yc

    bct   = zeros(Int,ncy+1)
    bct[1] = 1; bct[end] = 1
    @time Kuu, Kuu_nz = CreateSparseKuu(ncy+1, NumVx, NumVy, bct)
    @time UpdateSparseKuu!(Kuu_nz, dy, η, ncy+1, bct)

    # nonzeros(Kuu)
    # @show Kuu[1,:]
    # @show Kuu[1,:]
    # @show Kuu[2,:]
    # @show Kuu[3,:]
    # @show Kuu[4,:]
    # display(Kuu0)
    # display(Kuu)
   
    @time Kc  = cholesky(Kuu; check = false)
    @time cholesky!(Kc,Kuu; check = false)
    # # spy(Kuu1)
    return Kuu
end 

# for i=1:3
Main_1D()
# end

