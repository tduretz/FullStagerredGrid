using Plots, Printf, SparseArrays, LinearAlgebra

function NonZeroCOO!(I,J,V, i,j,v, inz)
    I[inz] = i
    J[inz] = j
    V[inz] = v
    return inz += 1 
end

function CreateSparseK(nnod, Num, bct)
    # Construct initial sparse matrix from triplets
    I   = ones(Int,3*nnod)
    J   = ones(Int,3*nnod)
    V   = zeros(3*nnod)
    inz = 1
    for inode in 1:nnod # Loop on equations
        iC = Num[inode] 
        if bct[inode] == 1 # BC Dirichlet
            inz = NonZeroCOO!(I,J,V, iC,iC,inz, inz)
        else
            iS = Num[inode-1] 
            iN = Num[inode+1] 
            if bct[inode-1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                inz = NonZeroCOO!(I,J,V, iS,iC,inz, inz) # transpose insertion because for CSC, brrr...
            end
            inz = UpdateCOO!(I,J,V, iC,iC,inz, inz)
            if bct[inode+1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                inz = NonZeroCOO!(I,J,V, iN,iC,inz, inz) # transpose insertion because for CSC, brrr...
            end
        end
    end    
    # Final assembly
    K = sparse(I, J, V)
    return K, nonzeros(K)   
end

function UpdateSparseK!(K_nz, dy, η, nnod, bct)
    # Update non-zeros, K_nz, in CSC format
    inz = 1
    for inode in 1:nnod  # Loop on equations
        if bct[inode] == 1
            K_nz[inz] = 1.0; inz+=1
        else
            eS = η[inode-1]
            eN = η[inode]
            cC = (1.0 * eN ./ dy + 1.0 * eS ./ dy) ./ dy
            cS = -1.0 * eS ./ dy .^ 2
            cN = -1.0 * eN ./ dy .^ 2
            if bct[inode-1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                K_nz[inz] = cS; inz+=1
            end
            K_nz[inz] = cC; inz+=1
            if bct[inode+1] == 0 # Symmetrise by removing unnecessary Dirichlet connection
                K_nz[inz] = cN; inz+=1
            end
        end
    end
    return nothing    
end

function Main_1D_Poisson()
    L        = 1.0              # domain size
    ncy      = 1000             # number of cells, ncy+1=number of nodes
    Δy       = L/ncy            # spacing
    Num      = 1:1:(ncy+1)      # dof numbering
    η        = ones(ncy)        # coefficient
    bct      = zeros(Int,ncy+1) # array that identifies dof type (here Dirichlet:1, else: 0)
    bct[1]   = 1                # identify BC nodes as Dirichlets
    bct[end] = 1                # identify BC nodes as Dirichlets 
    println("Create initial sparse matrix") 
    @time Kuu, K_nz = CreateSparseK(ncy+1, Num, bct)
    println("Update non-zeros. Achtung, it is CSC: so not trivial!")
    @time UpdateSparseK!(K_nz, Δy, η, ncy+1, bct)
    println("Initial Cholesky factorisation")
    @time Kc  = cholesky(Kuu; check = false)
    println("Update Cholesky factors reusing old ones: why does this allocate?")
    @time cholesky!(Kc,Kuu; check = false)
end 

Main_1D_Poisson()