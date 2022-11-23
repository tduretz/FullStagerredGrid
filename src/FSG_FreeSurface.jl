function FreeSurfaceCoefficients( D, hx, dkdx, dkdy, dedx, dedy)
    # ------------------------------------------
    # These coefficients are to be evaluated along the free surface vertices set
    # ------------------------------------------
    # Common denominator
    Cd = 2 * D.D11 .* D.D22 .* dedx .* dedy .* hx + 2 * D.D11 .* D.D23 .* dedx .^ 2 .* hx + D.D11 .* D.D23 .* dedy .^ 2 .* hx + 2 * D.D11 .* D.D32 .* dedx .* dedy .* hx .^ 2 + 2 * D.D11 .* D.D33 .* dedx .^ 2 .* hx .^ 2 + D.D11 .* D.D33 .* dedy .^ 2 .* hx .^ 2 - 2 * D.D12 .* D.D21 .* dedx .* dedy .* hx - D.D12 .* D.D23 .* dedx .^ 2 .* hx - 2 * D.D12 .* D.D23 .* dedy .^ 2 .* hx - 2 * D.D12 .* D.D31 .* dedx .* dedy .* hx .^ 2 - D.D12 .* D.D33 .* dedx .^ 2 .* hx .^ 2 - 2 * D.D12 .* D.D33 .* dedy .^ 2 .* hx .^ 2 - 2 * D.D13 .* D.D21 .* dedx .^ 2 .* hx - D.D13 .* D.D21 .* dedy .^ 2 .* hx + D.D13 .* D.D22 .* dedx .^ 2 .* hx + 2 * D.D13 .* D.D22 .* dedy .^ 2 .* hx - 2 * D.D13 .* D.D31 .* dedx .^ 2 .* hx .^ 2 - D.D13 .* D.D31 .* dedy .^ 2 .* hx .^ 2 + D.D13 .* D.D32 .* dedx .^ 2 .* hx .^ 2 + 2 * D.D13 .* D.D32 .* dedy .^ 2 .* hx .^ 2 - 2 * D.D21 .* D.D32 .* dedx .* dedy - 2 * D.D21 .* D.D33 .* dedx .^ 2 - D.D21 .* D.D33 .* dedy .^ 2 + 2 * D.D22 .* D.D31 .* dedx .* dedy + D.D22 .* D.D33 .* dedx .^ 2 + 2 * D.D22 .* D.D33 .* dedy .^ 2 + 2 * D.D23 .* D.D31 .* dedx .^ 2 + D.D23 .* D.D31 .* dedy .^ 2 - D.D23 .* D.D32 .* dedx .^ 2 - 2 * D.D23 .* D.D32 .* dedy .^ 2
    # Coefficients for ∂Vx∂y = C0*P + C1*∂Vy∂ξ + C2*∂Vy∂η
    C0 = (2 * D.D11 .* dedy .* hx - 4 * D.D12 .* dedy .* hx - 3 * D.D13 .* dedx .* hx - 2 * D.D21 .* dedy .* hx + 4 * D.D22 .* dedy .* hx + 3 * D.D23 .* dedx .* hx - 2 * D.D31 .* dedy .* hx .^ 2 + 2 * D.D31 .* dedy + 4 * D.D32 .* dedy .* hx .^ 2 - 4 * D.D32 .* dedy + 3 * D.D33 .* dedx .* hx .^ 2 - 3 * D.D33 .* dedx) ./ Cd
    C1 = (-2 * D.D11 .* D.D22 .* dedy .* dkdx .* hx - 2 * D.D11 .* D.D23 .* dedx .* dkdx .* hx - D.D11 .* D.D23 .* dedy .* dkdy .* hx - 2 * D.D11 .* D.D32 .* dedy .* dkdx .* hx .^ 2 - 2 * D.D11 .* D.D33 .* dedx .* dkdx .* hx .^ 2 - D.D11 .* D.D33 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D12 .* D.D21 .* dedy .* dkdx .* hx + D.D12 .* D.D23 .* dedx .* dkdx .* hx + 2 * D.D12 .* D.D23 .* dedy .* dkdy .* hx + 2 * D.D12 .* D.D31 .* dedy .* dkdx .* hx .^ 2 + D.D12 .* D.D33 .* dedx .* dkdx .* hx .^ 2 + 2 * D.D12 .* D.D33 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D13 .* D.D21 .* dedx .* dkdx .* hx + D.D13 .* D.D21 .* dedy .* dkdy .* hx - D.D13 .* D.D22 .* dedx .* dkdx .* hx - 2 * D.D13 .* D.D22 .* dedy .* dkdy .* hx + 2 * D.D13 .* D.D31 .* dedx .* dkdx .* hx .^ 2 + D.D13 .* D.D31 .* dedy .* dkdy .* hx .^ 2 - D.D13 .* D.D32 .* dedx .* dkdx .* hx .^ 2 - 2 * D.D13 .* D.D32 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D21 .* D.D32 .* dedy .* dkdx + 2 * D.D21 .* D.D33 .* dedx .* dkdx + D.D21 .* D.D33 .* dedy .* dkdy - 2 * D.D22 .* D.D31 .* dedy .* dkdx - D.D22 .* D.D33 .* dedx .* dkdx - 2 * D.D22 .* D.D33 .* dedy .* dkdy - 2 * D.D23 .* D.D31 .* dedx .* dkdx - D.D23 .* D.D31 .* dedy .* dkdy + D.D23 .* D.D32 .* dedx .* dkdx + 2 * D.D23 .* D.D32 .* dedy .* dkdy) ./ Cd
    C2 = (D.D11 .* D.D23 .* dedx .* dkdy .* hx - D.D11 .* D.D23 .* dedy .* dkdx .* hx + D.D11 .* D.D33 .* dedx .* dkdy .* hx .^ 2 - D.D11 .* D.D33 .* dedy .* dkdx .* hx .^ 2 - 2 * D.D12 .* D.D23 .* dedx .* dkdy .* hx + 2 * D.D12 .* D.D23 .* dedy .* dkdx .* hx - 2 * D.D12 .* D.D33 .* dedx .* dkdy .* hx .^ 2 + 2 * D.D12 .* D.D33 .* dedy .* dkdx .* hx .^ 2 - D.D13 .* D.D21 .* dedx .* dkdy .* hx + D.D13 .* D.D21 .* dedy .* dkdx .* hx + 2 * D.D13 .* D.D22 .* dedx .* dkdy .* hx - 2 * D.D13 .* D.D22 .* dedy .* dkdx .* hx - D.D13 .* D.D31 .* dedx .* dkdy .* hx .^ 2 + D.D13 .* D.D31 .* dedy .* dkdx .* hx .^ 2 + 2 * D.D13 .* D.D32 .* dedx .* dkdy .* hx .^ 2 - 2 * D.D13 .* D.D32 .* dedy .* dkdx .* hx .^ 2 - D.D21 .* D.D33 .* dedx .* dkdy + D.D21 .* D.D33 .* dedy .* dkdx + 2 * D.D22 .* D.D33 .* dedx .* dkdy - 2 * D.D22 .* D.D33 .* dedy .* dkdx + D.D23 .* D.D31 .* dedx .* dkdy - D.D23 .* D.D31 .* dedy .* dkdx - 2 * D.D23 .* D.D32 .* dedx .* dkdy + 2 * D.D23 .* D.D32 .* dedy .* dkdx) ./ Cd
    # Coefficients for ∂Vy∂y = D0*P + D1*∂Vy∂ξ + D2*∂Vy∂η
    D0 = (4 * D.D11 .* dedx .* hx - 2 * D.D12 .* dedx .* hx + 3 * D.D13 .* dedy .* hx - 4 * D.D21 .* dedx .* hx + 2 * D.D22 .* dedx .* hx - 3 * D.D23 .* dedy .* hx - 4 * D.D31 .* dedx .* hx .^ 2 + 4 * D.D31 .* dedx + 2 * D.D32 .* dedx .* hx .^ 2 - 2 * D.D32 .* dedx - 3 * D.D33 .* dedy .* hx .^ 2 + 3 * D.D33 .* dedy) ./ Cd
    D1 = (-2 * D.D11 .* D.D23 .* dedx .* dkdy .* hx + 2 * D.D11 .* D.D23 .* dedy .* dkdx .* hx - 2 * D.D11 .* D.D33 .* dedx .* dkdy .* hx .^ 2 + 2 * D.D11 .* D.D33 .* dedy .* dkdx .* hx .^ 2 + D.D12 .* D.D23 .* dedx .* dkdy .* hx - D.D12 .* D.D23 .* dedy .* dkdx .* hx + D.D12 .* D.D33 .* dedx .* dkdy .* hx .^ 2 - D.D12 .* D.D33 .* dedy .* dkdx .* hx .^ 2 + 2 * D.D13 .* D.D21 .* dedx .* dkdy .* hx - 2 * D.D13 .* D.D21 .* dedy .* dkdx .* hx - D.D13 .* D.D22 .* dedx .* dkdy .* hx + D.D13 .* D.D22 .* dedy .* dkdx .* hx + 2 * D.D13 .* D.D31 .* dedx .* dkdy .* hx .^ 2 - 2 * D.D13 .* D.D31 .* dedy .* dkdx .* hx .^ 2 - D.D13 .* D.D32 .* dedx .* dkdy .* hx .^ 2 + D.D13 .* D.D32 .* dedy .* dkdx .* hx .^ 2 + 2 * D.D21 .* D.D33 .* dedx .* dkdy - 2 * D.D21 .* D.D33 .* dedy .* dkdx - D.D22 .* D.D33 .* dedx .* dkdy + D.D22 .* D.D33 .* dedy .* dkdx - 2 * D.D23 .* D.D31 .* dedx .* dkdy + 2 * D.D23 .* D.D31 .* dedy .* dkdx + D.D23 .* D.D32 .* dedx .* dkdy - D.D23 .* D.D32 .* dedy .* dkdx) ./ Cd
    D2 = (-2 * D.D11 .* D.D22 .* dedx .* dkdy .* hx - 2 * D.D11 .* D.D23 .* dedx .* dkdx .* hx - D.D11 .* D.D23 .* dedy .* dkdy .* hx - 2 * D.D11 .* D.D32 .* dedx .* dkdy .* hx .^ 2 - 2 * D.D11 .* D.D33 .* dedx .* dkdx .* hx .^ 2 - D.D11 .* D.D33 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D12 .* D.D21 .* dedx .* dkdy .* hx + D.D12 .* D.D23 .* dedx .* dkdx .* hx + 2 * D.D12 .* D.D23 .* dedy .* dkdy .* hx + 2 * D.D12 .* D.D31 .* dedx .* dkdy .* hx .^ 2 + D.D12 .* D.D33 .* dedx .* dkdx .* hx .^ 2 + 2 * D.D12 .* D.D33 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D13 .* D.D21 .* dedx .* dkdx .* hx + D.D13 .* D.D21 .* dedy .* dkdy .* hx - D.D13 .* D.D22 .* dedx .* dkdx .* hx - 2 * D.D13 .* D.D22 .* dedy .* dkdy .* hx + 2 * D.D13 .* D.D31 .* dedx .* dkdx .* hx .^ 2 + D.D13 .* D.D31 .* dedy .* dkdy .* hx .^ 2 - D.D13 .* D.D32 .* dedx .* dkdx .* hx .^ 2 - 2 * D.D13 .* D.D32 .* dedy .* dkdy .* hx .^ 2 + 2 * D.D21 .* D.D32 .* dedx .* dkdy + 2 * D.D21 .* D.D33 .* dedx .* dkdx + D.D21 .* D.D33 .* dedy .* dkdy - 2 * D.D22 .* D.D31 .* dedx .* dkdy - D.D22 .* D.D33 .* dedx .* dkdx - 2 * D.D22 .* D.D33 .* dedy .* dkdy - 2 * D.D23 .* D.D31 .* dedx .* dkdx - D.D23 .* D.D31 .* dedy .* dkdy + D.D23 .* D.D32 .* dedx .* dkdx + 2 * D.D23 .* D.D32 .* dedy .* dkdy) ./ Cd

    # h_x = hx
    # eta = D.D11/2
    # C1 = 1 .* (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
    # C2 = 1 .* (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
    # C0    = (3 // 2) .* 1 .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2))
    # D1 = 1 .* (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
    # D2 = 1 .* (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)
    # D0    = (3 // 2) .* 1 .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2))

    return (C0=C0, C1=C1, C2=C2, D0=D0, D1=D1, D2=D2)
end

function UpdateFreeSurfaceCoefficients!( BC, D, ∂ξ, ∂η )
    j = size(D.v11.ex,2)-1
    for i in eachindex(BC.C0)

        D_sten = (D11=D.v11.ex[i,j], D12=D.v12.ex[i,j], D13=D.v13.ex[i,j],
                  D21=D.v21.ex[i,j], D22=D.v22.ex[i,j], D23=D.v23.ex[i,j],
                  D31=D.v31.ex[i,j], D32=D.v32.ex[i,j], D33=D.v33.ex[i,j])
        hx = BC.∂h∂x[i]
        coefficients = FreeSurfaceCoefficients( D_sten, hx, ∂ξ.∂x.ex[i,j], ∂ξ.∂y.ex[i,j], ∂η.∂x.ex[i,j], ∂η.∂y.ex[i,j])
                
        BC.C0[i]     = coefficients.C0
        BC.C1[i]     = coefficients.C1
        BC.C2[i]     = coefficients.C2
        BC.D0[i]     = coefficients.D0
        BC.D1[i]     = coefficients.D1
        BC.D2[i]     = coefficients.D2
    end
    print("\n")
end
