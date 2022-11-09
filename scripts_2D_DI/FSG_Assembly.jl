@views   function AddToExtSparse!(K, i, j, Tag_i, Tag_j, v) 
    if ((Tag_j==0 || Tag_j==2) || (j==i && Tag_i==1)) K[i,j]  = v end
end

####################

@views function CoefficientsKuu_Vx!(v_uu, a, b, c, d, D, dx, dy )
    v_uu[1] = (dx.^2 .*(b.C.*(4*D.D11N.*b.N + 4*D.D11S.*b.S - 2*D.D12N.*b.N - 2*D.D12S.*b.S + 3*D.D13N.*d.N + 3*D.D13S.*d.S) + d.C.*(4*D.D31N.*b.N + 4*D.D31S.*b.S - 2*D.D32N.*b.N - 2*D.D32S.*b.S + 3*D.D33N.*d.N + 3*D.D33S.*d.S)) + dy.^2 .*(a.C.*(4*D.D11E.*a.E + 4*D.D11W.*a.W - 2*D.D12E.*a.E - 2*D.D12W.*a.W + 3*D.D13E.*c.E + 3*D.D13W.*c.W) + c.C.*(4*D.D31E.*a.E + 4*D.D31W.*a.W - 2*D.D32E.*a.E - 2*D.D32W.*a.W + 3*D.D33E.*c.E + 3*D.D33W.*c.W)))./(6*dx.^2 .*dy.^2)
    v_uu[2] = (-a.C.*(4*D.D11W.*a.W - 2*D.D12W.*a.W + 3*D.D13W.*c.W) - c.C.*(4*D.D31W.*a.W - 2*D.D32W.*a.W + 3*D.D33W.*c.W))./(6*dx.^2)
    v_uu[3] = (-a.C.*(4*D.D11E.*a.E - 2*D.D12E.*a.E + 3*D.D13E.*c.E) - c.C.*(4*D.D31E.*a.E - 2*D.D32E.*a.E + 3*D.D33E.*c.E))./(6*dx.^2)
    v_uu[4] = (-b.C.*(4*D.D11S.*b.S - 2*D.D12S.*b.S + 3*D.D13S.*d.S) - d.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S))./(6*dy.^2)
    v_uu[5] = (-b.C.*(4*D.D11N.*b.N - 2*D.D12N.*b.N + 3*D.D13N.*d.N) - d.C.*(4*D.D31N.*b.N - 2*D.D32N.*b.N + 3*D.D33N.*d.N))./(6*dy.^2)
    v_uu[6] = (-a.C.*(4*D.D11W.*b.W - 2*D.D12W.*b.W + 3*D.D13W.*d.W) - b.C.*(4*D.D11S.*a.S - 2*D.D12S.*a.S + 3*D.D13S.*c.S) - c.C.*(4*D.D31W.*b.W - 2*D.D32W.*b.W + 3*D.D33W.*d.W) - d.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S))./(6*dx.*dy)
    v_uu[7] = (a.C.*(4*D.D11E.*b.E - 2*D.D12E.*b.E + 3*D.D13E.*d.E) + b.C.*(4*D.D11S.*a.S - 2*D.D12S.*a.S + 3*D.D13S.*c.S) + c.C.*(4*D.D31E.*b.E - 2*D.D32E.*b.E + 3*D.D33E.*d.E) + d.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S))./(6*dx.*dy)
    v_uu[8] = (a.C.*(4*D.D11W.*b.W - 2*D.D12W.*b.W + 3*D.D13W.*d.W) + b.C.*(4*D.D11N.*a.N - 2*D.D12N.*a.N + 3*D.D13N.*c.N) + c.C.*(4*D.D31W.*b.W - 2*D.D32W.*b.W + 3*D.D33W.*d.W) + d.C.*(4*D.D31N.*a.N - 2*D.D32N.*a.N + 3*D.D33N.*c.N))./(6*dx.*dy)
    v_uu[9] = (-a.C.*(4*D.D11E.*b.E - 2*D.D12E.*b.E + 3*D.D13E.*d.E) - b.C.*(4*D.D11N.*a.N - 2*D.D12N.*a.N + 3*D.D13N.*c.N) - c.C.*(4*D.D31E.*b.E - 2*D.D32E.*b.E + 3*D.D33E.*d.E) - d.C.*(4*D.D31N.*a.N - 2*D.D32N.*a.N + 3*D.D33N.*c.N))./(6*dx.*dy)
    v_uu[10] = (dx.^2 .*(b.C.*(-2*D.D11N.*d.N - 2*D.D11S.*d.S + 4*D.D12N.*d.N + 4*D.D12S.*d.S + 3*D.D13N.*b.N + 3*D.D13S.*b.S) + d.C.*(-2*D.D31N.*d.N - 2*D.D31S.*d.S + 4*D.D32N.*d.N + 4*D.D32S.*d.S + 3*D.D33N.*b.N + 3*D.D33S.*b.S)) + dy.^2 .*(a.C.*(-2*D.D11E.*c.E - 2*D.D11W.*c.W + 4*D.D12E.*c.E + 4*D.D12W.*c.W + 3*D.D13E.*a.E + 3*D.D13W.*a.W) + c.C.*(-2*D.D31E.*c.E - 2*D.D31W.*c.W + 4*D.D32E.*c.E + 4*D.D32W.*c.W + 3*D.D33E.*a.E + 3*D.D33W.*a.W)))./(6*dx.^2 .*dy.^2)
    v_uu[11] = (-a.C.*(-2*D.D11W.*c.W + 4*D.D12W.*c.W + 3*D.D13W.*a.W) - c.C.*(-2*D.D31W.*c.W + 4*D.D32W.*c.W + 3*D.D33W.*a.W))./(6*dx.^2)
    v_uu[12] = (-a.C.*(-2*D.D11E.*c.E + 4*D.D12E.*c.E + 3*D.D13E.*a.E) - c.C.*(-2*D.D31E.*c.E + 4*D.D32E.*c.E + 3*D.D33E.*a.E))./(6*dx.^2)
    v_uu[13] = (-b.C.*(-2*D.D11S.*d.S + 4*D.D12S.*d.S + 3*D.D13S.*b.S) - d.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S))./(6*dy.^2)
    v_uu[14] = (-b.C.*(-2*D.D11N.*d.N + 4*D.D12N.*d.N + 3*D.D13N.*b.N) - d.C.*(-2*D.D31N.*d.N + 4*D.D32N.*d.N + 3*D.D33N.*b.N))./(6*dy.^2)
    v_uu[15] = (-a.C.*(-2*D.D11W.*d.W + 4*D.D12W.*d.W + 3*D.D13W.*b.W) - b.C.*(-2*D.D11S.*c.S + 4*D.D12S.*c.S + 3*D.D13S.*a.S) - c.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W) - d.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S))./(6*dx.*dy)
    v_uu[16] = (a.C.*(-2*D.D11E.*d.E + 4*D.D12E.*d.E + 3*D.D13E.*b.E) + b.C.*(-2*D.D11S.*c.S + 4*D.D12S.*c.S + 3*D.D13S.*a.S) + c.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E) + d.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S))./(6*dx.*dy)
    v_uu[17] = (a.C.*(-2*D.D11W.*d.W + 4*D.D12W.*d.W + 3*D.D13W.*b.W) + b.C.*(-2*D.D11N.*c.N + 4*D.D12N.*c.N + 3*D.D13N.*a.N) + c.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W) + d.C.*(-2*D.D31N.*c.N + 4*D.D32N.*c.N + 3*D.D33N.*a.N))./(6*dx.*dy)
    v_uu[18] = (-a.C.*(-2*D.D11E.*d.E + 4*D.D12E.*d.E + 3*D.D13E.*b.E) - b.C.*(-2*D.D11N.*c.N + 4*D.D12N.*c.N + 3*D.D13N.*a.N) - c.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E) - d.C.*(-2*D.D31N.*c.N + 4*D.D32N.*c.N + 3*D.D33N.*a.N))./(6*dx.*dy)
end

@views function CoefficientsKuu_Vx_FreeSurf!(v_uu, a, b, c, d, D, dx, dy )
    v_uu[1] = (2*dx.^2 .*(b.C.*(4*D.D11S.*b.S - 2*D.D12S.*b.S + 3*D.D13S.*d.S) + d.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S)) + dy.^2 .*(a.C.*(2*D.D11E.*(4*a.E - d.E) + 2*D.D11W.*(4*a.W - d.W) - 4*D.D12E.*(a.E - d.E) - 4*D.D12W.*(a.W - d.W) + 3*D.D13E.*(b.E + 2*c.E) + 3*D.D13W.*(b.W + 2*c.W)) + c.C.*(2*D.D31E.*(4*a.E - d.E) + 2*D.D31W.*(4*a.W - d.W) - 4*D.D32E.*(a.E - d.E) - 4*D.D32W.*(a.W - d.W) + 3*D.D33E.*(b.E + 2*c.E) + 3*D.D33W.*(b.W + 2*c.W))))./(12*dx.^2 .*dy.^2)
    v_uu[2] = (-a.C.*(2*D.D11W.*(4*a.W - d.W) - 4*D.D12W.*(a.W - d.W) + 3*D.D13W.*(b.W + 2*c.W)) - c.C.*(2*D.D31W.*(4*a.W - d.W) - 4*D.D32W.*(a.W - d.W) + 3*D.D33W.*(b.W + 2*c.W)))./(12*dx.^2)
    v_uu[3] = (-a.C.*(2*D.D11E.*(4*a.E - d.E) - 4*D.D12E.*(a.E - d.E) + 3*D.D13E.*(b.E + 2*c.E)) - c.C.*(2*D.D31E.*(4*a.E - d.E) - 4*D.D32E.*(a.E - d.E) + 3*D.D33E.*(b.E + 2*c.E)))./(12*dx.^2)
    v_uu[4] = (-b.C.*(4*D.D11S.*b.S - 2*D.D12S.*b.S + 3*D.D13S.*d.S) - d.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S))./(6*dy.^2)
    v_uu[5] = 0
    v_uu[6] = (-b.C.*(4*D.D11S.*a.S - 2*D.D12S.*a.S + 3*D.D13S.*c.S) - d.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S))./(6*dx.*dy)
    v_uu[7] = (b.C.*(4*D.D11S.*a.S - 2*D.D12S.*a.S + 3*D.D13S.*c.S) + d.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S))./(6*dx.*dy)
    v_uu[8] = 0
    v_uu[9] = 0
    v_uu[10] = (dx.^2 .*(b.C.*(-2*D.D11S.*d.S + 4*D.D12S.*d.S + 3*D.D13S.*b.S) + d.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S)) + dy.^2 .*(a.C.*(-2*D.D11E.*(2*b.E + c.E) - 2*D.D11W.*(2*b.W + c.W) + 2*D.D12E.*(b.E + 2*c.E) + 2*D.D12W.*(b.W + 2*c.W) + 3*D.D13E.*(a.E - d.E) + 3*D.D13W.*(a.W - d.W)) + c.C.*(-2*D.D31E.*(2*b.E + c.E) - 2*D.D31W.*(2*b.W + c.W) + 2*D.D32E.*(b.E + 2*c.E) + 2*D.D32W.*(b.W + 2*c.W) + 3*D.D33E.*(a.E - d.E) + 3*D.D33W.*(a.W - d.W))))./(6*dx.^2 .*dy.^2)
    v_uu[11] = (-a.C.*(-2*D.D11W.*(2*b.W + c.W) + 2*D.D12W.*(b.W + 2*c.W) + 3*D.D13W.*(a.W - d.W)) - c.C.*(-2*D.D31W.*(2*b.W + c.W) + 2*D.D32W.*(b.W + 2*c.W) + 3*D.D33W.*(a.W - d.W)))./(6*dx.^2)
    v_uu[12] = (-a.C.*(-2*D.D11E.*(2*b.E + c.E) + 2*D.D12E.*(b.E + 2*c.E) + 3*D.D13E.*(a.E - d.E)) - c.C.*(-2*D.D31E.*(2*b.E + c.E) + 2*D.D32E.*(b.E + 2*c.E) + 3*D.D33E.*(a.E - d.E)))./(6*dx.^2)
    v_uu[13] = (-b.C.*(-2*D.D11S.*d.S + 4*D.D12S.*d.S + 3*D.D13S.*b.S) - d.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S))./(6*dy.^2)
    v_uu[14] = 0
    v_uu[15] = (-b.C.*(-2*D.D11S.*c.S + 4*D.D12S.*c.S + 3*D.D13S.*a.S) - d.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S))./(6*dx.*dy)
    v_uu[16] = (b.C.*(-2*D.D11S.*c.S + 4*D.D12S.*c.S + 3*D.D13S.*a.S) + d.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S))./(6*dx.*dy)
    v_uu[17] = 0
    v_uu[18] = 0
end

####################

@views function CoefficientsKup_Vx!(v_up, aC, bC, dx, dy )
    v_up[1] = -aC./dx
    v_up[2] = aC./dx
    v_up[3] = -bC./dy
    v_up[4] = bC./dy
end

@views function CoefficientsKup_Vx_FreeSurf!(v_up, a, b, c, d, D, dx, dy )
    eta_W = D.D11W/2
    eta_E = D.D11E/2
    v_up[1] = 0*( (-8*a.C.*eta_W + a.C.*(-2*D.D11W.*d.W + 4*D.D12W.*d.W + 3*D.D13W.*b.W) + c.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W))./(8*dx.*eta_W) )
    v_up[2] = 0*( ( 8*a.C.*eta_E - a.C.*(-2*D.D11E.*d.E + 4*D.D12E.*d.E + 3*D.D13E.*b.E) - c.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E))./(8*dx.*eta_E) )
    v_up[3] = -b.C./dy
    v_up[4] = b.C./dy
end

####################

# Same as V part?
@views function CoefficientsKuu_Vy!(v_uu, a, b, c, d, D, dx, dy )
    v_uu[1] = (dx.^2 .*(b.C.*(4*D.D11N.*b.N + 4*D.D11S.*b.S - 2*D.D12N.*b.N - 2*D.D12S.*b.S + 3*D.D13N.*d.N + 3*D.D13S.*d.S) + d.C.*(4*D.D31N.*b.N + 4*D.D31S.*b.S - 2*D.D32N.*b.N - 2*D.D32S.*b.S + 3*D.D33N.*d.N + 3*D.D33S.*d.S)) + dy.^2 .*(a.C.*(4*D.D11E.*a.E + 4*D.D11W.*a.W - 2*D.D12E.*a.E - 2*D.D12W.*a.W + 3*D.D13E.*c.E + 3*D.D13W.*c.W) + c.C.*(4*D.D31E.*a.E + 4*D.D31W.*a.W - 2*D.D32E.*a.E - 2*D.D32W.*a.W + 3*D.D33E.*c.E + 3*D.D33W.*c.W)))./(6*dx.^2 .*dy.^2)
    v_uu[2] = (-a.C.*(4*D.D11W.*a.W - 2*D.D12W.*a.W + 3*D.D13W.*c.W) - c.C.*(4*D.D31W.*a.W - 2*D.D32W.*a.W + 3*D.D33W.*c.W))./(6*dx.^2)
    v_uu[1] = (dx.^2 .*(b.C.*(4*D.D31N.*b.N + 4*D.D31S.*b.S - 2*D.D32N.*b.N - 2*D.D32S.*b.S + 3*D.D33N.*d.N + 3*D.D33S.*d.S) + d.C.*(4*D.D21N.*b.N + 4*D.D21S.*b.S - 2*D.D22N.*b.N - 2*D.D22S.*b.S + 3*D.D23N.*d.N + 3*D.D23S.*d.S)) + dy.^2 .*(a.C.*(4*D.D31E.*a.E + 4*D.D31W.*a.W - 2*D.D32E.*a.E - 2*D.D32W.*a.W + 3*D.D33E.*c.E + 3*D.D33W.*c.W) + c.C.*(4*D.D21E.*a.E + 4*D.D21W.*a.W - 2*D.D22E.*a.E - 2*D.D22W.*a.W + 3*D.D23E.*c.E + 3*D.D23W.*c.W)))./(6*dx.^2 .*dy.^2)
    v_uu[2] = (-a.C.*(4*D.D31W.*a.W - 2*D.D32W.*a.W + 3*D.D33W.*c.W) - c.C.*(4*D.D21W.*a.W - 2*D.D22W.*a.W + 3*D.D23W.*c.W))./(6*dx.^2)
    v_uu[3] = (-a.C.*(4*D.D31E.*a.E - 2*D.D32E.*a.E + 3*D.D33E.*c.E) - c.C.*(4*D.D21E.*a.E - 2*D.D22E.*a.E + 3*D.D23E.*c.E))./(6*dx.^2)
    v_uu[4] = (-b.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S) - d.C.*(4*D.D21S.*b.S - 2*D.D22S.*b.S + 3*D.D23S.*d.S))./(6*dy.^2)
    v_uu[5] = (-b.C.*(4*D.D31N.*b.N - 2*D.D32N.*b.N + 3*D.D33N.*d.N) - d.C.*(4*D.D21N.*b.N - 2*D.D22N.*b.N + 3*D.D23N.*d.N))./(6*dy.^2)
    v_uu[6] = (-a.C.*(4*D.D31W.*b.W - 2*D.D32W.*b.W + 3*D.D33W.*d.W) - b.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S) - c.C.*(4*D.D21W.*b.W - 2*D.D22W.*b.W + 3*D.D23W.*d.W) - d.C.*(4*D.D21S.*a.S - 2*D.D22S.*a.S + 3*D.D23S.*c.S))./(6*dx.*dy)
    v_uu[7] = (a.C.*(4*D.D31E.*b.E - 2*D.D32E.*b.E + 3*D.D33E.*d.E) + b.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S) + c.C.*(4*D.D21E.*b.E - 2*D.D22E.*b.E + 3*D.D23E.*d.E) + d.C.*(4*D.D21S.*a.S - 2*D.D22S.*a.S + 3*D.D23S.*c.S))./(6*dx.*dy)
    v_uu[8] = (a.C.*(4*D.D31W.*b.W - 2*D.D32W.*b.W + 3*D.D33W.*d.W) + b.C.*(4*D.D31N.*a.N - 2*D.D32N.*a.N + 3*D.D33N.*c.N) + c.C.*(4*D.D21W.*b.W - 2*D.D22W.*b.W + 3*D.D23W.*d.W) + d.C.*(4*D.D21N.*a.N - 2*D.D22N.*a.N + 3*D.D23N.*c.N))./(6*dx.*dy)
    v_uu[9] = (-a.C.*(4*D.D31E.*b.E - 2*D.D32E.*b.E + 3*D.D33E.*d.E) - b.C.*(4*D.D31N.*a.N - 2*D.D32N.*a.N + 3*D.D33N.*c.N) - c.C.*(4*D.D21E.*b.E - 2*D.D22E.*b.E + 3*D.D23E.*d.E) - d.C.*(4*D.D21N.*a.N - 2*D.D22N.*a.N + 3*D.D23N.*c.N))./(6*dx.*dy)
    v_uu[10] = (dx.^2 .*(b.C.*(-2*D.D31N.*d.N - 2*D.D31S.*d.S + 4*D.D32N.*d.N + 4*D.D32S.*d.S + 3*D.D33N.*b.N + 3*D.D33S.*b.S) + d.C.*(-2*D.D21N.*d.N - 2*D.D21S.*d.S + 4*D.D22N.*d.N + 4*D.D22S.*d.S + 3*D.D23N.*b.N + 3*D.D23S.*b.S)) + dy.^2 .*(a.C.*(-2*D.D31E.*c.E - 2*D.D31W.*c.W + 4*D.D32E.*c.E + 4*D.D32W.*c.W + 3*D.D33E.*a.E + 3*D.D33W.*a.W) + c.C.*(-2*D.D21E.*c.E - 2*D.D21W.*c.W + 4*D.D22E.*c.E + 4*D.D22W.*c.W + 3*D.D23E.*a.E + 3*D.D23W.*a.W)))./(6*dx.^2 .*dy.^2)
    v_uu[11] = (-a.C.*(-2*D.D31W.*c.W + 4*D.D32W.*c.W + 3*D.D33W.*a.W) - c.C.*(-2*D.D21W.*c.W + 4*D.D22W.*c.W + 3*D.D23W.*a.W))./(6*dx.^2)
    v_uu[12] = (-a.C.*(-2*D.D31E.*c.E + 4*D.D32E.*c.E + 3*D.D33E.*a.E) - c.C.*(-2*D.D21E.*c.E + 4*D.D22E.*c.E + 3*D.D23E.*a.E))./(6*dx.^2)
    v_uu[13] = (-b.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S) - d.C.*(-2*D.D21S.*d.S + 4*D.D22S.*d.S + 3*D.D23S.*b.S))./(6*dy.^2)
    v_uu[14] = (-b.C.*(-2*D.D31N.*d.N + 4*D.D32N.*d.N + 3*D.D33N.*b.N) - d.C.*(-2*D.D21N.*d.N + 4*D.D22N.*d.N + 3*D.D23N.*b.N))./(6*dy.^2)
    v_uu[15] = (-a.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W) - b.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S) - c.C.*(-2*D.D21W.*d.W + 4*D.D22W.*d.W + 3*D.D23W.*b.W) - d.C.*(-2*D.D21S.*c.S + 4*D.D22S.*c.S + 3*D.D23S.*a.S))./(6*dx.*dy)
    v_uu[16] = (a.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E) + b.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S) + c.C.*(-2*D.D21E.*d.E + 4*D.D22E.*d.E + 3*D.D23E.*b.E) + d.C.*(-2*D.D21S.*c.S + 4*D.D22S.*c.S + 3*D.D23S.*a.S))./(6*dx.*dy)
    v_uu[17] = (a.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W) + b.C.*(-2*D.D31N.*c.N + 4*D.D32N.*c.N + 3*D.D33N.*a.N) + c.C.*(-2*D.D21W.*d.W + 4*D.D22W.*d.W + 3*D.D23W.*b.W) + d.C.*(-2*D.D21N.*c.N + 4*D.D22N.*c.N + 3*D.D23N.*a.N))./(6*dx.*dy)
    v_uu[18] = (-a.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E) - b.C.*(-2*D.D31N.*c.N + 4*D.D32N.*c.N + 3*D.D33N.*a.N) - c.C.*(-2*D.D21E.*d.E + 4*D.D22E.*d.E + 3*D.D23E.*b.E) - d.C.*(-2*D.D21N.*c.N + 4*D.D22N.*c.N + 3*D.D23N.*a.N))./(6*dx.*dy)
end

@views function CoefficientsKuu_Vy_FreeSurf!(v_uu, a, b, c, d, D, dx, dy )
    v_uu[1] = (2*dx.^2 .*(b.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S) + d.C.*(4*D.D21S.*b.S - 2*D.D22S.*b.S + 3*D.D23S.*d.S)) + dy.^2 .*(a.C.*(2*D.D31E.*(4*a.E - d.E) + 2*D.D31W.*(4*a.W - d.W) - 4*D.D32E.*(a.E - d.E) - 4*D.D32W.*(a.W - d.W) + 3*D.D33E.*(b.E + 2*c.E) + 3*D.D33W.*(b.W + 2*c.W)) + c.C.*(2*D.D21E.*(4*a.E - d.E) + 2*D.D21W.*(4*a.W - d.W) - 4*D.D22E.*(a.E - d.E) - 4*D.D22W.*(a.W - d.W) + 3*D.D23E.*(b.E + 2*c.E) + 3*D.D23W.*(b.W + 2*c.W))))./(12*dx.^2 .*dy.^2)
    v_uu[2] = (-a.C.*(2*D.D31W.*(4*a.W - d.W) - 4*D.D32W.*(a.W - d.W) + 3*D.D33W.*(b.W + 2*c.W)) - c.C.*(2*D.D21W.*(4*a.W - d.W) - 4*D.D22W.*(a.W - d.W) + 3*D.D23W.*(b.W + 2*c.W)))./(12*dx.^2)
    v_uu[3] = (-a.C.*(2*D.D31E.*(4*a.E - d.E) - 4*D.D32E.*(a.E - d.E) + 3*D.D33E.*(b.E + 2*c.E)) - c.C.*(2*D.D21E.*(4*a.E - d.E) - 4*D.D22E.*(a.E - d.E) + 3*D.D23E.*(b.E + 2*c.E)))./(12*dx.^2)
    v_uu[4] = (-b.C.*(4*D.D31S.*b.S - 2*D.D32S.*b.S + 3*D.D33S.*d.S) - d.C.*(4*D.D21S.*b.S - 2*D.D22S.*b.S + 3*D.D23S.*d.S))./(6*dy.^2)
    v_uu[5] = 0
    v_uu[6] = (-b.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S) - d.C.*(4*D.D21S.*a.S - 2*D.D22S.*a.S + 3*D.D23S.*c.S))./(6*dx.*dy)
    v_uu[7] = (b.C.*(4*D.D31S.*a.S - 2*D.D32S.*a.S + 3*D.D33S.*c.S) + d.C.*(4*D.D21S.*a.S - 2*D.D22S.*a.S + 3*D.D23S.*c.S))./(6*dx.*dy)
    v_uu[8] = 0
    v_uu[9] = 0
    v_uu[10] = (dx.^2 .*(b.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S) + d.C.*(-2*D.D21S.*d.S + 4*D.D22S.*d.S + 3*D.D23S.*b.S)) + dy.^2 .*(a.C.*(-2*D.D31E.*(2*b.E + c.E) - 2*D.D31W.*(2*b.W + c.W) + 2*D.D32E.*(b.E + 2*c.E) + 2*D.D32W.*(b.W + 2*c.W) + 3*D.D33E.*(a.E - d.E) + 3*D.D33W.*(a.W - d.W)) + c.C.*(-2*D.D21E.*(2*b.E + c.E) - 2*D.D21W.*(2*b.W + c.W) + 2*D.D22E.*(b.E + 2*c.E) + 2*D.D22W.*(b.W + 2*c.W) + 3*D.D23E.*(a.E - d.E) + 3*D.D23W.*(a.W - d.W))))./(6*dx.^2 .*dy.^2)
    v_uu[11] = (-a.C.*(-2*D.D31W.*(2*b.W + c.W) + 2*D.D32W.*(b.W + 2*c.W) + 3*D.D33W.*(a.W - d.W)) - c.C.*(-2*D.D21W.*(2*b.W + c.W) + 2*D.D22W.*(b.W + 2*c.W) + 3*D.D23W.*(a.W - d.W)))./(6*dx.^2)
    v_uu[12] = (-a.C.*(-2*D.D31E.*(2*b.E + c.E) + 2*D.D32E.*(b.E + 2*c.E) + 3*D.D33E.*(a.E - d.E)) - c.C.*(-2*D.D21E.*(2*b.E + c.E) + 2*D.D22E.*(b.E + 2*c.E) + 3*D.D23E.*(a.E - d.E)))./(6*dx.^2)
    v_uu[13] = (-b.C.*(-2*D.D31S.*d.S + 4*D.D32S.*d.S + 3*D.D33S.*b.S) - d.C.*(-2*D.D21S.*d.S + 4*D.D22S.*d.S + 3*D.D23S.*b.S))./(6*dy.^2)
    v_uu[14] = 0
    v_uu[15] = (-b.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S) - d.C.*(-2*D.D21S.*c.S + 4*D.D22S.*c.S + 3*D.D23S.*a.S))./(6*dx.*dy)
    v_uu[16] = (b.C.*(-2*D.D31S.*c.S + 4*D.D32S.*c.S + 3*D.D33S.*a.S) + d.C.*(-2*D.D21S.*c.S + 4*D.D22S.*c.S + 3*D.D23S.*a.S))./(6*dx.*dy)
    v_uu[17] = 0
    v_uu[18] = 0
end

####################

@views function CoefficientsKup_Vy!(v_up, cC, dC, dx, dy )
    v_up[1] = -cC./dx
    v_up[2] = cC./dx
    v_up[3] = -dC./dy
    v_up[4] = dC./dy
end

@views function CoefficientsKup_Vy_FreeSurf!(v_up, a, b, c, d, D, dx, dy )
    eta_W = D.D22W/2
    eta_E = D.D22E/2
    v_up[1] = 0*( ( a.C.*(-2*D.D31W.*d.W + 4*D.D32W.*d.W + 3*D.D33W.*b.W) - 8*c.C.*eta_W + c.C.*(-2*D.D21W.*d.W + 4*D.D22W.*d.W + 3*D.D23W.*b.W))./(8*dx.*eta_W) )
    v_up[2] = 0*( (-a.C.*(-2*D.D31E.*d.E + 4*D.D32E.*d.E + 3*D.D33E.*b.E) + 8*c.C.*eta_E - c.C.*(-2*D.D21E.*d.E + 4*D.D22E.*d.E + 3*D.D23E.*b.E))./(8*dx.*eta_E) )
    v_up[3] = -d.C./dy
    v_up[4] = d.C./dy                                                
end

####################

@views function CoefficientsKpu!(v_pu,aC,bC,cC,dC,K,dx,dy,dt)
    v_pu[1] = -aC./dx
    v_pu[2] = aC./dx
    v_pu[3] = -bC./dy
    v_pu[4] = bC./dy
    v_pu[5] = -cC./dx
    v_pu[6] = cC./dx
    v_pu[7] = -dC./dy
    v_pu[8] = dC./dy
    v_pu[9] = 1 ./(K.*dt)
end

@views function CoefficientsKpu_FreeSurf!(v_pu,aC,bC,cC,dC,K,eta_C,dx,dy,dt)
    v_pu[1] = (-aC + bC - dC/2)./dx
    v_pu[2] = (aC - bC + dC/2)./dx
    v_pu[3] = 0
    v_pu[4] = 0
    v_pu[5] = -cC./dx
    v_pu[6] = cC./dx
    v_pu[7] = 0
    v_pu[8] = 0
    v_pu[9] = 3*dC./(4*eta_C) + 1 ./(K.*dt)
end

@views function AssembleKuuKupKpu!(Kuu, Kup, Kpu, Kpp, Num, BC, D, ∂ξ, ∂η, Δ, nc, nv)
    i_uu    = zeros(Int, 18); t_uu    = zeros(Int, 18); v_uu   = ones(18)
    i_up    = zeros(Int,  4); t_up    = zeros(Int,  4); v_up   = ones( 4)
    i_pu    = zeros(Int,  9); t_pu    = zeros(Int,  9); v_pu   = ones( 9)
    # Fake compressebility
    K = 1e10
    # ==================================== Kuu, Kup ==================================== #
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
        i_uu[12] = Num.y.v[i+1,j];   t_uu[12] = BC.y.v[i+1,j]    # E
        i_uu[13] = Num.y.v[i,j-1];   t_uu[13] = BC.y.v[i,j-1]    # S
        i_uu[14] = Num.y.v[i,j+1];   t_uu[14] = BC.y.v[i,j+1]    # N
        i_uu[15] = Num.y.c[i-1,j-1]; t_uu[15] = BC.y.c[i-1,j-1]  # SW
        i_uu[16] = Num.y.c[i,j-1];   t_uu[16] = BC.y.c[i,j-1]    # SE
        i_uu[17] = Num.y.c[i-1,j];   t_uu[17] = BC.y.c[i-1,j]    # NW
        i_uu[18] = Num.y.c[i,j];     t_uu[18] = BC.y.c[i,j]      # NE
        #--------------
        i_up[1]  = Num.p.ex[i-1,j];  t_up[1]  = BC.p.ex[i-1,j]   # W
        i_up[2]  = Num.p.ex[i,j];    t_up[2]  = BC.p.ex[i,j]     # E
        i_up[3]  = Num.p.ey[i,j-1];  t_up[3]  = BC.p.ey[i,j-1]   # S   
        i_up[4]  = Num.p.ey[i,j];    t_up[4]  = BC.p.ey[i,j]     # N
        #--------------
        D_sten = ( D11W=D.v11.ex[i-1,j], D11E=D.v11.ex[i,j], D11S=D.v11.ey[i,j-1], D11N=D.v11.ey[i,j], 
                   D12W=D.v12.ex[i-1,j], D12E=D.v12.ex[i,j], D12S=D.v12.ey[i,j-1], D12N=D.v12.ey[i,j],
                   D13W=D.v13.ex[i-1,j], D13E=D.v13.ex[i,j], D13S=D.v13.ey[i,j-1], D13N=D.v13.ey[i,j],
                   D21W=D.v21.ex[i-1,j], D21E=D.v21.ex[i,j], D21S=D.v21.ey[i,j-1], D21N=D.v21.ey[i,j],
                   D22W=D.v22.ex[i-1,j], D22E=D.v22.ex[i,j], D22S=D.v22.ey[i,j-1], D22N=D.v22.ey[i,j],
                   D23W=D.v23.ex[i-1,j], D23E=D.v23.ex[i,j], D23S=D.v23.ey[i,j-1], D23N=D.v23.ey[i,j],
                   D31W=D.v31.ex[i-1,j], D31E=D.v31.ex[i,j], D31S=D.v31.ey[i,j-1], D31N=D.v31.ey[i,j],
                   D32W=D.v32.ex[i-1,j], D32E=D.v32.ex[i,j], D32S=D.v32.ey[i,j-1], D32N=D.v32.ey[i,j],
                   D33W=D.v33.ex[i-1,j], D33E=D.v33.ex[i,j], D33S=D.v33.ey[i,j-1], D33N=D.v33.ey[i,j],
        )
        a = (C=∂ξ.∂x.v[i,j], W=∂ξ.∂x.ex[i-1,j], E=∂ξ.∂x.ex[i,j], S=∂ξ.∂x.ey[i,j-1], N=∂ξ.∂x.ey[i,j])
        b = (C=∂ξ.∂y.v[i,j], W=∂ξ.∂y.ex[i-1,j], E=∂ξ.∂y.ex[i,j], S=∂ξ.∂y.ey[i,j-1], N=∂ξ.∂y.ey[i,j])
        c = (C=∂η.∂x.v[i,j], W=∂η.∂x.ex[i-1,j], E=∂η.∂x.ex[i,j], S=∂η.∂x.ey[i,j-1], N=∂η.∂x.ey[i,j])
        d = (C=∂η.∂y.v[i,j], W=∂η.∂y.ex[i-1,j], E=∂η.∂y.ex[i,j], S=∂η.∂y.ey[i,j-1], N=∂η.∂y.ey[i,j])
        #--------------
        # Vx
        if BC.x.v[i,j] == 0
            CoefficientsKuu_Vx!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
            CoefficientsKup_Vx!(v_up, a.C, b.C, Δ.x, Δ.y )
        elseif BC.x.v[i,j] == 2
            println("FREE x")
            CoefficientsKuu_Vx_FreeSurf!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
            CoefficientsKup_Vx_FreeSurf!(v_up, a, b, c, d, D_sten, Δ.x, Δ.y )
        end
        # Dirichlets
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[1], i_uu[1], BC.x.v[i,j], t_uu[1], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[1], i_uu[ii], BC.x.v[i,j],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[1], i_up[ii], BC.x.v[i,j],  t_up[ii],  v_up[ii])
            end
        end
        #--------------
        # Vy
        if BC.y.v[i,j] == 0
            CoefficientsKuu_Vy!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
            CoefficientsKup_Vy!(v_up, c.C, d.C, Δ.x, Δ.y )
        elseif BC.x.v[i,j] == 2
            println("FREE y")
            CoefficientsKuu_Vy_FreeSurf!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
            CoefficientsKup_Vy_FreeSurf!(v_up, a, b, c, d, D_sten, Δ.x, Δ.y )
        end
        # Dirichlets
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[10], i_uu[10], BC.y.v[i,j], t_uu[10], 1.0)
        else
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[10], i_uu[ii], BC.y.v[i,j],  t_uu[ii],  v_uu[ii])
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[10], i_up[ii], BC.y.v[i,j],  t_up[ii],  v_up[ii])
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
        i_uu[12] = Num.y.c[i+1,j];   t_uu[12] = BC.y.c[i+1,j]    # E
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
        #--------------
        D_sten = ( D11W=D.v11.ey[i,j], D11E=D.v11.ey[i+1,j], D11S=D.v11.ex[i,j], D11N=D.v11.ex[i,j+1], 
                   D12W=D.v12.ey[i,j], D12E=D.v12.ey[i+1,j], D12S=D.v12.ex[i,j], D12N=D.v12.ex[i,j+1],
                   D13W=D.v13.ey[i,j], D13E=D.v13.ey[i+1,j], D13S=D.v13.ex[i,j], D13N=D.v13.ex[i,j+1],
                   D21W=D.v21.ey[i,j], D21E=D.v21.ey[i+1,j], D21S=D.v21.ex[i,j], D21N=D.v21.ex[i,j+1],
                   D22W=D.v22.ey[i,j], D22E=D.v22.ey[i+1,j], D22S=D.v22.ex[i,j], D22N=D.v22.ex[i,j+1],
                   D23W=D.v23.ey[i,j], D23E=D.v23.ey[i+1,j], D23S=D.v23.ex[i,j], D23N=D.v23.ex[i,j+1],
                   D31W=D.v31.ey[i,j], D31E=D.v31.ey[i+1,j], D31S=D.v31.ex[i,j], D31N=D.v31.ex[i,j+1],
                   D32W=D.v32.ey[i,j], D32E=D.v32.ey[i+1,j], D32S=D.v32.ex[i,j], D32N=D.v32.ex[i,j+1],
                   D33W=D.v33.ey[i,j], D33E=D.v33.ey[i+1,j], D33S=D.v33.ex[i,j], D33N=D.v33.ex[i,j+1],
        )
        a = (C=∂ξ.∂x.c[i,j], W=∂ξ.∂x.ey[i,j], E=∂ξ.∂x.ey[i+1,j], S=∂ξ.∂x.ex[i,j], N=∂ξ.∂x.ex[i,j+1])
        b = (C=∂ξ.∂y.c[i,j], W=∂ξ.∂y.ey[i,j], E=∂ξ.∂y.ey[i+1,j], S=∂ξ.∂y.ex[i,j], N=∂ξ.∂y.ex[i,j+1])
        c = (C=∂η.∂x.c[i,j], W=∂η.∂x.ey[i,j], E=∂η.∂x.ey[i+1,j], S=∂η.∂x.ex[i,j], N=∂η.∂x.ex[i,j+1])
        d = (C=∂η.∂y.c[i,j], W=∂η.∂y.ey[i,j], E=∂η.∂y.ey[i+1,j], S=∂η.∂y.ex[i,j], N=∂η.∂y.ex[i,j+1])
        #--------------
        # Vx
        CoefficientsKuu_Vx!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
        CoefficientsKup_Vx!(v_up, a.C, b.C, Δ.x, Δ.y )
        for ii in eachindex(v_uu)
            AddToExtSparse!(Kuu, i_uu[1], i_uu[ii], BC.x.c[i,j],  t_uu[ii],  v_uu[ii])
        end
        #--------------
        for ii in eachindex(v_up)
            AddToExtSparse!(Kup, i_uu[1], i_up[ii], BC.x.c[i,j],  t_up[ii],  v_up[ii])
        end
        #--------------
        # Vy 
        CoefficientsKuu_Vy!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y )
        CoefficientsKup_Vy!(v_up, c.C, d.C, Δ.x, Δ.y )
        for ii in eachindex(v_uu)
            AddToExtSparse!(Kuu, i_uu[10], i_uu[ii], BC.y.c[i,j],  t_uu[ii],  v_uu[ii])
        end
        #--------------
        for ii in eachindex(v_up)
            AddToExtSparse!(Kup, i_uu[10], i_up[ii], BC.y.c[i,j],  t_up[ii],  v_up[ii])
        end
    end 
    # ==================================== Kpu ==================================== #
    # Loop on horizontal edges
    for j in 2:nv.y+1, i in 2:nc.x+1
        i_pp    =  Num.p.ex[i,j]
        #--------------
        i_pu[1] =  Num.x.v[i,j];   t_pu[1] =  BC.x.v[i,j]    # W
        i_pu[2] =  Num.x.v[i+1,j]; t_pu[2] =  BC.x.v[i+1,j]  # E
        i_pu[3] =  Num.x.c[i,j-1]; t_pu[3] =  BC.x.c[i,j-1]  # S
        i_pu[4] =  Num.x.c[i,j];   t_pu[4] =  BC.x.c[i,j]    # N
        #--------------
        i_pu[5] =  Num.y.v[i,j];   t_pu[5] =  BC.y.v[i,j]   
        i_pu[6] =  Num.y.v[i+1,j]; t_pu[6] =  BC.y.v[i+1,j] 
        i_pu[7] =  Num.y.c[i,j-1]; t_pu[7] =  BC.y.c[i,j-1] 
        i_pu[8] =  Num.y.c[i,j];   t_pu[8] =  BC.y.c[i,j]
        #--------------
        i_pu[9] =  Num.p.ex[i,j];  t_pu[9] =  BC.p.ex[i,j]
        #--------------
        if BC.p.ex[i,j] == 0
            CoefficientsKpu!(v_pu, ∂ξ.∂x.ex[i,j], ∂ξ.∂y.ex[i,j], ∂η.∂x.ex[i,j], ∂η.∂y.ex[i,j], K,  Δ.x, Δ.y, Δ.t)
        else BC.p.ex[i,j] == 2
            println("fripé!")
            CoefficientsKpu_FreeSurf!(v_pu, ∂ξ.∂x.ex[i,j], ∂ξ.∂y.ex[i,j], ∂η.∂x.ex[i,j], ∂η.∂y.ex[i,j], K, D.v11.ex[i,j]/2, Δ.x, Δ.y, Δ.t)
        end
        #--------------  
        for ii=1:8
            AddToExtSparse!(Kpu, i_pp, i_pu[ii], BC.p.ex[i,j],  t_pu[ii],  v_pu[ii])
        end
        AddToExtSparse!(Kpp, i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9],  v_pu[9])
    end

    # Loop on vertical edges
    for j in 2:nc.y+1, i in 2:nv.x+1
        i_pp    =  Num.p.ey[i,j]
        #--------------
        i_pu[1] =  Num.x.c[i-1,j]; t_pu[1] =  BC.x.c[i-1,j]   # W
        i_pu[2] =  Num.x.c[i,j];   t_pu[2] =  BC.x.c[i,j]     # E
        i_pu[3] =  Num.x.v[i,j];   t_pu[3] =  BC.x.v[i,j]     # S
        i_pu[4] =  Num.x.v[i,j+1]; t_pu[4] =  BC.x.v[i,j+1]   # N   
        #--------------
        i_pu[5] =  Num.y.c[i-1,j]; t_pu[5] =  BC.y.c[i-1,j]   
        i_pu[6] =  Num.y.c[i,j];   t_pu[6] =  BC.y.c[i,j] 
        i_pu[7] =  Num.y.v[i,j];   t_pu[7] =  BC.y.v[i,j] 
        i_pu[8] =  Num.y.v[i,j+1]; t_pu[8] =  BC.y.v[i,j+1]
        #--------------
        i_pu[9] =  Num.p.ey[i,j];  t_pu[9] =  BC.p.ey[i,j]
        #--------------
        CoefficientsKpu!(v_pu, ∂ξ.∂x.ey[i,j], ∂ξ.∂y.ey[i,j], ∂η.∂x.ey[i,j], ∂η.∂y.ey[i,j], K, Δ.x, Δ.y, Δ.t)
        #--------------
        for ii=1:8
            AddToExtSparse!(Kpu, i_pp, i_pu[ii], BC.p.ey[i,j], t_pu[ii],  v_pu[ii])
        end
        AddToExtSparse!(Kpp, i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9],  v_pu[9])
    end
    flush!(Kuu), flush!(Kup), flush!(Kpu), flush!(Kpp)
end
