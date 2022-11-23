
@views function CoeffSymmetry( a, b, c, d, symmetric )
    if symmetric
        return 1.0
    else
        return 1.0 / (a.C*d.C)
    end
end

@views function AddToExtSparse!(K, i, j, Tag_i, Tag_j, v) 
    if ((Tag_j==0 || Tag_j==2) || (j==i && Tag_i==1)) K[i,j]  = v end
end

####################

@views function CoefficientsKuu_Vx!(v_uu, a, b, c, d, D, dx, dy, symmetric )
    if symmetric
        v_uu[1] = (dx .^ 2 .* (4 * D.D11N .* b.N .^ 2 + 4 * D.D11S .* b.S .^ 2 + 3 * D.D33N .* d.N .^ 2 + 3 * D.D33S .* d.S .^ 2) + dy .^ 2 .* (4 * D.D11E .* a.E .^ 2 + 4 * D.D11W .* a.W .^ 2 + 3 * D.D33E .* c.E .^ 2 + 3 * D.D33W .* c.W .^ 2)) ./ (6 * dx .^ 2 .* dy .^ 2)
        v_uu[2] = (-4 * D.D11W .* a.W .^ 2 - 3 * D.D33W .* c.W .^ 2) ./ (6 * dx .^ 2)
        v_uu[3] = (-4 * D.D11E .* a.E .^ 2 - 3 * D.D33E .* c.E .^ 2) ./ (6 * dx .^ 2)
        v_uu[4] = (-4 * D.D11S .* b.S .^ 2 - 3 * D.D33S .* d.S .^ 2) ./ (6 * dy .^ 2)
        v_uu[5] = (-4 * D.D11N .* b.N .^ 2 - 3 * D.D33N .* d.N .^ 2) ./ (6 * dy .^ 2)
        v_uu[6] = (-4 * D.D11S .* a.S .* b.S - 4 * D.D11W .* a.W .* b.W - 3 * D.D33S .* c.S .* d.S - 3 * D.D33W .* c.W .* d.W) ./ (6 * dx .* dy)
        v_uu[7] = (4 * D.D11E .* a.E .* b.E + 4 * D.D11S .* a.S .* b.S + 3 * D.D33E .* c.E .* d.E + 3 * D.D33S .* c.S .* d.S) ./ (6 * dx .* dy)
        v_uu[8] = (4 * D.D11N .* a.N .* b.N + 4 * D.D11W .* a.W .* b.W + 3 * D.D33N .* c.N .* d.N + 3 * D.D33W .* c.W .* d.W) ./ (6 * dx .* dy)
        v_uu[9] = (-4 * D.D11E .* a.E .* b.E - 4 * D.D11N .* a.N .* b.N - 3 * D.D33E .* c.E .* d.E - 3 * D.D33N .* c.N .* d.N) ./ (6 * dx .* dy)
        v_uu[10] = (dx .^ 2 .* (-2 * D.D11N .* b.N .* d.N - 2 * D.D11S .* b.S .* d.S + 3 * D.D33N .* b.N .* d.N + 3 * D.D33S .* b.S .* d.S) + dy .^ 2 .* (-2 * D.D11E .* a.E .* c.E - 2 * D.D11W .* a.W .* c.W + 3 * D.D33E .* a.E .* c.E + 3 * D.D33W .* a.W .* c.W)) ./ (6 * dx .^ 2 .* dy .^ 2)
        v_uu[11] = a.W .* c.W .* (2 * D.D11W - 3 * D.D33W) ./ (6 * dx .^ 2)
        v_uu[12] = a.E .* c.E .* (2 * D.D11E - 3 * D.D33E) ./ (6 * dx .^ 2)
        v_uu[13] = b.S .* d.S .* (2 * D.D11S - 3 * D.D33S) ./ (6 * dy .^ 2)
        v_uu[14] = b.N .* d.N .* (2 * D.D11N - 3 * D.D33N) ./ (6 * dy .^ 2)
        v_uu[15] = (D.D11S .* b.S .* c.S / 3 + D.D11W .* a.W .* d.W / 3 - D.D33S .* a.S .* d.S / 2 - D.D33W .* b.W .* c.W / 2) ./ (dx .* dy)
        v_uu[16] = (-D.D11E .* a.E .* d.E / 3 - D.D11S .* b.S .* c.S / 3 + D.D33E .* b.E .* c.E / 2 + D.D33S .* a.S .* d.S / 2) ./ (dx .* dy)
        v_uu[17] = (-D.D11N .* b.N .* c.N / 3 - D.D11W .* a.W .* d.W / 3 + D.D33N .* a.N .* d.N / 2 + D.D33W .* b.W .* c.W / 2) ./ (dx .* dy)
        v_uu[18] = (D.D11E .* a.E .* d.E / 3 + D.D11N .* b.N .* c.N / 3 - D.D33E .* b.E .* c.E / 2 - D.D33N .* a.N .* d.N / 2) ./ (dx .* dy) 
    else
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
end

@views function CoefficientsKuu_Vxv_FreeSurf!(v_uu, a, b, c, d, D, fs, dx, dy, symmetric )
    v_uu[1] = (dx .^ 2 .* (b.C .* (4 * D.D11S .* b.S - 2 * D.D12S .* b.S + 3 * D.D13S .* d.S) + d.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S)) + dy .^ 2 .* (a.C .* (2 * D.D11E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) + 2 * D.D11W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D12E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) - 2 * D.D12W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D13E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E) + 3 * D.D13W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)) + c.C .* (2 * D.D31E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) + 2 * D.D31W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D32E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) - 2 * D.D32W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D33E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E) + 3 * D.D33W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[2] = (-a.C .* (2 * D.D11W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D12W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D13W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)) - c.C .* (2 * D.D31W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D32W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D33W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W))) ./ (6 * dx .^ 2)
    v_uu[3] = (-a.C .* (2 * D.D11E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) - 2 * D.D12E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) + 3 * D.D13E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E)) - c.C .* (2 * D.D31E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) - 2 * D.D32E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) + 3 * D.D33E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E))) ./ (6 * dx .^ 2)
    v_uu[4] = (-b.C .* (4 * D.D11S .* b.S - 2 * D.D12S .* b.S + 3 * D.D13S .* d.S) - d.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S)) ./ (6 * dy .^ 2)
    v_uu[5] = 0
    v_uu[6] = (-b.C .* (4 * D.D11S .* a.S - 2 * D.D12S .* a.S + 3 * D.D13S .* c.S) - d.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S)) ./ (6 * dx .* dy)
    v_uu[7] = (b.C .* (4 * D.D11S .* a.S - 2 * D.D12S .* a.S + 3 * D.D13S .* c.S) + d.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S)) ./ (6 * dx .* dy)
    v_uu[8] = 0
    v_uu[9] = 0
    v_uu[10] = (dx .^ 2 .* (b.C .* (-2 * D.D11S .* d.S + 4 * D.D12S .* d.S + 3 * D.D13S .* b.S) + d.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S)) + dy .^ 2 .* (a.C .* (-2 * D.D11E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) - 2 * D.D11W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D12E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 2 * D.D12W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D13E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E) + 3 * D.D13W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)) + c.C .* (-2 * D.D31E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) - 2 * D.D31W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D32E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 2 * D.D32W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D33E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E) + 3 * D.D33W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[11] = (-a.C .* (-2 * D.D11W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D12W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D13W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)) - c.C .* (-2 * D.D31W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D32W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D33W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W))) ./ (6 * dx .^ 2)
    v_uu[12] = (-a.C .* (-2 * D.D11E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) + 2 * D.D12E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 3 * D.D13E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E)) - c.C .* (-2 * D.D31E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) + 2 * D.D32E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 3 * D.D33E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E))) ./ (6 * dx .^ 2)
    v_uu[13] = (-b.C .* (-2 * D.D11S .* d.S + 4 * D.D12S .* d.S + 3 * D.D13S .* b.S) - d.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S)) ./ (6 * dy .^ 2)
    v_uu[14] = 0
    v_uu[15] = (-b.C .* (-2 * D.D11S .* c.S + 4 * D.D12S .* c.S + 3 * D.D13S .* a.S) - d.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S)) ./ (6 * dx .* dy)
    v_uu[16] = (b.C .* (-2 * D.D11S .* c.S + 4 * D.D12S .* c.S + 3 * D.D13S .* a.S) + d.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S)) ./ (6 * dx .* dy)
    v_uu[17] = 0
    v_uu[18] = 0
end

@views function CoefficientsKuu_Vxc_FreeSurf!(v_uu, a, b, c, d, D, fs, dx, dy, symmetric )
    v_uu[1] = (dx .^ 2 .* (b.C .* (4 * D.D11S .* b.S - 2 * D.D12S .* b.S + 3 * D.D13S .* d.S) + d.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S)) + dy .^ 2 .* (a.C .* (4 * D.D11E .* a.E + 4 * D.D11W .* a.W - 2 * D.D12E .* a.E - 2 * D.D12W .* a.W + 3 * D.D13E .* c.E + 3 * D.D13W .* c.W) + c.C .* (4 * D.D31E .* a.E + 4 * D.D31W .* a.W - 2 * D.D32E .* a.E - 2 * D.D32W .* a.W + 3 * D.D33E .* c.E + 3 * D.D33W .* c.W))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[2] = (-a.C .* (4 * D.D11W .* a.W - 2 * D.D12W .* a.W + 3 * D.D13W .* c.W) - c.C .* (4 * D.D31W .* a.W - 2 * D.D32W .* a.W + 3 * D.D33W .* c.W)) ./ (6 * dx .^ 2)
    v_uu[3] = (-a.C .* (4 * D.D11E .* a.E - 2 * D.D12E .* a.E + 3 * D.D13E .* c.E) - c.C .* (4 * D.D31E .* a.E - 2 * D.D32E .* a.E + 3 * D.D33E .* c.E)) ./ (6 * dx .^ 2)
    v_uu[4] = (-b.C .* (4 * D.D11S .* b.S - 2 * D.D12S .* b.S + 3 * D.D13S .* d.S) - d.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S)) ./ (6 * dy .^ 2)
    v_uu[5] = 0
    v_uu[6] = (-a.C .* (4 * D.D11W .* b.W - 2 * D.D12W .* b.W + 3 * D.D13W .* d.W) - b.C .* (4 * D.D11S .* a.S - 2 * D.D12S .* a.S + 3 * D.D13S .* c.S) - c.C .* (4 * D.D31W .* b.W - 2 * D.D32W .* b.W + 3 * D.D33W .* d.W) - d.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S)) ./ (6 * dx .* dy)
    v_uu[7] = (a.C .* (4 * D.D11E .* b.E - 2 * D.D12E .* b.E + 3 * D.D13E .* d.E) + b.C .* (4 * D.D11S .* a.S - 2 * D.D12S .* a.S + 3 * D.D13S .* c.S) + c.C .* (4 * D.D31E .* b.E - 2 * D.D32E .* b.E + 3 * D.D33E .* d.E) + d.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S)) ./ (6 * dx .* dy)
    v_uu[8] = (a.C .* (4 * D.D11W .* b.W - 2 * D.D12W .* b.W + 3 * D.D13W .* d.W) + b.C .* (2 * D.D11N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D12N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D13N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N)) + c.C .* (4 * D.D31W .* b.W - 2 * D.D32W .* b.W + 3 * D.D33W .* d.W) + d.C .* (2 * D.D31N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D32N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D33N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N))) ./ (6 * dx .* dy)
    v_uu[9] = (-a.C .* (4 * D.D11E .* b.E - 2 * D.D12E .* b.E + 3 * D.D13E .* d.E) - b.C .* (2 * D.D11N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D12N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D13N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N)) - c.C .* (4 * D.D31E .* b.E - 2 * D.D32E .* b.E + 3 * D.D33E .* d.E) - d.C .* (2 * D.D31N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D32N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D33N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N))) ./ (6 * dx .* dy)
    v_uu[10] = (dx .^ 2 .* (b.C .* (-2 * D.D11S .* d.S + 4 * D.D12S .* d.S + 3 * D.D13S .* b.S) + d.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S)) + dy .^ 2 .* (a.C .* (-2 * D.D11E .* c.E - 2 * D.D11W .* c.W + 4 * D.D12E .* c.E + 4 * D.D12W .* c.W + 3 * D.D13E .* a.E + 3 * D.D13W .* a.W) + c.C .* (-2 * D.D31E .* c.E - 2 * D.D31W .* c.W + 4 * D.D32E .* c.E + 4 * D.D32W .* c.W + 3 * D.D33E .* a.E + 3 * D.D33W .* a.W))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[11] = (-a.C .* (-2 * D.D11W .* c.W + 4 * D.D12W .* c.W + 3 * D.D13W .* a.W) - c.C .* (-2 * D.D31W .* c.W + 4 * D.D32W .* c.W + 3 * D.D33W .* a.W)) ./ (6 * dx .^ 2)
    v_uu[12] = (-a.C .* (-2 * D.D11E .* c.E + 4 * D.D12E .* c.E + 3 * D.D13E .* a.E) - c.C .* (-2 * D.D31E .* c.E + 4 * D.D32E .* c.E + 3 * D.D33E .* a.E)) ./ (6 * dx .^ 2)
    v_uu[13] = (-b.C .* (-2 * D.D11S .* d.S + 4 * D.D12S .* d.S + 3 * D.D13S .* b.S) - d.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S)) ./ (6 * dy .^ 2)
    v_uu[14] = 0
    v_uu[15] = (-a.C .* (-2 * D.D11W .* d.W + 4 * D.D12W .* d.W + 3 * D.D13W .* b.W) - b.C .* (-2 * D.D11S .* c.S + 4 * D.D12S .* c.S + 3 * D.D13S .* a.S) - c.C .* (-2 * D.D31W .* d.W + 4 * D.D32W .* d.W + 3 * D.D33W .* b.W) - d.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S)) ./ (6 * dx .* dy)
    v_uu[16] = (a.C .* (-2 * D.D11E .* d.E + 4 * D.D12E .* d.E + 3 * D.D13E .* b.E) + b.C .* (-2 * D.D11S .* c.S + 4 * D.D12S .* c.S + 3 * D.D13S .* a.S) + c.C .* (-2 * D.D31E .* d.E + 4 * D.D32E .* d.E + 3 * D.D33E .* b.E) + d.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S)) ./ (6 * dx .* dy)
    v_uu[17] = (a.C .* (-2 * D.D11W .* d.W + 4 * D.D12W .* d.W + 3 * D.D13W .* b.W) + b.C .* (-2 * D.D11N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D12N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D13N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N)) + c.C .* (-2 * D.D31W .* d.W + 4 * D.D32W .* d.W + 3 * D.D33W .* b.W) + d.C .* (-2 * D.D31N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D32N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D33N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N))) ./ (6 * dx .* dy)
    v_uu[18] = (-a.C .* (-2 * D.D11E .* d.E + 4 * D.D12E .* d.E + 3 * D.D13E .* b.E) - b.C .* (-2 * D.D11N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D12N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D13N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N)) - c.C .* (-2 * D.D31E .* d.E + 4 * D.D32E .* d.E + 3 * D.D33E .* b.E) - d.C .* (-2 * D.D31N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D32N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D33N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N))) ./ (6 * dx .* dy)
end

####################

@views function CoefficientsKup_Vx!(v_up, a, b, dx, dy, symmetric )
    if symmetric
        v_up[1] = -a.W ./ dx
        v_up[2] = a.E ./ dx
        v_up[3] = -b.S ./ dy
        v_up[4] = b.N ./ dy
    else
        v_up[1] = -a.C./dx
        v_up[2] = a.C./dx
        v_up[3] = -b.C./dy
        v_up[4] = b.C./dy
    end
end

@views function CoefficientsKup_Vxv_FreeSurf!(v_up, a, b, c, d, D, fs, dx, dy, symmetric )
    v_up[1] = (a.C .* (2 * D.D11W .* (2 * b.W .* fs.C0W - fs.D0W .* d.W) - 2 * D.D12W .* (b.W .* fs.C0W - 2 * fs.D0W .* d.W) + 3 * D.D13W .* (b.W .* fs.D0W + fs.C0W .* d.W)) - 6 * a.C + c.C .* (2 * D.D31W .* (2 * b.W .* fs.C0W - fs.D0W .* d.W) - 2 * D.D32W .* (b.W .* fs.C0W - 2 * fs.D0W .* d.W) + 3 * D.D33W .* (b.W .* fs.D0W + fs.C0W .* d.W))) ./ (6 * dx)
    v_up[2] = (-a.C .* (2 * D.D11E .* (2 * b.E .* fs.C0E - fs.D0E .* d.E) - 2 * D.D12E .* (b.E .* fs.C0E - 2 * fs.D0E .* d.E) + 3 * D.D13E .* (b.E .* fs.D0E + fs.C0E .* d.E)) + 6 * a.C - c.C .* (2 * D.D31E .* (2 * b.E .* fs.C0E - fs.D0E .* d.E) - 2 * D.D32E .* (b.E .* fs.C0E - 2 * fs.D0E .* d.E) + 3 * D.D33E .* (b.E .* fs.D0E + fs.C0E .* d.E))) ./ (6 * dx)
    v_up[3] = -b.C ./ dy
    v_up[4] = b.C ./ dy
end

@views function CoefficientsKup_Vxc_FreeSurf!(v_up, a, b, c, d, D, fs, dx, dy, symmetric )
    v_up[1] = -a.C ./ dx
    v_up[2] = a.C ./ dx
    v_up[3] = -b.C ./ dy
    v_up[4] = (-b.C .* (2 * D.D11N .* (2 * b.N .* fs.C0N - fs.D0N .* d.N) - 2 * D.D12N .* (b.N .* fs.C0N - 2 * fs.D0N .* d.N) + 3 * D.D13N .* (b.N .* fs.D0N + fs.C0N .* d.N)) + 6 * b.C - d.C .* (2 * D.D31N .* (2 * b.N .* fs.C0N - fs.D0N .* d.N) - 2 * D.D32N .* (b.N .* fs.C0N - 2 * fs.D0N .* d.N) + 3 * D.D33N .* (b.N .* fs.D0N + fs.C0N .* d.N))) ./ (6 * dy)
end
####################

# Same as V part?
@views function CoefficientsKuu_Vy!(v_uu, a, b, c, d, D, dx, dy, symmetric )
    if symmetric
        v_uu[1] = (dx .^ 2 .* (-2 * D.D22N .* b.N .* d.N - 2 * D.D22S .* b.S .* d.S + 3 * D.D33N .* b.N .* d.N + 3 * D.D33S .* b.S .* d.S) + dy .^ 2 .* (-2 * D.D22E .* a.E .* c.E - 2 * D.D22W .* a.W .* c.W + 3 * D.D33E .* a.E .* c.E + 3 * D.D33W .* a.W .* c.W)) ./ (6 * dx .^ 2 .* dy .^ 2)
        v_uu[2] = a.W .* c.W .* (2 * D.D22W - 3 * D.D33W) ./ (6 * dx .^ 2)
        v_uu[3] = a.E .* c.E .* (2 * D.D22E - 3 * D.D33E) ./ (6 * dx .^ 2)
        v_uu[4] = b.S .* d.S .* (2 * D.D22S - 3 * D.D33S) ./ (6 * dy .^ 2)
        v_uu[5] = b.N .* d.N .* (2 * D.D22N - 3 * D.D33N) ./ (6 * dy .^ 2)
        v_uu[6] = (D.D22S .* a.S .* d.S / 3 + D.D22W .* b.W .* c.W / 3 - D.D33S .* b.S .* c.S / 2 - D.D33W .* a.W .* d.W / 2) ./ (dx .* dy)
        v_uu[7] = (-D.D22E .* b.E .* c.E / 3 - D.D22S .* a.S .* d.S / 3 + D.D33E .* a.E .* d.E / 2 + D.D33S .* b.S .* c.S / 2) ./ (dx .* dy)
        v_uu[8] = (-D.D22N .* a.N .* d.N / 3 - D.D22W .* b.W .* c.W / 3 + D.D33N .* b.N .* c.N / 2 + D.D33W .* a.W .* d.W / 2) ./ (dx .* dy)
        v_uu[9] = (D.D22E .* b.E .* c.E / 3 + D.D22N .* a.N .* d.N / 3 - D.D33E .* a.E .* d.E / 2 - D.D33N .* b.N .* c.N / 2) ./ (dx .* dy)
        v_uu[10] = (dx .^ 2 .* (4 * D.D22N .* d.N .^ 2 + 4 * D.D22S .* d.S .^ 2 + 3 * D.D33N .* b.N .^ 2 + 3 * D.D33S .* b.S .^ 2) + dy .^ 2 .* (4 * D.D22E .* c.E .^ 2 + 4 * D.D22W .* c.W .^ 2 + 3 * D.D33E .* a.E .^ 2 + 3 * D.D33W .* a.W .^ 2)) ./ (6 * dx .^ 2 .* dy .^ 2)
        v_uu[11] = (-4 * D.D22W .* c.W .^ 2 - 3 * D.D33W .* a.W .^ 2) ./ (6 * dx .^ 2)
        v_uu[12] = (-4 * D.D22E .* c.E .^ 2 - 3 * D.D33E .* a.E .^ 2) ./ (6 * dx .^ 2)
        v_uu[13] = (-4 * D.D22S .* d.S .^ 2 - 3 * D.D33S .* b.S .^ 2) ./ (6 * dy .^ 2)
        v_uu[14] = (-4 * D.D22N .* d.N .^ 2 - 3 * D.D33N .* b.N .^ 2) ./ (6 * dy .^ 2)
        v_uu[15] = (-4 * D.D22S .* c.S .* d.S - 4 * D.D22W .* c.W .* d.W - 3 * D.D33S .* a.S .* b.S - 3 * D.D33W .* a.W .* b.W) ./ (6 * dx .* dy)
        v_uu[16] = (4 * D.D22E .* c.E .* d.E + 4 * D.D22S .* c.S .* d.S + 3 * D.D33E .* a.E .* b.E + 3 * D.D33S .* a.S .* b.S) ./ (6 * dx .* dy)
        v_uu[17] = (4 * D.D22N .* c.N .* d.N + 4 * D.D22W .* c.W .* d.W + 3 * D.D33N .* a.N .* b.N + 3 * D.D33W .* a.W .* b.W) ./ (6 * dx .* dy)
        v_uu[18] = (-4 * D.D22E .* c.E .* d.E - 4 * D.D22N .* c.N .* d.N - 3 * D.D33E .* a.E .* b.E - 3 * D.D33N .* a.N .* b.N) ./ (6 * dx .* dy)
    else
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
end

@views function CoefficientsKuu_Vyv_FreeSurf!(v_uu, a, b, c, d, D, fs, dx, dy, symmetric )
    v_uu[1] = (dx .^ 2 .* (b.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S) + d.C .* (4 * D.D21S .* b.S - 2 * D.D22S .* b.S + 3 * D.D23S .* d.S)) + dy .^ 2 .* (a.C .* (2 * D.D31E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) + 2 * D.D31W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D32E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) - 2 * D.D32W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D33E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E) + 3 * D.D33W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)) + c.C .* (2 * D.D21E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) + 2 * D.D21W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D22E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) - 2 * D.D22W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D23E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E) + 3 * D.D23W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[2] = (-a.C .* (2 * D.D31W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D32W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D33W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W)) - c.C .* (2 * D.D21W .* (2 * a.W + 2 * b.W .* fs.C1W - fs.D1W .* d.W) - 2 * D.D22W .* (a.W + b.W .* fs.C1W - 2 * fs.D1W .* d.W) + 3 * D.D23W .* (b.W .* fs.D1W + fs.C1W .* d.W + c.W))) ./ (6 * dx .^ 2)
    v_uu[3] = (-a.C .* (2 * D.D31E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) - 2 * D.D32E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) + 3 * D.D33E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E)) - c.C .* (2 * D.D21E .* (2 * a.E + 2 * b.E .* fs.C1E - fs.D1E .* d.E) - 2 * D.D22E .* (a.E + b.E .* fs.C1E - 2 * fs.D1E .* d.E) + 3 * D.D23E .* (b.E .* fs.D1E + fs.C1E .* d.E + c.E))) ./ (6 * dx .^ 2)
    v_uu[4] = (-b.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S) - d.C .* (4 * D.D21S .* b.S - 2 * D.D22S .* b.S + 3 * D.D23S .* d.S)) ./ (6 * dy .^ 2)
    v_uu[5] = 0
    v_uu[6] = (-b.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S) - d.C .* (4 * D.D21S .* a.S - 2 * D.D22S .* a.S + 3 * D.D23S .* c.S)) ./ (6 * dx .* dy)
    v_uu[7] = (b.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S) + d.C .* (4 * D.D21S .* a.S - 2 * D.D22S .* a.S + 3 * D.D23S .* c.S)) ./ (6 * dx .* dy)
    v_uu[8] = 0
    v_uu[9] = 0
    v_uu[10] = (dx .^ 2 .* (b.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S) + d.C .* (-2 * D.D21S .* d.S + 4 * D.D22S .* d.S + 3 * D.D23S .* b.S)) + dy .^ 2 .* (a.C .* (-2 * D.D31E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) - 2 * D.D31W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D32E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 2 * D.D32W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D33E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E) + 3 * D.D33W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)) + c.C .* (-2 * D.D21E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) - 2 * D.D21W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D22E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 2 * D.D22W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D23E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E) + 3 * D.D23W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[11] = (-a.C .* (-2 * D.D31W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D32W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D33W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W)) - c.C .* (-2 * D.D21W .* (-2 * b.W .* fs.C2W + c.W + fs.D2W .* d.W) + 2 * D.D22W .* (-b.W .* fs.C2W + 2 * c.W + 2 * fs.D2W .* d.W) + 3 * D.D23W .* (a.W + b.W .* fs.D2W + fs.C2W .* d.W))) ./ (6 * dx .^ 2)
    v_uu[12] = (-a.C .* (-2 * D.D31E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) + 2 * D.D32E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 3 * D.D33E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E)) - c.C .* (-2 * D.D21E .* (-2 * b.E .* fs.C2E + c.E + fs.D2E .* d.E) + 2 * D.D22E .* (-b.E .* fs.C2E + 2 * c.E + 2 * fs.D2E .* d.E) + 3 * D.D23E .* (a.E + b.E .* fs.D2E + fs.C2E .* d.E))) ./ (6 * dx .^ 2)
    v_uu[13] = (-b.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S) - d.C .* (-2 * D.D21S .* d.S + 4 * D.D22S .* d.S + 3 * D.D23S .* b.S)) ./ (6 * dy .^ 2)
    v_uu[14] = 0
    v_uu[15] = (-b.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S) - d.C .* (-2 * D.D21S .* c.S + 4 * D.D22S .* c.S + 3 * D.D23S .* a.S)) ./ (6 * dx .* dy)
    v_uu[16] = (b.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S) + d.C .* (-2 * D.D21S .* c.S + 4 * D.D22S .* c.S + 3 * D.D23S .* a.S)) ./ (6 * dx .* dy)
    v_uu[17] = 0
    v_uu[18] = 0
end

@views function CoefficientsKuu_Vyc_FreeSurf!(v_uu, a, b, c, d, D, fs, dx, dy, symmetric )
    v_uu[1] = (dx .^ 2 .* (b.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S) + d.C .* (4 * D.D21S .* b.S - 2 * D.D22S .* b.S + 3 * D.D23S .* d.S)) + dy .^ 2 .* (a.C .* (4 * D.D31E .* a.E + 4 * D.D31W .* a.W - 2 * D.D32E .* a.E - 2 * D.D32W .* a.W + 3 * D.D33E .* c.E + 3 * D.D33W .* c.W) + c.C .* (4 * D.D21E .* a.E + 4 * D.D21W .* a.W - 2 * D.D22E .* a.E - 2 * D.D22W .* a.W + 3 * D.D23E .* c.E + 3 * D.D23W .* c.W))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[2] = (-a.C .* (4 * D.D31W .* a.W - 2 * D.D32W .* a.W + 3 * D.D33W .* c.W) - c.C .* (4 * D.D21W .* a.W - 2 * D.D22W .* a.W + 3 * D.D23W .* c.W)) ./ (6 * dx .^ 2)
    v_uu[3] = (-a.C .* (4 * D.D31E .* a.E - 2 * D.D32E .* a.E + 3 * D.D33E .* c.E) - c.C .* (4 * D.D21E .* a.E - 2 * D.D22E .* a.E + 3 * D.D23E .* c.E)) ./ (6 * dx .^ 2)
    v_uu[4] = (-b.C .* (4 * D.D31S .* b.S - 2 * D.D32S .* b.S + 3 * D.D33S .* d.S) - d.C .* (4 * D.D21S .* b.S - 2 * D.D22S .* b.S + 3 * D.D23S .* d.S)) ./ (6 * dy .^ 2)
    v_uu[5] = 0
    v_uu[6] = (-a.C .* (4 * D.D31W .* b.W - 2 * D.D32W .* b.W + 3 * D.D33W .* d.W) - b.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S) - c.C .* (4 * D.D21W .* b.W - 2 * D.D22W .* b.W + 3 * D.D23W .* d.W) - d.C .* (4 * D.D21S .* a.S - 2 * D.D22S .* a.S + 3 * D.D23S .* c.S)) ./ (6 * dx .* dy)
    v_uu[7] = (a.C .* (4 * D.D31E .* b.E - 2 * D.D32E .* b.E + 3 * D.D33E .* d.E) + b.C .* (4 * D.D31S .* a.S - 2 * D.D32S .* a.S + 3 * D.D33S .* c.S) + c.C .* (4 * D.D21E .* b.E - 2 * D.D22E .* b.E + 3 * D.D23E .* d.E) + d.C .* (4 * D.D21S .* a.S - 2 * D.D22S .* a.S + 3 * D.D23S .* c.S)) ./ (6 * dx .* dy)
    v_uu[8] = (a.C .* (4 * D.D31W .* b.W - 2 * D.D32W .* b.W + 3 * D.D33W .* d.W) + b.C .* (2 * D.D31N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D32N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D33N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N)) + c.C .* (4 * D.D21W .* b.W - 2 * D.D22W .* b.W + 3 * D.D23W .* d.W) + d.C .* (2 * D.D21N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D22N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D23N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N))) ./ (6 * dx .* dy)
    v_uu[9] = (-a.C .* (4 * D.D31E .* b.E - 2 * D.D32E .* b.E + 3 * D.D33E .* d.E) - b.C .* (2 * D.D31N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D32N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D33N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N)) - c.C .* (4 * D.D21E .* b.E - 2 * D.D22E .* b.E + 3 * D.D23E .* d.E) - d.C .* (2 * D.D21N .* (2 * a.N + 2 * b.N .* fs.C1N - fs.D1N .* d.N) - 2 * D.D22N .* (a.N + b.N .* fs.C1N - 2 * fs.D1N .* d.N) + 3 * D.D23N .* (b.N .* fs.D1N + fs.C1N .* d.N + c.N))) ./ (6 * dx .* dy)
    v_uu[10] = (dx .^ 2 .* (b.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S) + d.C .* (-2 * D.D21S .* d.S + 4 * D.D22S .* d.S + 3 * D.D23S .* b.S)) + dy .^ 2 .* (a.C .* (-2 * D.D31E .* c.E - 2 * D.D31W .* c.W + 4 * D.D32E .* c.E + 4 * D.D32W .* c.W + 3 * D.D33E .* a.E + 3 * D.D33W .* a.W) + c.C .* (-2 * D.D21E .* c.E - 2 * D.D21W .* c.W + 4 * D.D22E .* c.E + 4 * D.D22W .* c.W + 3 * D.D23E .* a.E + 3 * D.D23W .* a.W))) ./ (6 * dx .^ 2 .* dy .^ 2)
    v_uu[11] = (-a.C .* (-2 * D.D31W .* c.W + 4 * D.D32W .* c.W + 3 * D.D33W .* a.W) - c.C .* (-2 * D.D21W .* c.W + 4 * D.D22W .* c.W + 3 * D.D23W .* a.W)) ./ (6 * dx .^ 2)
    v_uu[12] = (-a.C .* (-2 * D.D31E .* c.E + 4 * D.D32E .* c.E + 3 * D.D33E .* a.E) - c.C .* (-2 * D.D21E .* c.E + 4 * D.D22E .* c.E + 3 * D.D23E .* a.E)) ./ (6 * dx .^ 2)
    v_uu[13] = (-b.C .* (-2 * D.D31S .* d.S + 4 * D.D32S .* d.S + 3 * D.D33S .* b.S) - d.C .* (-2 * D.D21S .* d.S + 4 * D.D22S .* d.S + 3 * D.D23S .* b.S)) ./ (6 * dy .^ 2)
    v_uu[14] = 0
    v_uu[15] = (-a.C .* (-2 * D.D31W .* d.W + 4 * D.D32W .* d.W + 3 * D.D33W .* b.W) - b.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S) - c.C .* (-2 * D.D21W .* d.W + 4 * D.D22W .* d.W + 3 * D.D23W .* b.W) - d.C .* (-2 * D.D21S .* c.S + 4 * D.D22S .* c.S + 3 * D.D23S .* a.S)) ./ (6 * dx .* dy)
    v_uu[16] = (a.C .* (-2 * D.D31E .* d.E + 4 * D.D32E .* d.E + 3 * D.D33E .* b.E) + b.C .* (-2 * D.D31S .* c.S + 4 * D.D32S .* c.S + 3 * D.D33S .* a.S) + c.C .* (-2 * D.D21E .* d.E + 4 * D.D22E .* d.E + 3 * D.D23E .* b.E) + d.C .* (-2 * D.D21S .* c.S + 4 * D.D22S .* c.S + 3 * D.D23S .* a.S)) ./ (6 * dx .* dy)
    v_uu[17] = (a.C .* (-2 * D.D31W .* d.W + 4 * D.D32W .* d.W + 3 * D.D33W .* b.W) + b.C .* (-2 * D.D31N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D32N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D33N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N)) + c.C .* (-2 * D.D21W .* d.W + 4 * D.D22W .* d.W + 3 * D.D23W .* b.W) + d.C .* (-2 * D.D21N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D22N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D23N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N))) ./ (6 * dx .* dy)
    v_uu[18] = (-a.C .* (-2 * D.D31E .* d.E + 4 * D.D32E .* d.E + 3 * D.D33E .* b.E) - b.C .* (-2 * D.D31N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D32N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D33N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N)) - c.C .* (-2 * D.D21E .* d.E + 4 * D.D22E .* d.E + 3 * D.D23E .* b.E) - d.C .* (-2 * D.D21N .* (-2 * b.N .* fs.C2N + c.N + fs.D2N .* d.N) + 2 * D.D22N .* (-b.N .* fs.C2N + 2 * c.N + 2 * fs.D2N .* d.N) + 3 * D.D23N .* (a.N + b.N .* fs.D2N + fs.C2N .* d.N))) ./ (6 * dx .* dy)
end
####################

@views function CoefficientsKup_Vy!(v_up, c, d, dx, dy, symmetric )
    if symmetric
        v_up[1] = -c.W ./ dx
        v_up[2] = c.E ./ dx
        v_up[3] = -d.S ./ dy
        v_up[4] = d.N ./ dy
    else
        v_up[1] = -c.C./dx
        v_up[2] = c.C./dx
        v_up[3] = -d.C./dy
        v_up[4] = d.C./dy
    end
end

@views function CoefficientsKup_Vyv_FreeSurf!(v_up, a, b, c, d, D, fs, dx, dy, symmetric )
    v_up[1] = (a.C .* (2 * D.D31W .* (2 * b.W .* fs.C0W - fs.D0W .* d.W) - 2 * D.D32W .* (b.W .* fs.C0W - 2 * fs.D0W .* d.W) + 3 * D.D33W .* (b.W .* fs.D0W + fs.C0W .* d.W)) + c.C .* (2 * D.D21W .* (2 * b.W .* fs.C0W - fs.D0W .* d.W) - 2 * D.D22W .* (b.W .* fs.C0W - 2 * fs.D0W .* d.W) + 3 * D.D23W .* (b.W .* fs.D0W + fs.C0W .* d.W)) - 6 * c.C) ./ (6 * dx)
    v_up[2] = (-a.C .* (2 * D.D31E .* (2 * b.E .* fs.C0E - fs.D0E .* d.E) - 2 * D.D32E .* (b.E .* fs.C0E - 2 * fs.D0E .* d.E) + 3 * D.D33E .* (b.E .* fs.D0E + fs.C0E .* d.E)) - c.C .* (2 * D.D21E .* (2 * b.E .* fs.C0E - fs.D0E .* d.E) - 2 * D.D22E .* (b.E .* fs.C0E - 2 * fs.D0E .* d.E) + 3 * D.D23E .* (b.E .* fs.D0E + fs.C0E .* d.E)) + 6 * c.C) ./ (6 * dx)
    v_up[3] = -d.C ./ dy
    v_up[4] = d.C ./ dy                                               
end

@views function CoefficientsKup_Vyc_FreeSurf!(v_up, a, b, c, d, D, fs, dx, dy, symmetric )
    v_up[1] = -c.C ./ dx
    v_up[2] = c.C ./ dx
    v_up[3] = -d.C ./ dy
    v_up[4] = (-b.C .* (2 * D.D31N .* (2 * b.N .* fs.C0N - fs.D0N .* d.N) - 2 * D.D32N .* (b.N .* fs.C0N - 2 * fs.D0N .* d.N) + 3 * D.D33N .* (b.N .* fs.D0N + fs.C0N .* d.N)) - d.C .* (2 * D.D21N .* (2 * b.N .* fs.C0N - fs.D0N .* d.N) - 2 * D.D22N .* (b.N .* fs.C0N - 2 * fs.D0N .* d.N) + 3 * D.D23N .* (b.N .* fs.D0N + fs.C0N .* d.N)) + 6 * d.C) ./ (6 * dy)
end
####################

@views function CoefficientsKpu!(v_pu,aC,bC,cC,dC,K,dx,dy,dt, symmetric)
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

@views function CoefficientsKpu_FreeSurf!(v_pu,aC,bC,cC,dC,K,fs,dx,dy,dt, symmetric)
    v_pu[1] = -aC ./ dx - bC .* fs.C1C ./ dx - fs.D1C .* dC ./ dx
    v_pu[2] = aC ./ dx + bC .* fs.C1C ./ dx + fs.D1C .* dC ./ dx
    v_pu[3] = 0
    v_pu[4] = 0
    v_pu[5] = -bC .* fs.C2C ./ dx - cC ./ dx - fs.D2C .* dC ./ dx
    v_pu[6] = bC .* fs.C2C ./ dx + cC ./ dx + fs.D2C .* dC ./ dx
    v_pu[7] = 0
    v_pu[8] = 0
    v_pu[9] = bC .* fs.C0C + fs.D0C .* dC + 1 ./ (K .* dt)
end

@views function AssembleKuuKupKpu!(Kuu, Kup, Kpu, Kpp, Kppi, Num, BC, D, β, K, ∂ξ, ∂η, Δ, nc, nv, penalty, comp, symmetric)
    i_uu    = zeros(Int, 18); t_uu    = zeros(Int, 18); v_uu   = ones(18)
    i_up    = zeros(Int,  4); t_up    = zeros(Int,  4); v_up   = ones( 4)
    i_pu    = zeros(Int,  9); t_pu    = zeros(Int,  9); v_pu   = ones( 9)
    # Fake compressebility
    # K = 1e20
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
        b = (C=∂η.∂x.v[i,j], W=∂η.∂x.ex[i-1,j], E=∂η.∂x.ex[i,j], S=∂η.∂x.ey[i,j-1], N=∂η.∂x.ey[i,j])
        c = (C=∂ξ.∂y.v[i,j], W=∂ξ.∂y.ex[i-1,j], E=∂ξ.∂y.ex[i,j], S=∂ξ.∂y.ey[i,j-1], N=∂ξ.∂y.ey[i,j])
        d = (C=∂η.∂y.v[i,j], W=∂η.∂y.ex[i-1,j], E=∂η.∂y.ex[i,j], S=∂η.∂y.ey[i,j-1], N=∂η.∂y.ey[i,j])
        #--------------
        # Vx
        if BC.x.v[i,j] == 0
            CoefficientsKuu_Vx!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vx!(v_up, a, b, Δ.x, Δ.y, symmetric )
        elseif BC.x.v[i,j] == 2
            fs = (C0W=BC.C0[i-1], C1W=BC.C1[i-1], C2W=BC.C2[i-1], D0W=BC.D0[i-1], D1W=BC.D1[i-1], D2W=BC.D2[i-1],
                  C0E=BC.C0[i],   C1E=BC.C1[i],   C2E=BC.C2[i],   D0E=BC.D0[i],   D1E=BC.D1[i],   D2E=BC.D2[i])
            CoefficientsKuu_Vxv_FreeSurf!(v_uu, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vxv_FreeSurf!(v_up, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
        end
        # Dirichlets
        if BC.x.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[1], i_uu[1], BC.x.v[i,j], t_uu[1], 1.0)
        else
            c_sym = CoeffSymmetry( a, b, c, d, symmetric )
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[1], i_uu[ii], BC.x.v[i,j],  t_uu[ii],  v_uu[ii] * c_sym) 
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[1], i_up[ii], BC.x.v[i,j],  t_up[ii],  v_up[ii] * c_sym)
            end
        end
        #--------------
        # Vy
        if BC.y.v[i,j] == 0
            CoefficientsKuu_Vy!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vy!(v_up, c, d, Δ.x, Δ.y, symmetric )
        elseif BC.x.v[i,j] == 2
            fs = (C0W=BC.C0[i-1], C1W=BC.C1[i-1], C2W=BC.C2[i-1], D0W=BC.D0[i-1], D1W=BC.D1[i-1], D2W=BC.D2[i-1],
                  C0E=BC.C0[i],   C1E=BC.C1[i],   C2E=BC.C2[i],   D0E=BC.D0[i],   D1E=BC.D1[i],   D2E=BC.D2[i])
            CoefficientsKuu_Vyv_FreeSurf!(v_uu, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vyv_FreeSurf!(v_up, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
        end
        # Dirichlets
        if BC.y.v[i,j] == 1
            AddToExtSparse!(Kuu, i_uu[10], i_uu[10], BC.y.v[i,j], t_uu[10], 1.0)
        else
            c_sym = CoeffSymmetry( a, b, c, d, symmetric )
            for ii in eachindex(v_uu)
                AddToExtSparse!(Kuu, i_uu[10], i_uu[ii], BC.y.v[i,j],  t_uu[ii],  v_uu[ii] * c_sym)
            end
            #--------------
            for ii in eachindex(v_up)
                AddToExtSparse!(Kup, i_uu[10], i_up[ii], BC.y.v[i,j],  t_up[ii],  v_up[ii] * c_sym)
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
        b = (C=∂η.∂x.c[i,j], W=∂η.∂x.ey[i,j], E=∂η.∂x.ey[i+1,j], S=∂η.∂x.ex[i,j], N=∂η.∂x.ex[i,j+1])
        c = (C=∂ξ.∂y.c[i,j], W=∂ξ.∂y.ey[i,j], E=∂ξ.∂y.ey[i+1,j], S=∂ξ.∂y.ex[i,j], N=∂ξ.∂y.ex[i,j+1])
        d = (C=∂η.∂y.c[i,j], W=∂η.∂y.ey[i,j], E=∂η.∂y.ey[i+1,j], S=∂η.∂y.ex[i,j], N=∂η.∂y.ex[i,j+1])
        #--------------
        # Vx
        if BC.x.c[i,j] == 0
            CoefficientsKuu_Vx!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vx!(v_up, a, b, Δ.x, Δ.y, symmetric )
        elseif BC.x.c[i,j] == 2
            fs = (C0N=BC.C0[i], C1N=BC.C1[i], C2N=BC.C2[i], D0N=BC.D0[i], D1N=BC.D1[i], D2N=BC.D2[i])
            CoefficientsKuu_Vxc_FreeSurf!(v_uu, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vxc_FreeSurf!(v_up, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
        end
        c_sym = CoeffSymmetry( a, b, c, d, symmetric )
        for ii in eachindex(v_uu)
            AddToExtSparse!(Kuu, i_uu[1], i_uu[ii], BC.x.c[i,j],  t_uu[ii],  v_uu[ii] * c_sym)
        end
        #--------------
        for ii in eachindex(v_up)
            AddToExtSparse!(Kup, i_uu[1], i_up[ii], BC.x.c[i,j],  t_up[ii],  v_up[ii] * c_sym)
        end
        #--------------
        # Vy
        if BC.y.c[i,j] == 0 
            CoefficientsKuu_Vy!(v_uu, a, b, c, d, D_sten, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vy!(v_up, c, d, Δ.x, Δ.y, symmetric )
        elseif BC.x.c[i,j] == 2
            fs = (C0N=BC.C0[i], C1N=BC.C1[i], C2N=BC.C2[i], D0N=BC.D0[i], D1N=BC.D1[i], D2N=BC.D2[i])
            CoefficientsKuu_Vyc_FreeSurf!(v_uu, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
            CoefficientsKup_Vyc_FreeSurf!(v_up, a, b, c, d, D_sten, fs, Δ.x, Δ.y, symmetric )
        end
        c_sym = CoeffSymmetry( a, b, c, d, symmetric )
        for ii in eachindex(v_uu)
            AddToExtSparse!(Kuu, i_uu[10], i_uu[ii], BC.y.c[i,j],  t_uu[ii],  v_uu[ii] * c_sym)
        end
        #--------------
        for ii in eachindex(v_up)
            AddToExtSparse!(Kup, i_uu[10], i_up[ii], BC.y.c[i,j],  t_up[ii],  v_up[ii] * c_sym)
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
            CoefficientsKpu!(v_pu, ∂ξ.∂x.ex[i,j], ∂η.∂x.ex[i,j], ∂ξ.∂y.ex[i,j], ∂η.∂y.ex[i,j], K.ex[i,j],  Δ.x, Δ.y, Δ.t, symmetric)
        else BC.p.ex[i,j] == 2
            fs_sten = (C0C=BC.C0[i], C1C=BC.C1[i], C2C=BC.C2[i], D0C=BC.D0[i], D1C=BC.D1[i], D2C=BC.D2[i])
            CoefficientsKpu_FreeSurf!(v_pu, ∂ξ.∂x.ex[i,j], ∂η.∂x.ex[i,j], ∂ξ.∂y.ex[i,j], ∂η.∂y.ex[i,j], K.ex[i,j], fs_sten, Δ.x, Δ.y, Δ.t, symmetric)
        end
        #--------------  
        for ii=1:8
            AddToExtSparse!(Kpu, i_pp, i_pu[ii], BC.p.ex[i,j],  t_pu[ii],  v_pu[ii])
        end
        if BC.p.ex[i,j] == 0
            if β.ex[i,j] 
                AddToExtSparse!(Kpp,  i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9],     v_pu[9])
                AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9], 1.0/v_pu[9]) 
            else
                AddToExtSparse!(Kpp,  i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9],          0.)
                AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9],     penalty) 
            end
        else BC.p.ex[i,j] == 2
            AddToExtSparse!(Kpp,  i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9],     v_pu[9])
            # AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9],     penalty) 
            AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ex[i,j], t_pu[9], 1.0/v_pu[9])
        end
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
        CoefficientsKpu!(v_pu, ∂ξ.∂x.ey[i,j], ∂η.∂x.ey[i,j], ∂ξ.∂y.ey[i,j], ∂η.∂y.ey[i,j], K.ey[i,j], Δ.x, Δ.y, Δ.t, symmetric)
        #--------------
        for ii=1:8
            AddToExtSparse!(Kpu, i_pp, i_pu[ii], BC.p.ey[i,j], t_pu[ii],  v_pu[ii])
        end
        if β.ey[i,j] 
            AddToExtSparse!(Kpp,  i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9],     v_pu[9])
            AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9], 1.0/v_pu[9])
        else
            AddToExtSparse!(Kpp,  i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9],          0.)
            AddToExtSparse!(Kppi, i_pp, i_pu[9], BC.p.ey[i,j], t_pu[9],     penalty)
        end
    end
    flush!(Kuu), flush!(Kup), flush!(Kpu), flush!(Kpp), flush!(Kppi)
end
