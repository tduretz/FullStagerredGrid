
function TensorFSG(DAT, device, location, ncx, ncy)
    if location==:XY
        Tensor = (;
        xx = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1)),
        yy = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1)),
        xy = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1)),
        )
        [fill!(field, DAT(0.0)) for tuple in Tensor for field in tuple]
        return Tensor
    elseif location==:CV
        Tensor = (;
        xx = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        yy = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        xy = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        )
        [fill!(field, DAT(0.0)) for tuple in Tensor for field in tuple]
        return Tensor
    end
end

function JacobianFSG(DAT, device, ncx, ncy)
    Jacobian = (;
    ξ = (∂x = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1), v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
         ∂y = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1), v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)) ),
    η = (∂x = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1), v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
         ∂y = (x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1), v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)) ),
    )
    [fill!(field, DAT(0.0)) for tuple in Jacobian for subtuple in tuple for field in subtuple]
return Jacobian
end

function VectorFSG(DAT, device, ncx, ncy)
    Vector = (;
        x = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        y = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
    )
    [fill!(field, DAT(0.0)) for tuple in Vector for field in tuple]
    return Vector
end


function ScalarFSG(DAT, device, location, ncx, ncy)
    if location==:XY
        Scalar = (; x=device_array(DAT, device, ncx+1, ncy+2), y=device_array(DAT, device, ncx+2, ncy+1) )
        [fill!(field, DAT(0.0)) for field in Scalar]
        return Scalar
    elseif location==:CV
        Scalar = (; c=device_array(DAT, device, ncx+2, ncy+2), v=device_array(DAT, device, ncx+1, ncy+1) )
        [fill!(field, DAT(0.0)) for field in Scalar]
        return Scalar
    end
end