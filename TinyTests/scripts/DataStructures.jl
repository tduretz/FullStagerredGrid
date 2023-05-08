function TensorFSG(DAT, device, ncx, ncy)
    Tensor = (;
    xx = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    yy = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    xy = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    )
    [fill!(field, DAT(0.0)) for tuple in Tensor for field in tuple]
    return Tensor
end

function VectorFSG(DAT, device, ncx, ncy)
    Vector = (;
        x = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        y = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
    )
    [fill!(field, DAT(0.0)) for tuple in Vector for field in tuple]
    return Vector
end

function ScalarFSG(DAT, device, ncx, ncy)
    Scalar = (; x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1) )
    [fill!(field, DAT(0.0)) for field in Scalar]
    return Scalar
end