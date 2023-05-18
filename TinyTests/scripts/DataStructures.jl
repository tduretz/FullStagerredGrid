function TensorFSG(DAT, device, ncx, ncy)
    Tensor = (;
    xx = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    yy = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    xy = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    yx = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    )
    [fill!(field, DAT(0.0)) for tuple in Tensor for field in tuple]
    return Tensor
end

function Tensor4FSG(DAT, device, ncx, ncy, val)
    Tensor = (;
    d11 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d12 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d13 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d14 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d21 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d22 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d23 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d24 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d31 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d32 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d33 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d34 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d41 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d42 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d43 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    d44 = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
    )
    [fill!(field, DAT(0.0)) for tuple in Tensor for field in tuple]
    fill!(Tensor.d11.x, DAT(val))
    fill!(Tensor.d22.x, DAT(val))
    fill!(Tensor.d33.x, DAT(val))
    fill!(Tensor.d44.x, DAT(val))
    fill!(Tensor.d11.y, DAT(val))
    fill!(Tensor.d22.y, DAT(val))
    fill!(Tensor.d33.y, DAT(val))
    fill!(Tensor.d44.y, DAT(val))
    return Tensor
end

function VectorFSG(DAT, device, ncx, ncy, pos, val)
    if pos == :nodes
        Vector = (;
            x = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
            y = (v=device_array(DAT, device, ncx+1, ncy+1), c=device_array(DAT, device, ncx+2, ncy+2)),
        )
        #[fill!(field, DAT(val[i])) for (tuple,i) in Vector for field in tuple]
        fill!(Vector.x.v, DAT(val[1]))
        fill!(Vector.x.c, DAT(val[1]))
        fill!(Vector.y.v, DAT(val[2]))
        fill!(Vector.y.c, DAT(val[2]))
    elseif pos == :edges
        Vector = (;
            x = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
            y = (x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1)),
        )
        #[fill!(field, DAT(val[i])) for (tuple,i) in Vector for field in tuple]
        fill!(Vector.x.x, DAT(val[1]))
        fill!(Vector.x.y, DAT(val[1]))
        fill!(Vector.y.x, DAT(val[2]))
        fill!(Vector.y.y, DAT(val[2]))
    end
    return Vector
end

function ScalarFSG(DAT, device, ncx, ncy)
    Scalar = (; x=device_array(DAT, device, ncx+1, ncy+0), y=device_array(DAT, device, ncx+0, ncy+1) )
    [fill!(field, DAT(0.0)) for field in Scalar]
    return Scalar
end