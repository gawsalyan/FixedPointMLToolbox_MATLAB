function A = stdfi(Sig, FBITS, SHIFT_BIT)

n = length(Sig);
meaan = fix(sum(Sig)/n);
A = bitshift(fix(sqrt(fix(fix(sum(bitshift(fix((Sig - meaan).^2),-(FBITS - SHIFT_BIT),'int32')))/(n-1)))),SHIFT_BIT,'int32');
    

end

