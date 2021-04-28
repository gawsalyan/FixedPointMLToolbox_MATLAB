function A = wstdfi( Sig, W, FBITS, SHIFT_BIT)

n = length(Sig);
Sig(isnan(Sig))=0;
meaan = fix(sum(Sig,'omitnan')/n);
A = bitshift(fix((Sig - meaan).^2.*W),-(FBITS - SHIFT_BIT),'int64');
A = bitshift(fix(sqrt(fix(fix(sum(A,'omitnan'))/(n)))),SHIFT_BIT,'int32');
    

end