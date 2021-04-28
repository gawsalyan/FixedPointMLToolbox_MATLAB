function A = maapstd(Sig, FBITS, SHIFT_BIT)

    meaan = fix(sum(Sig)/175);
%     if meaan >=0
%     else
%         meaan = meaan+1;
%     end

    %SD = floor(std(Sig));
    
    SDD = bitshift(fix(sqrt(fix(fix(sum(bitshift(fix((Sig - meaan).^2),-(FBITS - SHIFT_BIT),'int32')))/174))),SHIFT_BIT,'int32');
    
    A = fix(bitshift((Sig - meaan),FBITS,'int32')./SDD);
%     for ii = 1:length(A)
%         if A(ii) >=0
%         else
%             A(ii) = A(ii)+1;
%         end
%     end
    
end