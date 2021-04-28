function A = findPCAcoeff_fi(sig, SB, factor)

    S = sig;
    [lenSig, noSample] = size(S);
    [lenSB, nPCA] = size(SB);
    if lenSB ~= lenSig
        error('no signal length match between PC and Signal');
    end
    
    %Find the mean vector:
    S_m = fix(mean(S));
        
    %Represent each sample i.e., image as a linear combination of basis vectors.
    A = zeros(noSample,nPCA);
    range = 1:noSample;
    for selBeat=range
        temp = (((S(:,selBeat) - S_m(selBeat)).*SB));
        temp = bitshift(temp,-8,'int32'); %(temp./(factor/2^4));
        A(selBeat,:) = bitshift(sum(temp),-4,'int32'); %fix(fix(sum(temp))/2^4);
    end
    
end