function A = findPCAcoeff(sig, SB)

    S = sig;
    [lenSig, noSample] = size(S);
    [lenSB, nPCA] = size(SB);
    if lenSB ~= lenSig
        error('no signal length match between PC and Signal');
    end
    
    %Find the mean vector:
    S_m = mean(S);
        
    %Represent each sample i.e., image as a linear combination of basis vectors.
    A = zeros(noSample,nPCA);
    range = 1:noSample;
    for selBeat=range
        A(selBeat,:) = (S(:,selBeat) - S_m(selBeat))'*SB;
    end
    
end