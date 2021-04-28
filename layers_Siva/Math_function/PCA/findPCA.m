function SB = findPCA(sig, nPCA)
%%PCA
%----------------------------------------------------------
%   S - sig[length of signal, no of sample]
%   S_m - mean across each individual sample
%%end
    
    S = sig;
    [lenSig, noSample] = size(S);
    if (noSample < nPCA)
       error(['not enough sample to calculate ', nPCA, ' no of principle components']);  
    end
       
    %Assemble the mean adjusted matrix:
    S_ma = S - mean(S);
    
    %Compute the covariance matrix:
    C = cov(S_ma);
    
    %Compute the Eigen vectors and Eigen values of the covariance matrix
    [eVec, eVal] = eig(C);
    [~,ind] = sort(diag(eVal),'descend');
    %eVal_S = eVal(ind,ind);
    eVec_S = eVec(:,ind);
    
    %Select most prominent eigen vectors
    EV = eVec_S(:,1:nPCA);
    
    %Compute the basis vectors
    SB = S_ma * EV;
    
end