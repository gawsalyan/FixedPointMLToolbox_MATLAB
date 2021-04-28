function y = dfastsoftmax_fi(x, factor, sftMx)
%%
% fastsoftmax_fi = (1+x)/sum(1+x);
% d
%%

if nargin < 3
    y = fastsoftmax_fi(x,factor);
else
    y = sftMx;
end

 coeff = ones(size(x)).*factor;
 den = coeff + x;
 [i,j] = size(den);
 for ii = 1:i
     for jj = 1:j
        if den(ii,jj) == 0
           den(ii,jj) = 1; 
        end
     end
 end
 y = round((coeff./den).*(y.*(factor-y)./factor));

 
end

