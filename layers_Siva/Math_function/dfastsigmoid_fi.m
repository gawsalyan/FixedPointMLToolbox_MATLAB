function y = dfastsigmoid_fi(x,factor,sigX)
   
    if nargin<3
        sigX = fastsigmoid_fi(x, factor);
    end

    [i,j] = size(x);
    
    for ii = 1:i
        for jj = 1:j 
            if x(ii,jj) > 0.1*factor
                y(ii,jj) = round((sigX(ii,jj) - 0.5*factor) .* (factor - (sigX(ii,jj) - 0.5*factor))./x(ii,jj));
            elseif x(ii,jj) < - 0.1*factor
                y(ii,jj) = round((sigX(ii,jj) - 0.5*factor) .* (factor + (sigX(ii,jj) - 0.5*factor))./x(ii,jj));    
            else
                y(ii,jj) = round(0.5 * factor);    
            end    
        end
    end
    
end