function y = dfasttanh_fi(x, factor)
    % dtanh(x) = 1 - tanh(x).^2;
    
    maxPr = round(0.96016 * factor);
    maxRng = round(1.92033 * factor);
    A = 0.26037;
    factor2 = factor * factor;
    
    [i,j] = size(x);
    
    for ii = 1:i
        for jj = 1:j
            if x(ii,jj) > maxRng
                y(ii,jj) = 1;   %ideally zero
            elseif x(ii,jj) > 0
                y(ii,jj) = - round(2 .* A .* (x(ii,jj) - maxRng));
            elseif x(ii,jj) < -maxRng
                y(ii,jj) = 1;          %ideally zero  
            elseif x(ii,jj) < 0
                y(ii,jj) = round(2 .* A .* (x(ii,jj) + maxRng));  
            else    
                y(ii,jj) = factor; 
            end
        end
    end
end