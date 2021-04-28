function OUT = fasttanh_fi(x,factor)
% https://github.com/rrottmann/anguita
%% Author Info:
% Gawsalyan Sivapalan

%% Parse Inputs: 

narginchk(1,4) 
if nargin<3
    a = 1; 
else
    assert(isscalar(a)==1,'a must be a scalar.') 
end

if nargin<4
    c = 0; 
else
    assert(isscalar(c)==1,'c must be a scalar.') 
end

%% Perform mathematics: 
    f = log2(factor);
    
    maxPr = round(0.96016 * factor);
    maxRng = round(1.92033 * factor);
    A = round(0.26037 * factor);
    factor2 = factor * factor;
    
    [i,j] = size(x);
    
    for ii = 1:i
        for jj = 1:j
            if x(ii,jj) > maxRng
                OUT(ii,jj) = maxPr;
            elseif x(ii,jj) > 0
                OUT(ii,jj) = bitshift(bitshift((x(ii,jj) - maxRng) .* (x(ii,jj) - maxRng),-(f-4),'int32'),-4,'int32');
                OUT(ii,jj) = maxPr - bitshift(bitshift(A .* OUT(ii,jj),-(f-4),'int32'),-4,'int32');
            elseif x(ii,jj) < -maxRng
                OUT(ii,jj) = -maxPr;  
            elseif x(ii,jj) < 0
                OUT(ii,jj) = bitshift(bitshift((x(ii,jj) + maxRng) .* (x(ii,jj) + maxRng),-(f-4),'int32'),-4,'int32');
                OUT(ii,jj) = bitshift(bitshift(A .* OUT(ii,jj),-(f-4),'int32'),-4,'int32') - maxPr;
                %OUT(ii,jj) = fix(bitshift(A .* bitshift((x(ii,jj) + maxRng) .* (x(ii,jj) + maxRng),-f,'int32'),-f,'int16')- maxPr);
            else    
                OUT(ii,jj) = 1;  % ideally 0
            end
        end
    end
        
end