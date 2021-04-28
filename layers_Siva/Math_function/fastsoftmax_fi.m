function OUT = fastsoftmax_fi(x, factor)
%% Parse Inputs: 

narginchk(1,3) 
if nargin<4
    a = 1; 
else
    assert(isscalar(a)==1,'a must be a scalar.') 
end

if nargin<3
    c = 0; 
else
    assert(isscalar(c)==1,'c must be a scalar.') 
end

%% Perform mathematics: 
    f = log2(factor);
    ff =  factor + x;   
    den = sum(ff);
    
    [i,j] = size(den);
    for ii=1:i
        for jj = 1:j
            if den(ii,jj) == 0
                den(ii,jj) = 1;
            end
        end
    end
    
    OUT  = fix(bitshift(ff,f,'int32')./den);

end