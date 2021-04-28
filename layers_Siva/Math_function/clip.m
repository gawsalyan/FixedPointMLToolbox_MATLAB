function xclp = clip(x,xmin,xmax)
    xclp = x;
    xclp(x>=xmax) = xmax;
    xclp(x<=xmin) = xmin;
end
%varargout{k+1} = find(n); % returns indices