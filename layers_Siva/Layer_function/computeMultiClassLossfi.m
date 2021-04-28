function L = computeMultiClassLossfi(Y, Y_hat, factor, lossname)
   m = size(Y,2);  
   
   if nargin <2
    factor = 1;
   end
   
   if nargin > 3
   L = -(1/m)*sum(Y.*log(Y_hat)+ (1-Y).*log(1-Y_hat),'all');
   else
   L =  sum((Y_hat-Y).^2,'all')./factor;  
   end
end
