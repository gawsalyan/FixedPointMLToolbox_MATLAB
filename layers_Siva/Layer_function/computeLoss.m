function L = computeLoss(Y, Y_hat)
   m = size(Y,2);        
   L = -(1/m)*sum(Y.*log(Y_hat) + (1-Y).*log(1-Y_hat));
end

