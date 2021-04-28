function theMatrixProduct = mul32fi(x,y,f,s)

[rowsx, colsx] = size(x);
[rowsy, colsy] = size(y);
theMatrixProduct = zeros(rowsx, colsy);
for row = 1 : rowsx
  %row % Print progress to command window.
  for col = 1 : colsy
    theSum = 0;
    for k = 1 : colsx
      temp = x(row, k) * y(k, col);
      theSum = theSum + bitshift(int32(temp),-(f-s),'int32');
    end
    theMatrixProduct(row, col) = theSum;
  end
end

end