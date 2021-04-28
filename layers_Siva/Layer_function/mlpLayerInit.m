function mlpLayer = mlpLayerInit(n_h, n_y, n_x, digits, weightFactor)

mlpLayer = containers.Map;
inputSize = n_y*n_x;
mlpLayer('W1') = rand(n_h, inputSize) * weightFactor * sqrt(1/(inputSize));
mlpLayer('b1') = zeros(n_h,1) * weightFactor * sqrt(1/inputSize);
mlpLayer('W2') = rand(digits, n_h) * weightFactor * sqrt(1/n_h);
mlpLayer('b2') = zeros(digits,1) * weightFactor * sqrt(1/n_h);

end