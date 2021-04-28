function y = dsigmoid(x)

    sigX = sigmoid(x);
    y = sigX .* (1 - sigX);

end