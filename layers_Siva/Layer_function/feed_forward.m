function cache = feed_forward(layer, X)

    cache = containers.Map;
    cache('Z1') = layer('W1')*X + layer('b1');
    cache('A1') = sigmoid(cache('Z1'));
    cache('Z2') = layer('W2')*cache('A1') + layer('b2');
    expZ2 = exp(cache('Z2'));
    cache('A2') = expZ2./sum(expZ2,1);
    
end