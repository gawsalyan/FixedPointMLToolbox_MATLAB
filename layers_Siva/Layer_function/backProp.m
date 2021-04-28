function grads = backProp(layer,X, Y,memory)
    
    dLdZ2 = memory('A2') - Y;
    dLdW2 = (1/memory('m_Batch')) * ( dLdZ2 * memory('A1')');
    dLdb2 = (1/memory('m_Batch')) * sum( dLdZ2,2);  
    dLdA1 = layer('W2')' * dLdZ2;
    
    dLdZ1 = dLdA1 .* memory('A1') .* ( 1 -  memory('A1'));
    dLdW1 = (1/memory('m_Batch')) * dLdZ1 * X';
    dLdb1 = (1/memory('m_Batch')) * sum(dLdZ1,2); 
    dLdX = layer('W1')' * dLdZ1;    % added seperately not tested

    grads = containers.Map;
    grads('dLdW1') = dLdW1;
    grads('dLdb1') = dLdb1;
    grads('dLdW2') = dLdW2;
    grads('dLdb2') = dLdb2;
    grads('dLdX') = dLdX;           % added seperately not tested
    
end