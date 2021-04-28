function  gradients = lstm_cell_backward_fi(dy_next, da_next, dc_next, cache, factor)
   
%     Implement the backward pass for the LSTM-cell (single time-step).
% 
%     Arguments:
%     da_next -- Gradients of next hidden state, of shape (n_a, m)
%     dc_next -- Gradients of next cell state, of shape (n_a, m)
%     cache -- cache storing information from the forward pass
% 
%     Returns:
%     gradients -- python dictionary containing:
%                         dxt -- Gradient of input data at time-step t, of shape (n_x, m)
%                         da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
%                         dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
%                         dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
%                         dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
%                         dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
%                         dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
%                         dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
%                         dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
%                         dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
%                         dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)

    % Retrieve information from "cache"
    [yt_pred, yyt, a_next, c_next, a_prev, c_prev, ftin, ft, itin, it, cctin, cct, oint, ot, xt, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by] = cache{:};
    
    concat = [a_prev; xt];
    
    % Retrieve dimensions from xt's and a_next's shape 
    [n_x, m] = size(xt);
    [n_a, m]= size(a_next);
    
    
    dyyt = round(dy_next.*dfastsoftmax_fi(yyt,factor)./factor);
    
    if isnan(dyyt(1))
        pause;
    end
    
    % Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10)
    da_next = round((Wy'*dyyt)./factor + da_next);
    dot = round((da_next .* fasttanh_fi(c_next,factor)./factor).* (dfastsigmoid_fi(oint, factor,ot)));
    dcct_temp = round((((da_next .* ot ./factor) .* dfasttanh_fi(c_next,factor))./factor + dc_next));
    dcct = round((((dcct_temp .* it)./factor) .* dfasttanh_fi(cctin,factor))./factor);
    dit = round((((dcct_temp .* cct)./factor) .* dfastsigmoid_fi(itin,factor,it))./factor);
    dft = round((((dcct_temp .* c_prev)./factor) .* dfastsigmoid_fi(ftin,factor,ft))./factor);

    % Compute parameters related derivatives. Use equations (11)-(14) (?8 lines    
    dWy = round((dyyt)*a_next'./factor);   %%added need to test
    dWo = round(dot*concat'./factor);
    dWc = round(dcct*concat'./factor);
    dWi = round(dit*concat'./factor);
    dWf = round(dft*concat'./factor); 
    
    if isnan(dWo)
        pause;
    end
    
    dby = round(sum(dyyt,2));   %%added need to test
    dbo = round(sum(dot, 2));
    dbc = round(sum(dcct, 2));
    dbi = round(sum(dit, 2));
    dbf = round(sum(dft, 2));    

    % Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17)    
    da_prev = round((Wf(:, 1:n_a)'*dft + Wc(:, 1:n_a)'*dcct + Wi(:, 1:n_a)'*dit + Wo(:, 1:n_a)'*dot)./factor);
    dxt = round((Wf(:, n_a+1:end)'*dft + Wc(:, n_a+1:end)'*dcct + Wi(:, n_a+1:end)'*dit + Wo(:, n_a+1:end)'*dot)./factor);
    dc_prev = round((dcct_temp .* ft)./factor);
    
    if isnan(dxt(1))
        pause;
    end
    
    % Save gradients in dictionary
    gradients = { dxt, da_prev, dc_prev, dWf, dbf,  dWi, dbi,...
                dWc, dbc,  dWo, dbo, dWy, dby};

end