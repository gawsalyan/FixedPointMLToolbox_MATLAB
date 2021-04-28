function gradients = lstm_backward_fi(dy, da, dc, caches, factor)
    
%     Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).
% 
%     Arguments:
%     da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
%     dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
%     caches -- cache storing information from the forward pass (lstm_forward)
% 
%     Returns:
%     gradients -- python dictionary containing:
%                         dx -- Gradient of inputs, of shape (n_x, m, T_x)
%                         da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
%                         dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
%                         dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
%                         dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
%                         dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
%                         dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
%                         dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
%                         dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
%                         dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)


    % Retrieve values from the first cache (t=1) of caches.
    [caches, x] = caches{:};
    
    [yt_pred, yyt, a_next, c_next, a_prev, c_prev, ftin, ft, itin, it, cctin, cct,oint, ot, xt, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by] = caches{1}{:};
       
    % Retrieve dimensions from da's and x1's shapes (?2 lines)
    [n_y, m, T_x] = size(dy);
    [n_a, m] = size(da);
    [n_x, m] = size(xt);
    
    % initialize the gradients with the right sizes (?12 lines)
    dx = zeros(n_x, m, T_x,'like',x);
    da0 = zeros(n_a, m,'like',da);
    dy_prevt = zeros(n_y, m,'like',dy);
    da_prevt = zeros(n_a, m,'like',da);
    dc_prevt = zeros(n_a, m,'like',da);
    dWf = zeros(n_a, n_a + n_x,'like',da);
    dWi = zeros(n_a, n_a + n_x,'like',da);
    dWc = zeros(n_a, n_a + n_x,'like',da);
    dWo = zeros(n_a, n_a + n_x,'like',da);
    dWy = zeros(n_y, n_a,'like',dy);
    dbf = zeros(n_a, 1,'like',da);
    dbi = zeros(n_a, 1,'like',da);
    dbc = zeros(n_a, 1,'like',da);
    dbo = zeros(n_a, 1,'like',da);
    dby = zeros(n_y, 1,'like',dy);
        
    % loop back over the whole sequence
    for t = T_x:-1:1
        % Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward_fi(dy(:,:,t) + dy_prevt,da_prevt, dc_prevt, caches{t}, factor);
        % Store or add the gradient to the parameters' previous step's gradient
        dx(:,:,t) = gradients{1};
        dWf = dWf + gradients{4};
        dWi = dWi + gradients{6};
        dWc = dWc + gradients{8};
        dWo = dWo + gradients{10};
        dWy = dWy + gradients{12};
        dbf = dbf + gradients{5};
        dbi = dbi + gradients{7};
        dbc = dbc + gradients{9};
        dbo = dbo + gradients{11};
        dby = dby + gradients{13};
        
        dy_prevt = zeros(n_y, m,'like',dy);
        da_prevt = gradients{2};
        dc_prevt = gradients{3};
    end
    
    % Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients{2};
    
    % Store the gradients in a python dictionary
    gradients = containers.Map;
    gradients('dLdX') = dx;
    gradients('dLda0') = da0;
    gradients('dLdWf') = dWf;  gradients('dLdbf') = dbf;
    gradients('dLdWi') = dWi;  gradients('dLdbi') = dbi;
    gradients('dLdWc') = dWc;  gradients('dLdbc') = dbc;
    gradients('dLdWo') = dWo;  gradients('dLdbo') = dbo;
    gradients('dLdWy') = dWy;  gradients('dLdby') = dby;
    
end