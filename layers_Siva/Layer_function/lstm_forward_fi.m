function [a, y, c, caches] = lstm_forward_fi(x, a0, params, factor)
    
%     Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).
% 
%     Arguments:
%     x -- Input data for every time-step, of shape (n_x, m, T_x).
%     a0 -- Initial hidden state, of shape (n_a, m)
%     parameters -- python dictionary containing:
%                         Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
%                         bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
%                         Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
%                         bi -- Bias of the update gate, numpy array of shape (n_a, 1)
%                         Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
%                         bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
%                         Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
%                         bo -- Bias of the output gate, numpy array of shape (n_a, 1)
%                         Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
%                         by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
%                         
%     Returns:
%     a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
%     y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
%     caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    Wf = params('Wf');  bf = params('bf'); 
    Wi = params('Wi');  bi = params('bi'); 
    Wc = params('Wc');  bc = params('bc'); 
    Wo = params('Wo');  bo = params('bo'); 
    Wy = params('Wy');  by = params('by'); 

    % Initialize "caches", which will track the list of all the caches
    caches = [];
    
    %Retrieve dimensions from shapes of x and Wy (?2 lines)
    [n_x, m, T_x] = size(x);
    [n_y, n_a] = size(Wy);
    
    % initialize "a", "c" and "y" with zeros (?3 lines)
    a = zeros(n_a, m, T_x,'like',x);
    c = zeros(n_a, m, T_x,'like',x);
    y = zeros(n_y, m, T_x,'like',x);
    
    % Initialize a_next and c_next (?2 lines)
    a_next = a0;
    c_next = zeros(size(a_next),'like',x);
    
    % loop over all time-steps
    for t = 1:T_x
        % Update next hidden state, next memory state, compute the prediction, get the cache (?1 line)
        %xt = reshape(x(:, :, t),[m,n_x]);
        [a_next, c_next, yt, cache] = lstm_cell_forward_fi(x(:, :, t), a_next, c_next, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by,factor);
        % Save the value of the new "next" hidden state in a (?1 line)
        a(:,:,t) = a_next;
        % Save the value of the prediction in y (?1 line)
        y(:,:,t) = yt;
        % Save the value of the next cell state (?1 line)
        c(:,:,t)  = c_next;
        % Append the cache into caches (?1 line)
        caches{t} = cache;

    end       
    % store values needed for backward propagation in cache
    caches = {caches, x};
end