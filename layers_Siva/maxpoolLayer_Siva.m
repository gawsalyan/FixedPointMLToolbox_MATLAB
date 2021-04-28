classdef maxpoolLayer_Siva
    
    properties
      Name = 'Pool_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      
      FilterSize = 1;
      Stride = 1;
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
    end
    
    methods
        
      function obj = maxpoolLayer_Siva(filterSize,stride,name)
           obj.FilterSize = filterSize;
           obj.Stride = stride;
           %obj.No_HiddenNodes = n_H;
           obj.Name = name;
      end
      
    end
    
    methods(Static)
        
      function obj = initLayer(obj, in, options)        
          obj.Learning_Rate = options('Learning_Rate');
          obj.Weight_Factor = options('Weight_Factor');
          obj.beta = options('beta');
          obj.InputSize = in.OutputSize;
          
          obj.OutputSize = (floor((obj.InputSize - obj.FilterSize)/obj.Stride) + 1) ;
      end
      
      function obj = setLayerFilter(obj, FilterIndex, FilterValue)
              
      end
      
      function A = predict(obj, X)
        [A,~] = maxpoolFilt(obj,X);        
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        [A, noOfStrides] = maxpoolFilt(obj,X);
        memory('A') = A;
        memory('X') = X;
        memory('noOfStrides') = noOfStrides;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         noOfStrides = memory('noOfStrides');
         m_batch = memory('m_Batch');
         dLdA = reshape(dLdA,[noOfStrides, m_batch]);
         
         dLdX = zeros(size(X));
            for i = 1:noOfStrides
                dLdA_loc = reshape(dLdA(i,:),[1,m_batch]);
                workRng = (i-1)*obj.Stride+1 : (i-1) * obj.Stride + obj.FilterSize; 
                mask = (X(workRng,:) == max(X(workRng,:)));
                dLdX(workRng,:) =  dLdX(workRng,:) + mask .* dLdA_loc;
            end      
         gradsCNN = containers.Map;       
         grads = clipGradients(gradsCNN);
      end
      
      function obj = updateLayer(obj, grads)

      end
           
    end
    
end



function [A_out, noOfStrides] = maxpoolFilt(obj,X)
        [totalL,m] = size(X); 
        noOfStrides = floor((totalL - obj.FilterSize)/obj.Stride)+ 1;
        
        
            Z = zeros( noOfStrides, m);
            for i = 1 : noOfStrides
                workRng = (i-1)*obj.Stride+1 : (i-1)*obj.Stride + obj.FilterSize;
                Z(i,:) = max(X(workRng,:));
            end
    
        %display([size(A),totalL, noOfStrides, obj.NumFilters,m]);
        A_out = Z;
end
