classdef fullyConnectedLayer_Sivafi
    
    properties
      Name = 'FC_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      W;
      b;
      V_dLdW;
      V_dLdb;
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
      factor;
      f;
      maxWeight;
    end
    
    methods
        
      function obj = fullyConnectedLayer_Sivafi(n_H, name)
           obj.No_HiddenNodes = n_H;
           obj.Name = name;
      end
      
    end
    
    methods(Static)
      
      function obj = setLearningRate(obj, LR)
            obj.Learning_Rate = LR;
      end
        
      function obj = initLayer(obj, in, options)        
          obj.Learning_Rate = options('Learning_Rate');
          obj.Weight_Factor = options('Weight_Factor');
          obj.factor = options('Fraction_Factor');
          obj.f = log2(obj.factor);
          obj.maxWeight = options('Max_Weigh');
          obj.beta = options('beta');
          
          obj.InputSize = in.OutputSize;
          
          obj.W = round(((rand(obj.No_HiddenNodes, obj.InputSize)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize).*obj.factor);
          obj.b = zeros(obj.No_HiddenNodes,1) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.V_dLdW = zeros(obj.No_HiddenNodes, obj.InputSize);
          obj.V_dLdb = zeros(obj.No_HiddenNodes,1);
          obj.OutputSize = obj.No_HiddenNodes;
      end
      
      function A = predict(obj, X)
         Z = fix(bitshift(mul32fi(obj.W,X,obj.f,4),-4,'int32') + obj.b);
         A = fastsigmoid_fi(Z,obj.factor);
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        Z = fix(bitshift(mul32fi(obj.W,X,obj.f,4),-4,'int32') + obj.b);
        A = fastsigmoid_fi(Z,obj.factor);
        
        for ii = 1:length(A)
            if isnan(A(ii)) || isinf(A(ii))
                pause;
            end
        end

        memory('Z') = Z; 
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory)
         if isnan(dLdA(1))
            pause(1);
         end
          
         dLdZ = round((dLdA .* dfastsigmoid_fi(memory('Z'),obj.factor,A))./obj.factor);
         dLdW = round(((1/memory('m_Batch')) * dLdZ * X')./obj.factor);        % n_h * n_x
         dLdb = round((1/memory('m_Batch')) * sum(dLdZ,2));      % n_h * 1
         dLdX = round((obj.W' * dLdZ)./obj.factor);
         grads = containers.Map;
         grads('dLdW') = dLdW;
         grads('dLdb') = dLdb;
         
         if isnan(dLdW(1))
            pause;
         end 
         if isnan(dLdX(1))
            pause;
         end
         
      end
      
      function obj = updateLayer(obj, grads)
         obj.V_dLdW = round(obj.beta * obj.V_dLdW + (1 - obj.beta) * grads('dLdW'));
         obj.V_dLdb = round(obj.beta * obj.V_dLdb + (1 - obj.beta) * grads('dLdb'));
         
%          tempCalc = obj.W - round(obj.Learning_Rate * obj.V_dLdW);
%          if abs(tempCalc) < obj.maxWeight
%             obj.W = tempCalc;
%          end
%          tempCalc = obj.b - round(obj.Learning_Rate * obj.V_dLdb);
%          if abs(tempCalc) < obj.maxWeight
%             obj.b = tempCalc;
%          end
          
         obj.W = obj.W - round(obj.Learning_Rate * obj.V_dLdW);
         obj.b = obj.b - round(obj.Learning_Rate * obj.V_dLdb);
      end
      
      function out  = calculatecontribution(obj,in)  
          out = obj.W\(in - obj.b);
      end
      
    end
    
end