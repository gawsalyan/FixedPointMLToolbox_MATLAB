classdef multiClassLayer_Sivafi
    
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
      mV;
      factor;
      f;
      maxWeight;
    end
    
    methods
        
      function obj = multiClassLayer_Sivafi(n_H, name)
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
          obj.mV = options('BiasTrainingFactor');
          
          obj.InputSize = in.OutputSize;
          
          obj.W = round(((rand(obj.No_HiddenNodes, obj.InputSize)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize).*obj.factor);
          obj.b = zeros(obj.No_HiddenNodes,1);
          obj.V_dLdW = zeros(obj.No_HiddenNodes, obj.InputSize);
          obj.V_dLdb = zeros(obj.No_HiddenNodes,1);
          obj.OutputSize = obj.No_HiddenNodes;
      end
      
      function A = predict(obj, X)
         Z = fix(bitshift(mul32fi(obj.W,X,obj.f,4),-4,'int32') + obj.b);
         A = fastsoftmax_fi(Z,obj.factor);
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        Z = fix(bitshift(mul32fi(obj.W,X,obj.f,4),-4,'int32') + obj.b);
        A = fastsoftmax_fi(Z,obj.factor);
        
        if isnan(A(1))
            pause;
        end
        
        memory('Z') = Z;
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,Y, memory)
%          if Y(1) == 0
%              dLdZ = (A - Y);
%                 if (A(1)) < 0.4 
%                     dLdZ(1) = zeros(size(dLdZ(1)));
%                 end
%                 if (A(2)) > 0.6
%                     dLdZ(2) = zeros(size(dLdZ(2)));
%                 end
%                 if isnan(dLdZ)
%                     %dLdZ = zeros(size(dLdZ));
%                 end
%          else
%              dLdZ = ((A - Y)./obj.mV);
%                 if (A(1)) > 0.6 
%                     dLdZ(1) = zeros(size(dLdZ(1)));
%                 end
%                 if (A(2)) < 0.4
%                     dLdZ(2) = zeros(size(dLdZ(2)));
%                 end
%                 if isnan(dLdZ)
%                     %dLdZ = zeros(size(dLdZ));
%                 end
%          end  
%         if isnan(dLdZ)
%            % pause(10);
%         end
        dLdZ = (A - Y);
        
        if isnan(dLdZ(1))
            pause;
        end
        
        dLdZ = round((dLdZ.*dfastsoftmax_fi(memory('Z'),obj.factor,A))./obj.factor);
        dLdW = round(((1/memory('m_Batch')) * dLdZ * X')./obj.factor);
        dLdb = round((1/memory('m_Batch')) * sum(dLdZ,2));  
        dLdX = round((obj.W' * dLdZ)./obj.factor);
        
        if isnan(dLdW(1))
            pause(1);
        end
        if isnan(dLdX(1))
            pause(1);
        end
        
        gradsMC = containers.Map;
        gradsMC('dLdW') = dLdW;
        gradsMC('dLdb') = dLdb;
        grads = clipGradientsfi(gradsMC, obj.factor);
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
         
         obj.W = round(obj.W - obj.Learning_Rate * obj.V_dLdW);
         obj.b = round(obj.b - obj.Learning_Rate * obj.V_dLdb);
         
      end
      
      function out  = calculatecontribution(obj,in)  
          out = obj.W\(in - obj.b);
      end
      
    end
    
end