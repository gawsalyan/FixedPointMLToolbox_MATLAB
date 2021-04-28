classdef lstmLayer_Sivafi
    
    properties
      Name = 'LSTM_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
      factor;       %fraction
      maxWeight;
      
      n_a = 1;   % No of features in Activation vector (A and C)
      n_f = 1;   % No of elements in input X vector i.e. it will represent the onehotvector 
      n_x = 1;   % No of features in input X vector
      n_y = 1;   % No of feature in output Y vector
      Ts = 10;
      
      Wf; Wi; Wc; Wo; Wy;
      bf; bi; bc; bo; by;
      
      a0;
      dy; da; dc;
      
      V_dLdWf; V_dLdWi; V_dLdWc; V_dLdWo; V_dLdWy;
      V_dLdbf; V_dLdbi; V_dLdbc; V_dLdbo; V_dLdby;
      
    end
    
    methods
        
      function obj = lstmLayer_Sivafi(n_a,n_x, n_y, name)
           obj.No_HiddenNodes = n_a;
           obj.n_a = n_a;
           obj.n_x = n_x;
           obj.n_y = n_y;
           %obj.Ts = Ts;
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
          obj.beta = options('beta');
          obj.maxWeight = options('Max_Weigh');
          
          obj.InputSize = in.OutputSize(1);
          obj.Ts = obj.InputSize;
                    
          obj.Wf = round(((rand(obj.n_a, obj.n_a + obj.n_x)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize) * obj.factor);
          obj.bf = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wi = round(((rand(obj.n_a, obj.n_a + obj.n_x)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize)* obj.factor);
          obj.bi = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wc = round(((rand(obj.n_a, obj.n_a + obj.n_x)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize)* obj.factor);
          obj.bc = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wo = round(((rand(obj.n_a, obj.n_a + obj.n_x)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize)* obj.factor);
          obj.bo = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wy = round(((rand(obj.n_y, obj.n_a)-0.5)*2) * obj.Weight_Factor * sqrt(1/obj.InputSize)* obj.factor);
          obj.by = zeros(obj.n_y, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          
          obj.V_dLdWf = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbf = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWi = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbi = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWc = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbc = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWo = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbo = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWy = zeros(obj.n_y, obj.n_a);
          obj.V_dLdby = zeros(obj.n_y, obj.n_f);
          
          obj.a0 = zeros(obj.n_a,obj.n_f);
          obj.da = zeros(obj.n_a,obj.n_f); 
          obj.dc = zeros(obj.n_a,obj.n_f);
          
          obj.OutputSize = obj.n_y * obj.n_f * obj.Ts;  %%need attentom
      end
      
      function A = predict(obj, X)  
        sizeX = size(X);
        if size(sizeX,2) > 2
            m_batch = sizeX(end);
        else
            m_batch = 1;
        end
        
        x = reshape(X,[obj.n_x, obj.n_f, obj.Ts, m_batch]);
        y = zeros(obj.n_y*obj.n_f, obj.Ts ,m_batch);
            for i = 1:m_batch
                xt_lstm  = reshape(x(:,:,:,i),[obj.n_x, obj.n_f, obj.Ts]);
                params = createLSTMPara(obj);
                [a, yloc, c, cachesloc] = lstm_forward_fi(xt_lstm, obj.a0, params, obj.factor);
                y(:,:,i) = reshape(yloc,[obj.n_y*obj.n_f, obj.Ts, 1]);
            end                        
         A = reshape(y,[obj.n_y*obj.n_f*obj.Ts, m_batch]);
      end
      
      function [A, memory] = forward(obj,X, m_batch)
       x = reshape(X,[obj.n_x, obj.n_f, obj.Ts, m_batch]);
       y = zeros(obj.n_y*obj.n_f, obj.Ts ,m_batch);
            for i = 1:m_batch
                xt_lstm  = reshape(x(:,:,:,i),[obj.n_x, obj.n_f, obj.Ts]);
                params = createLSTMPara(obj);
                [a, yloc, c, cachesloc] = lstm_forward_fi(xt_lstm, obj.a0, params, obj.factor);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                y(:,:,i) = reshape(yloc,[obj.n_y*obj.n_f, obj.Ts, 1]);
                memory{i} = cachesloc;
            end
         A = reshape(y,[obj.n_y*obj.n_f*obj.Ts, m_batch]);
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         sizeA = size(A);
         m_batch = sizeA(end);
         dy_batch = reshape(dLdA,[obj.n_y, obj.n_f, obj.Ts, m_batch]);
         
         dy = reshape(dy_batch(:,:,:,1),[obj.n_y,obj.n_f,obj.Ts]);
         gradLSTM = lstm_backward_fi(dy,obj.da,obj.dc, memory{1}, obj.factor);
         gradLSTM('dLdX')  = reshape(gradLSTM('dLdX'),[obj.n_x, obj.Ts]);
             for i = 2:m_batch
                dy = reshape(dy_batch(:,:,:,i),[obj.n_y,obj.n_f,obj.Ts]);
                gradients = lstm_backward_fi(dy,obj.da,obj.dc, memory{i},obj.factor);
                gradLSTM('dLdX') = [gradLSTM('dLdX')  reshape(gradients('dLdX'),[obj.n_x, obj.Ts])];
                gradLSTM('dLda0') = gradLSTM('dLda0') + gradients('dLda0');
                gradLSTM('dLdWf') = gradLSTM('dLdWf') + gradients('dLdWf');  
                gradLSTM('dLdbf') = gradLSTM('dLdbf') + gradients('dLdbf');
                gradLSTM('dLdWi') = gradLSTM('dLdWi') + gradients('dLdWi'); 
                gradLSTM('dLdbi') = gradLSTM('dLdbi') + gradients('dLdbi');
                gradLSTM('dLdWc') = gradLSTM('dLdWc') + gradients('dLdWc');
                gradLSTM('dLdbc') = gradLSTM('dLdbc') + gradients('dLdbc');
                gradLSTM('dLdWo') = gradLSTM('dLdWo') + gradients('dLdWo');  
                gradLSTM('dLdbo') = gradLSTM('dLdbo') + gradients('dLdbo');
                gradLSTM('dLdWy') = gradLSTM('dLdWy') + gradients('dLdWy');  
                gradLSTM('dLdby') = gradLSTM('dLdby') + gradients('dLdby');
            end
    
            gradLSTM('dLdX') =  gradLSTM('dLdX');
            gradLSTM('dLda0') = round((1/m_batch)* gradLSTM('dLda0'));
            gradLSTM('dLdWf') = round((1/m_batch)* gradLSTM('dLdWf'));  
            gradLSTM('dLdbf') = round((1/m_batch)* gradLSTM('dLdbf'));
            gradLSTM('dLdWi') = round((1/m_batch)* gradLSTM('dLdWi'));
            gradLSTM('dLdbi') = round((1/m_batch)* gradLSTM('dLdbi'));
            gradLSTM('dLdWc') = round((1/m_batch)* gradLSTM('dLdWc'));
            gradLSTM('dLdbc') = round((1/m_batch)* gradLSTM('dLdbc'));
            gradLSTM('dLdWo') = round((1/m_batch)* gradLSTM('dLdWo'));  
            gradLSTM('dLdbo') = round((1/m_batch)* gradLSTM('dLdbo'));
            gradLSTM('dLdWy') = round((1/m_batch)* gradLSTM('dLdWy'));  
            gradLSTM('dLdby') = round((1/m_batch)* gradLSTM('dLdby'));
            
            
            grads = clipGradientsfi(gradLSTM, obj.factor);
            dLdX = gradLSTM('dLdX');  % no clipping on dLdX
          
      end
      
      function obj = updateLayer(obj, grads)
         obj.V_dLdWf = round(obj.beta * obj.V_dLdWf + (1 - obj.beta) * grads('dLdWf'));
         obj.V_dLdbf = round(obj.beta * obj.V_dLdbf + (1 - obj.beta) * grads('dLdbf'));
         obj.V_dLdWi = round(obj.beta * obj.V_dLdWi + (1 - obj.beta) * grads('dLdWi'));
         obj.V_dLdbi = round(obj.beta * obj.V_dLdbi + (1 - obj.beta) * grads('dLdbi'));
         obj.V_dLdWc = round(obj.beta * obj.V_dLdWc + (1 - obj.beta) * grads('dLdWc'));
         obj.V_dLdbc = round(obj.beta * obj.V_dLdbc + (1 - obj.beta) * grads('dLdbc'));
         obj.V_dLdWo = round(obj.beta * obj.V_dLdWo + (1 - obj.beta) * grads('dLdWo'));
         obj.V_dLdbo = round(obj.beta * obj.V_dLdbo + (1 - obj.beta) * grads('dLdbo'));
         obj.V_dLdWy = round(obj.beta * obj.V_dLdWy + (1 - obj.beta) * grads('dLdWy'));
         obj.V_dLdby = round(obj.beta * obj.V_dLdby + (1 - obj.beta) * grads('dLdby'));
         
%          tempCalc = obj.Wf - round(obj.Learning_Rate * obj.V_dLdWf);
%          if abs(tempCalc) < obj.maxWeight
%             obj.Wf = tempCalc;
%          end   
%          tempCalc = obj.bf - round(obj.Learning_Rate * obj.V_dLdbf);
%          if abs(tempCalc) < obj.maxWeight
%             obj.bf = tempCalc;
%          end   
%          tempCalc = obj.Wi - round(obj.Learning_Rate * obj.V_dLdWi);
%          if abs(tempCalc) < obj.maxWeight
%             obj.Wi = tempCalc;
%          end   
%          tempCalc = obj.bi - round(obj.Learning_Rate * obj.V_dLdbi);
%          if abs(tempCalc) < obj.maxWeight
%             obj.bi = tempCalc;
%          end   
%          tempCalc = obj.Wc - round(obj.Learning_Rate * obj.V_dLdWc);
%          if abs(tempCalc) < obj.maxWeight
%             obj.Wc = tempCalc;
%          end   
%          tempCalc = obj.bc - round(obj.Learning_Rate * obj.V_dLdbc);
%          if abs(tempCalc) < obj.maxWeight
%             obj.bc = tempCalc;
%          end   
%          tempCalc = obj.Wo - round(obj.Learning_Rate * obj.V_dLdWo);
%          if abs(tempCalc) < obj.maxWeight
%             obj.Wo = tempCalc;
%          end   
%          tempCalc = obj.bo - round(obj.Learning_Rate * obj.V_dLdbo);
%          if abs(tempCalc) < obj.maxWeight
%             obj.bo = tempCalc;
%          end   
%          tempCalc = obj.Wy - round(obj.Learning_Rate * obj.V_dLdWy);
%          if abs(tempCalc) < obj.maxWeight
%             obj.Wy = tempCalc;
%          end   
%          tempCalc = obj.by - round(obj.Learning_Rate * obj.V_dLdby);
%          if abs(tempCalc) < obj.maxWeight
%             obj.by = tempCalc;
%          end   
         obj.Wf = obj.Wf - round(obj.Learning_Rate * obj.V_dLdWf);
         obj.bf = obj.bf - round(obj.Learning_Rate * obj.V_dLdbf);
         obj.Wi = obj.Wi - round(obj.Learning_Rate * obj.V_dLdWi);
         obj.bi = obj.bi - round(obj.Learning_Rate * obj.V_dLdbi);
         obj.Wc = obj.Wc - round(obj.Learning_Rate * obj.V_dLdWc);
         obj.bc = obj.bc - round(obj.Learning_Rate * obj.V_dLdbc);
         obj.Wo = obj.Wo - round(obj.Learning_Rate * obj.V_dLdWo);
         obj.bo = obj.bo - round(obj.Learning_Rate * obj.V_dLdbo);
         obj.Wy = obj.Wy - round(obj.Learning_Rate * obj.V_dLdWy);
         obj.by = obj.by - round(obj.Learning_Rate * obj.V_dLdby);
         
      end
      
    end
    
end


function params = createLSTMPara(obj)
%Temporary function to map the net class properties to layer function
%parameters, in future layer funtions need to be changed such that it
%straight away taking in the net properties
        params = containers.Map;
                params('Wf') = obj.Wf;  params('bf') = obj.bf; 
                params('Wi') = obj.Wi;  params('bi') = obj.bi; 
                params('Wc') = obj.Wc;  params('bc') = obj.bc; 
                params('Wo') = obj.Wo;  params('bo') = obj.bo; 
                params('Wy') = obj.Wy;  params('by') = obj.by; 
end