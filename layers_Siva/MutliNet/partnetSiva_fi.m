classdef partnetSiva_fi

    properties
      Name = 'partNET';
      no_ofLayer = 2;
      Layers;
    end
    
    methods       
      function netSiva = partnetSiva_fi(varargin)
           netSiva.no_ofLayer = nargin;           
           netSiva.Layers = varargin; 
            
      end   
    end
    
    methods(Static)
       
      function netSiva = setLearningRate(netSiva, LR)
           %netSiva.Layers{1} = netSiva.Layers{1}.setLearningRate(netSiva.Layers{1}, LR);
           for i = 2:netSiva.no_ofLayer 
               netSiva.Layers{i} = netSiva.Layers{i}.setLearningRate(netSiva.Layers{i},LR);
           end  
      end
      
       function netSiva = initNet(netSiva, options)
           netSiva.Layers{1} = netSiva.Layers{1}.initLayer(netSiva.Layers{1}, options);
           for i = 2:netSiva.no_ofLayer 
               netSiva.Layers{i} = netSiva.Layers{i}.initLayer(netSiva.Layers{i},netSiva.Layers{i-1},options);
           end             
       end
       
       function A  = predict(netSiva, X)                  
                    A{1} = netSiva.Layers{1}.predict(netSiva.Layers{1}, X);
                    for k = 2:netSiva.no_ofLayer 
                        A{k} = netSiva.Layers{k}.predict(netSiva.Layers{k},A{k-1});
                    end
       end
        
       function [A , memory] = forward(netSiva, X, m_batch)
                   
                    inputV = netSiva.Layers{1};
                    A{1} = inputV.predict(inputV, X);
                    for k = 2:netSiva.no_ofLayer 
                        [A{k}, memory{k}] = netSiva.Layers{k}.forward(netSiva.Layers{k},A{k-1}, m_batch);
                    end
       end
       
       function [dLdX,grads] = backward(netSiva, X, A, dLdA, memory)
                    endLayerIndex = netSiva.no_ofLayer;
                    [dLdX{endLayerIndex},grads{endLayerIndex}] = ...
                        netSiva.Layers{endLayerIndex}.backward(...
                            netSiva.Layers{endLayerIndex},A{endLayerIndex-1},A{endLayerIndex},dLdA, memory{endLayerIndex});
                    for k = netSiva.no_ofLayer-1:-1:2
                        [dLdX{k},grads{k}] = netSiva.Layers{k}.backward(netSiva.Layers{k},A{k-1},A{k},dLdX{k+1}, memory{k});
                    end
       end
       
       function netSiva = updateWeights(netSiva, grads)   
                    for k = netSiva.no_ofLayer:-1:2
                        netSiva.Layers{k} = netSiva.Layers{k}.updateLayer(netSiva.Layers{k}, grads{k});
                    end
       end
    
    end

end
    