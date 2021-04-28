classdef concatVectorLayer_Sivafi
    
    properties
      Name = 'input';
      OutputSize;
      InputSize;
    end
    
    methods
        
      function obj = concatVectorLayer_Sivafi(name)
           obj.Name = name;
      end
      
    end
    
    methods(Static)
        
      function A = predict(obj, X)
         A = X;
      end
      
      function obj = initLayer(obj, in1, in2, options)        
          obj.OutputSize = in1.OutputSize + in2.OutputSize;
          obj.InputSize = obj.OutputSize;
      end 
            
    end
    
end