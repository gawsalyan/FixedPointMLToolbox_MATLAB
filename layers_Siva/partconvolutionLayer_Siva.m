classdef partconvolutionLayer_Siva
    
    properties
      Name = 'FC_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      
      FilterSize = 1;
      Stride = 1;
      NumFilters = 1;
      W;
      b;
      V_dLdW;
      V_dLdb;
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
      
      gap;
      noOfStrides;
      
      partL = 84; 
    end
    
    methods
        
      function obj = partconvolutionLayer_Siva(partLen,filterSize,numFilters,stride,name)
           obj.FilterSize = filterSize;
           obj.NumFilters = numFilters;
           obj.Stride = stride;
           obj.partL = partLen;
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
          
          obj.W = rand(obj.NumFilters, obj.FilterSize) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.b = zeros(obj.NumFilters,1) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.V_dLdW = zeros(obj.NumFilters, obj.FilterSize);
          obj.V_dLdb = zeros(obj.NumFilters,1);
          
          obj.gap = floor((obj.InputSize-obj.FilterSize-obj.Stride)/obj.NumFilters);
          obj.noOfStrides = floor((obj.InputSize - ((obj.NumFilters-1)*obj.gap + 1) - obj.FilterSize)/obj.Stride)+1;
          obj.OutputSize = obj.noOfStrides*obj.NumFilters ;
      end
      
      function obj = setLayerFilter(obj, FilterIndex, FilterValue)
          obj.W(FilterIndex,:) = FilterValue; 
      end
      
      function A = predict(obj, X)
        [A,~] = convolFilt(obj,X);        
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        [A, noOfStrides] = convolFilt(obj,X);
        memory('A') = A;
        memory('X') = X;
        memory('noOfStrides') = noOfStrides;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         noOfStrides = memory('noOfStrides');
         m_batch = memory('m_Batch');
         [totalL,m] = size(X); 
         gap = obj.gap;
         
         dLdA = reshape(dLdA,[noOfStrides,obj.NumFilters, m_batch]);
         
         dLdX = zeros(size(X));
         dLdW = zeros(size(obj.W));
         dLdb = zeros(size(obj.b));
         for filt = 1:obj.NumFilters
            %gap = floor((totalL-obj.FilterSize-obj.Stride)/obj.NumFilters);
            for i = 1: noOfStrides
                dLdA_loc = reshape(dLdA(i,filt,:),[1,m_batch]);
                %workRng = (i-1)*obj.Stride+1 : (i-1) * obj.Stride + obj.FilterSize; 
                workRng =(filt-1)*gap +(i-1)*obj.Stride +1 : (filt-1)*gap  + (i-1)*obj.Stride + obj.FilterSize;
%                 while workRng(end) > 252
%                     workRng = workRng(1:end-1);
%                 end
                dLdX(workRng,:) =  dLdX(workRng,:) + obj.W(filt,:)' .* dLdA_loc;
                dLdW(filt,:) = dLdW(filt,:) + 1/m_batch .* sum(X(workRng,:) .* dLdA_loc,2)';
                dLdb(filt) = dLdb(filt) + 1/m_batch .* sum(dLdA_loc,2);
            end
         end        
         gradsCNN = containers.Map;
         gradsCNN('dLdW') = dLdW;
         gradsCNN('dLdb') = dLdb;
         
         grads = clipGradients(gradsCNN);
      end
      
      function obj = updateLayer(obj, grads)
         obj.V_dLdW = obj.beta * obj.V_dLdW + (1 - obj.beta) * grads('dLdW');
         obj.V_dLdb = obj.beta * obj.V_dLdb + (1 - obj.beta) * grads('dLdb');
         obj.W = obj.W - obj.Learning_Rate * obj.V_dLdW;
         obj.b = obj.b - obj.Learning_Rate * obj.V_dLdb;
      end
      
      function out  = calculatecontribution(obj,in)  
          out = obj.W\(in - obj.b);
      end
      
    end
    
end



function [A_out, noOfStrides] = convolFilt(obj,X)
        [totalL,m] = size(X); 
        gap = obj.gap;
        noOfStrides = obj.noOfStrides;
        %noOfStrides = floor((obj.partL - obj.FilterSize)/obj.Stride)+ 1;
        A = zeros(noOfStrides, obj.NumFilters, m);
        for filt = 1:obj.NumFilters
            Z = zeros( noOfStrides, m);
            for i = 1: noOfStrides
                workRng =(filt-1)*gap +(i-1)*obj.Stride +1 : (filt-1)*gap  + (i-1)*obj.Stride + obj.FilterSize;
%                 while workRng(end) > 252
%                     workRng = workRng(1:end-1);
%                 end
                Z(i,:) = sum(obj.W(filt,:)'.* X(workRng,:)) + obj.b(filt);
            end
            A(:,filt,:) = Z;
        end
        %display([size(A),totalL, noOfStrides, obj.NumFilters,m]);
        %display(A);
        A_out = reshape(A, noOfStrides * obj.NumFilters, m); 
        %display(A_out);
end



%%% for 2D Image

% function feature_maps = conv(img, conv_filter)
%     if length(size(img)) > 2 | length(size(conv_filter)) > 3 %Check if number of image channels matches the filter depth.
%         if size(img,end) ~= size(conv_filter,end)
%             display('Error: Number of channels in both image and filter must match.');
%         end
%     end
% %     if size(conv_filter,2) ~= size(conv_filter,3) % Check if filter dimensions are equal.
% %         display('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
% %     end
%     if mod(size(conv_filter,2),2) == 0 % Check if filter diemnsions are odd.
%         display('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.');
%     end
% 
%     % An empty feature map to hold the output of convolving the filter(s) with the image.
%     feature_maps = zeros(   size(img,1)-size(conv_filter,2)+1,...
%                             size(img,2) - size(conv_filter,2)+1, ... 
%                             size(conv_filter,1));
% 
%     % Convolving the image by the filter(s).
%     for filter_num = 1:size(conv_filter,1)
%         %print("Filter ", filter_num + 1)
%         curr_filter = conv_filter(filter_num, :); % getting a filter from the bank.
%         %Checking if there are mutliple channels for the single filter. If so, then each channel will convolve the image.
%         %The result of all convolutions are summed to return a single feature map.
%         if length(size(curr_filter)) > 2
% %             conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) % Array holding the sum of all feature maps.
% %             for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
% %                 conv_map = conv_map + conv_(img[:, :, ch_num], 
% %                                   curr_filter[:, :, ch_num])
% %             end
%         else % There is just a single channel in the filter.
%             conv_map = conv_(img, curr_filter);
%         end
%         
%         feature_maps(:, :, filter_num) = conv_map; % Holding feature map with the current filter.
%         
%     end
% end
% 
% function final_result = con_(img, conv_filter)
%     %https://github.com/avvineed/Convolutional-neural-network-Numpy/blob/master/CNN.ipynb
%     filter_size = size(conv_filter,2);
%     result = zeros(size(img));
%     %Looping through the image to apply the convolution operation.
%             
%     for r = filter_size/2:size(img,1)-filter_size/2+1   %for r in numpy.uint16(numpy.arange(filter_size/2.0,img.shape[0]-filter_size/2.0+1)):
%         for c = filter_size/2:size(img,2)-filter_size/2+1 %for c in numpy.uint16(numpy.arange(filter_size/2.0,img.shape[1]-filter_size/2.0+1)):
% %             Getting the current region to get multiplied with the filter. How to loop through the image and get the region based on 
% %             the image and filer sizes is the most tricky part of convolution.
%             curr_region = img(r-floor(filter_size/2):r+ceil(filter_size/2),c-floor(filter_size/2):c + ceil(filter_size/2));
%             %Element-wise multipliplication between the current region and the filter.
%             curr_result = curr_region .* conv_filter;
%             conv_sum = sum(curr_result,'all'); %Summing the result of multiplication.
%             result(r, c) = conv_sum; %Saving the summation in the convolution layer feature map.
%         end    
%     %Clipping the outliers of the result matrix.
%     final_result = result(filter_size/2: size(result,1)- (filter_size/2), (filter_size/2):size(result,2)-(filter_size/2));
%     end
% end
% 
% 
% function pool_out = pooling(feature_map, Size, Stride) % size = 2, stride = 2
%     %Preparing the output of the pooling operation.
%     pool_out = zeros(((size(feature_map,1)-Size+1)/stride+1),...
%                         ((size(feature_map,2)-Size+1)/stride+1),...
%                             size(feature_map,end));
%     for map_num = 1:size(feature_map,end)
%         r2 = 0;
%         for r = 1 : Stride : size(feature_map,1)- Size +1
%             c2 = 0;
%             for c = 1 : Stride : size(feature_map,2) - Size + 1
%                 pool_out(r2, c2, map_num) = max(feature_map(r:r+Size,  c:c+Size));
%                 c2 = c2 + 1;
%             end
%             r2 = r2 +1;
%         end
%     end 
% end
% 
% 
% function relu_out = relu(feature_map)
%     %Preparing the output of the ReLU activation function.
%     relu_out = zeros(size(feature_map));
%     for map_num = 1: size(feature_map,end)
%         for r = 1:size(feature_map,1)
%             for c = 1: size(feature_map,2)
%                 relu_out(r, c, map_num) = max(feature_map(r, c, map_num), 0);
%             end
%         end
%     end
% end