classdef netSiva

    properties
      Name = 'NET';
      no_ofLayer = 2;
      Layers;
      maxAcc = 0;
      maxAccNet;
      TrainAcc;
    end
    
    methods       
      function netSiva = netSiva(options, varargin)
           netSiva.no_ofLayer = nargin-1;          
           netSiva.Layers = varargin; 
           
           netSiva.Layers{1} = netSiva.Layers{1}.initLayer(netSiva.Layers{1}, options);
           for i = 2:netSiva.no_ofLayer 
               netSiva.Layers{i} = netSiva.Layers{i}.initLayer(netSiva.Layers{i},netSiva.Layers{i-1},options);
           end        
            
      end   
    end
    
    methods(Static)
        
       function netSiva = training(netSiva, X1, Y, options)
           startSpot = 0;
           step = 0.1;
           costX = 0;
           maxcost = 0;
           maxAcc = 0;
           costprev = 0;
            for i = 1:options('max_Epochs')
                %range = randperm(size(Y,2));
                %if size(size(X1),2) == 2
                %    X = X1(:,range);
                %else
                %    X = X1(:,:,range);
                %end
                %Y = Y(:,range);
                X = X1;
                
                for j=1:options('batches')

                    start = (j-1)*options('mini_BatchSize') + 1;
                    stop = min(start+options('mini_BatchSize') - 1, options('totalSamples'));
                    m_batch = stop - start + 1;
                    
                    if size(size(X),2) == 3
                         X_miniBatch = X(:,:,start:stop);
                    else
                        X_miniBatch = X(:,start:stop);
                    end
                    Y_miniBatch = Y(:,start:stop);
                    
                    inputV = netSiva.Layers{1};
                    A{1} = inputV.predict(inputV, X_miniBatch);
                    for k = 2:netSiva.no_ofLayer 
                        [A{k}, memory{k}] = netSiva.Layers{k}.forward(netSiva.Layers{k},A{k-1}, m_batch);
                    end
                    
                    endLayerIndex = netSiva.no_ofLayer;
                    [dLdX{endLayerIndex},grads{endLayerIndex}] = ...
                        netSiva.Layers{endLayerIndex}.backward(...
                            netSiva.Layers{endLayerIndex},A{endLayerIndex-1},A{endLayerIndex},Y_miniBatch, memory{endLayerIndex});
                    
                    k = endLayerIndex-1;
                    while k >= 2;
                        if isnan(dLdX{k+1})
                            display("nan BP cost detected");
                            break;
                        else
                            [dLdX{k},grads{k}] = netSiva.Layers{k}.backward(netSiva.Layers{k},A{k-1},A{k},dLdX{k+1}, memory{k});
                            k = k-1;  
                        end
                    end
                    
                    ki = endLayerIndex;
                    if isnan(dLdX{ki})
                        display("nan BP cost detected");
                    else
                        %display(grads{ki}('dLdW')); display(Y_miniBatch);
                        netSiva.Layers{ki} = netSiva.Layers{ki}.updateLayer(netSiva.Layers{ki}, grads{ki});
                        for ki = endLayerIndex-1:-1:2
                            netSiva.Layers{ki} = netSiva.Layers{ki}.updateLayer(netSiva.Layers{ki}, grads{ki});
                        end
                    end
                    

                end
                
                A = predictClasses(netSiva, X);
                [~,predictions] = max(A);
                [~,labels] = max(Y);
                TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0;
                                
                cost = computeMultiClassLoss(Y, A); 
                
                netSiva.TrainAcc = TrainingAccuracy;
                
                display(['Epoch: ', num2str(i), '      Training Acc: ', num2str(TrainingAccuracy), '      Training cost: ', num2str(cost),'      df: ', num2str(costprev-cost)])
                costprev = cost;
                
                %if ( maxAcc ~= max(maxAcc, TrainingAccuracy))
                    maxAcc = max(maxAcc, TrainingAccuracy);
                %end
                
                if isnan(cost)
                    
                else
                if i==1
                    costX = cost;
                    accX = TrainingAccuracy;
                else
                    costX = [costX,cost];
                    accX = [accX, TrainingAccuracy];
                end
                
                if (i-500 > 0)
                      startSpot = i-500;
                end
                
                figure(1);
                subplot1 = subplot(2,1,1);
                plot(accX, 'r-');
                ylim(subplot1,[0 100]);
                xlim([startSpot, (i+50)]);
                title(['Accuracy : ', num2str(TrainingAccuracy), '%']);
                grid on;
                subplot2 = subplot(2,1,2);
                plot(costX, 'b-');
                title(['Computed Multi Class Loss : ', num2str(cost)]);
                  
                maxcost = max(maxcost,cost);                    
                ylim(subplot2,[0 maxcost]);
                xlim([ startSpot, (i+50)]);
                %axis([ startSpot, (i+50), 0 , maxcost]);
                grid
                drawnow;  
                
                pause(step);
                
                figure(2);
                montage({netSiva.Layers{1,2}.Wf, netSiva.Layers{1,2}.Wi, netSiva.Layers{1,2}.Wc, netSiva.Layers{1,2}.Wo, netSiva.Layers{1,2}.Wy, netSiva.Layers{1,3}.W});
                %montage({netSiva.Layers{1,2}.W, netSiva.Layers{1,3}.W});
                %montage({netSiva.Layers{1,2}.W});
                
                end
                
                if cost<1e-06  % Threshold to stop learning
                    break;
                end
                
                clear A;              

            end
            display("Done...!");
       end
    
    end

end
    