%% MIT-BIH  Fs_Req https://physionet.org/content/mitdb
%https://physionet.org/physiobank/database/html/mitdbdir/intro.htm
clear all;
clc;

%% 1.load the data and generate feature
% X1Train   : Features of LSTM input for Training e.g.[feature, time, samples]
% X2Train   : Features of input for Training  e.g.[feature, samples] 
% YTrain    : Labels of Training   e.g.[[1,0];[1,0];[0,1];[1,0]] in fixedpoint two classlabel
% X1Test    : Features of LSTM input for Training e.g.[feature, time, samples]
% X2Test    : Features of input for Training  e.g.[feature, samples] 
% YTest     : Labels of Testing   e.g.[[1,0];[1,0];[0,1];[1,0]] in fixedpoint two classlabel

%% 11.Artificial Neural Network
clc
options = containers.Map;
options('Fraction_Factor') = 2^12;              % 12 bit fraction
options('totalSamples') = size(YTrain,2);       % Total number of samples  
options('Learning_Rate') = 1;                   % LR: 1/options('Fraction_Factor')
options('BiasTrainingFactor') = 1;              % default 1, increase for biased training
options('TrainAccThresh') = 95;                 % Stopping criteria
options('max_Epochs') = 10;                     % Number of training epochs
options('mini_BatchSize') = 128;                % Mini Batch Size
options('batches') = (options('totalSamples') / options('mini_BatchSize'));
options('beta') = 0.9;                          % Stochastic gradient beta value
options('Max_Weigh') = 2^15;                    % defualt:2^15 for 16 bit signed word size
options('Weight_Factor') = 1;                   % default 1

%% 11.1.Create the structure
% Structure with 2 input pipeline and a blend model

%LSTM Layer 
 inputV1 = inputVectorLayer_Sivafi(5 , 'input1');   % Number of time steps
 LSTM1 = lstmLayer_Sivafi(30, 6, 2,'lstm1');        % Hidden Layer size, feature vector size, output size  
 partNet1 = partnetSiva_fi(inputV1,LSTM1);          % partnet        
 
 inputV2 = inputVectorLayer_Sivafi(5 , 'input2');   % Input feature vector size
 FL_2 = fullyConnectedLayer_Sivafi(2,'mlp1');       % Number of Hidden nodes
 partNet2 = partnetSiva_fi(inputV2,FL_2);           % partnet
 
 concat = concatVectorLayer_Sivafi('concat');       % concat
 FL_X = fullyConnectedLayer_Sivafi(10,'mlp13');     % MLP layer   
 MLclass = multiClassLayer_Sivafi(2,'class');       % multiclass layer
 classnet = classnetSiva_fi(concat,FL_X, MLclass);  % classnet
  
 net1 = multinetSiva_fi(options, partNet1, partNet2, classnet); % create and initiate the net
 
 %% 11.2.Training of ANN
for i = 1:50

    net1 = net1.training(net1, X1Train(:,:,:), X2Train(:,:), YTrain(:,:), options);

%% Test Class Accuracy
    A = predictMulitinetClassesfi(net1, X1Test(:,:,:), X2Test(:,:));
    [~,predictions] = max(A);
    [~,labels] = max(YTest(:,:));
    TestAccuracy = sum((predictions==labels))/length(labels)*100.0

%% Plot the Test confusion
    figure(1);
    plotconfusion(categorical(labels),categorical(predictions));

%% Save the model
    name = ['modelLSTMfi_',num2str(i),'_',num2str(ceil(TestAccuracy*100.0))];
    save(name,'net1')

%% Threshold increased 
% optional
    pause(1);
    options('TrainAccThresh') = options('TrainAccThresh') + 0.1;
    
end











%%....................................................................
%% Section 2
%% ...................................................................
%%

%% Optional  
%% Training Class Accuracy
 
A = predictMulitinetClassesfi(net1, X1Train(:,:,:), X2Train(:,:));
[~,predictions] = max(A);
[~,labels] = max(YTrain(:,:));
TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0

figure(2);
plotconfusion(categorical(labels),categorical(predictions));


%% Change NETS Parameter or Prune
%% .......................................................................
%%
net2 = net1;    % save original net to temporary net

%% Quantization of the net1
%%
f = 6;          % New fraction bit

net1.Nets{1,1}.Layers{1,2}.factor = round(net2.Nets{1,1}.Layers{1,2}.factor/2^f);
net1.Nets{1,1}.Layers{1,2}.Wf = round(net2.Nets{1,1}.Layers{1,2}.Wf/2^f);
net1.Nets{1,1}.Layers{1,2}.bf = round(net2.Nets{1,1}.Layers{1,2}.bf/2^f);
net1.Nets{1,1}.Layers{1,2}.Wi = round(net2.Nets{1,1}.Layers{1,2}.Wi/2^f);
net1.Nets{1,1}.Layers{1,2}.bi = round(net2.Nets{1,1}.Layers{1,2}.bi/2^f);
net1.Nets{1,1}.Layers{1,2}.Wc = round(net2.Nets{1,1}.Layers{1,2}.Wc/2^f);
net1.Nets{1,1}.Layers{1,2}.bc = round(net2.Nets{1,1}.Layers{1,2}.bc/2^f);
net1.Nets{1,1}.Layers{1,2}.Wo = round(net2.Nets{1,1}.Layers{1,2}.Wo/2^f);
net1.Nets{1,1}.Layers{1,2}.bo = round(net2.Nets{1,1}.Layers{1,2}.bo/2^f);
net1.Nets{1,1}.Layers{1,2}.Wy = round(net2.Nets{1,1}.Layers{1,2}.Wy/2^f);
net1.Nets{1,1}.Layers{1,2}.by = round(net2.Nets{1,1}.Layers{1,2}.by/2^f);

net1.Nets{1,2}.Layers{1,2}.factor = round(net2.Nets{1,2}.Layers{1,2}.factor/2^f);
net1.Nets{1,2}.Layers{1,2}.W = round(net2.Nets{1,2}.Layers{1,2}.W/2^f);
net1.Nets{1,2}.Layers{1,2}.b = round(net2.Nets{1,2}.Layers{1,2}.b/2^f);

net1.Nets{1,3}.Layers{1,2}.factor = round(net2.Nets{1,3}.Layers{1,2}.factor/2^f);
net1.Nets{1,3}.Layers{1,2}.W = round(net2.Nets{1,3}.Layers{1,2}.W/2^f);
net1.Nets{1,3}.Layers{1,2}.b = round(net2.Nets{1,3}.Layers{1,2}.b/2^f);
net1.Nets{1,3}.Layers{1,3}.factor = round(net2.Nets{1,3}.Layers{1,2}.factor/2^f);
net1.Nets{1,3}.Layers{1,3}.W = round(net2.Nets{1,3}.Layers{1,3}.W/2^f);
net1.Nets{1,3}.Layers{1,3}.b = round(net2.Nets{1,3}.Layers{1,3}.b/2^f);

%% Save
save('ReducedNET','net1','f')

%% Train Data
% convert the training data to new fixed point representation 
% f = f_Original - f_New;

A = predictMulitinetClassesfi(net1, round(X1Train(:,:,:)/2^f), round(X2Train(:,:)/2^f));
[~,predictions] = max(A);
[~,labels] = max(YTrain(:,:));
TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0

figure;
plotconfusion(categorical(labels),categorical(predictions));

%% Test Data
% convert the training data to new fixed point representation 
% f = f_Original - f_New;

A = predictMulitinetClassesfi(net1, round(X1Test(:,:,:)/2^f), round(X2Test(:,:)/2^f));
[~,predictions] = max(A);
[~,labels] = max(YTest(:,:));
TestAccuracy = sum((predictions==labels))/length(labels)*100.0

figure;
plotconfusion(categorical(labels),categorical(predictions));




%% Some Details about the MN
%% Maximum Weight  
maxWf = max(max(net1.Nets{1,1}.Layers{1,2}.Wf))
maxbf = max(net1.Nets{1,1}.Layers{1,2}.bf)
maxWi = max(max(net1.Nets{1,1}.Layers{1,2}.Wi))
maxbi = max(net1.Nets{1,1}.Layers{1,2}.bi)
maxWc = max(max(net1.Nets{1,1}.Layers{1,2}.Wc))
maxbc = max(net1.Nets{1,1}.Layers{1,2}.bc)
maxWo = max(max(net1.Nets{1,1}.Layers{1,2}.Wo))
maxbo = max(net1.Nets{1,1}.Layers{1,2}.bo)
maxWy = max(max(net1.Nets{1,1}.Layers{1,2}.Wy))
maxby = max(net1.Nets{1,1}.Layers{1,2}.by)
maxW21 = max(max(net1.Nets{1,2}.Layers{1,2}.W))
maxb21 = max(net1.Nets{1,2}.Layers{1,2}.b)
maxW31 = max(max(net1.Nets{1,3}.Layers{1,2}.W))
maxb31 = max(net1.Nets{1,3}.Layers{1,2}.b)
maxW32 = max(max(net1.Nets{1,3}.Layers{1,3}.W))
maxb32= max(net1.Nets{1,3}.Layers{1,3}.b)

%% Minimum Weight
minWf = min(min(net1.Nets{1,1}.Layers{1,2}.Wf))
minbf = min(net1.Nets{1,1}.Layers{1,2}.bf)
minWi = min(min(net1.Nets{1,1}.Layers{1,2}.Wi))
minbi = min(net1.Nets{1,1}.Layers{1,2}.bi)
minWc = min(min(net1.Nets{1,1}.Layers{1,2}.Wc))
minbc = min(net1.Nets{1,1}.Layers{1,2}.bc)
minWo = min(min(net1.Nets{1,1}.Layers{1,2}.Wo))
minbo = min(net1.Nets{1,1}.Layers{1,2}.bo)
minWy = min(min(net1.Nets{1,1}.Layers{1,2}.Wy))
minby = min(net1.Nets{1,1}.Layers{1,2}.by)
minW21 = min(min(net1.Nets{1,2}.Layers{1,2}.W))
minb21 = min(net1.Nets{1,2}.Layers{1,2}.b)
minW31 = min(min(net1.Nets{1,3}.Layers{1,2}.W))
minb31 = min(net1.Nets{1,3}.Layers{1,2}.b)
minW32 = min(min(net1.Nets{1,3}.Layers{1,3}.W))
minb32= min(net1.Nets{1,3}.Layers{1,3}.b)


