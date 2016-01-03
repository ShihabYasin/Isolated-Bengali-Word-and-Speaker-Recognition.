clc
net = network;
net.numInputs = 1;
net.numLayers = 5;
net.biasConnect(1) = 0;
net.biasConnect(2) = 0;
net.biasConnect(3) = 0;
net.biasConnect(4) = 0;
net.biasConnect(5) = 0;
net.inputConnect(1,1) = 1;
net.outputConnect = [0 0 0 0 1];
net.layerConnect = [0 0 0 0 0 ; 1 0 0 0 1 ; 1 0 0 0 1 ; 0 1 0 0 0 ; 0 0 1 1 0];
net.inputs{1}.processFcns = {'mapminmax'};

net = init(net);
net.inputs{1}.size = 15;  %% INPUT VECTOR SIZE ,  SAMPLE INPUT KORI , SAMPLE ER SONKHA KOTO TA BUJAY

%  If the network is not sufficiently accurate, you can try
%  initializing the network and the training again by writing net = init(net); at command prompt.
%  Each time you initialize a feedforward network, the network parameters
%  are different and might produce different solutions.


net.layers{1}.size = 15;  % MEANS NO OF NEURONS IN LAYER 1
net.layers{1}.transferFcn = 'tansig';
net.layers{1}.initFcn = 'initnw';

net.layers{2}.size = 30;
net.layers{2}.transferFcn = 'tansig';
net.layers{2}.initFcn = 'initnw';

net.layers{3}.size = 35;
net.layers{3}.transferFcn = 'logsig';
net.layers{3}.initFcn = 'initnw';

net.layers{4}.size = 30;
net.layers{4}.transferFcn = 'tansig';
net.layers{4}.initFcn = 'initnw';

net.layers{5}.size = 3;     %% OUTPUT VECTOR SIZE ,  OUTPUT VECTOR E ELEMENT  SONKHA KOTO TA BUJAY
net.layers{5}.transferFcn = 'tansig';
net.layers{5}.initFcn = 'initnw';

net.biases{1}.initFcn = 'midpoint';
net.biases{1}.learn = 1;

net.biases{4}.initFcn = 'midpoint';
net.biases{4}.learn = 1;
net.biases{4}.learnFcn = '';

net.biases{2}.initFcn = 'midpoint';
net.biases{2}.learn = 1;
net.biases{2}.learnFcn = '';

net.biases{3}.initFcn = 'midpoint';
net.biases{3}.learn = 1;
net.biases{3}.learnFcn = '';

net.biases{5}.initFcn = 'midpoint';
net.biases{5}.learn = 1;
net.biases{5}.learnFcn = '';

net.inputWeights{1,1}.initFcn = 'initlay' ;
net.inputWeights{1,1}.learn = 1;
net.inputWeights{1,1}.learnFcn = 'learnp'
net.inputWeights{1,1}.weightFcn = 'dotprod';

net.layerWeights{2,1}.initFcn = 'initlay';
net.layerWeights{3,1}.initFcn = 'initlay';
net.layerWeights{4,2}.initFcn = 'initlay';
net.layerWeights{5,3}.initFcn = 'initlay';
net.layerWeights{5,4}.initFcn = 'initlay';

net.layerWeights{2,1}.learn = 1;
net.layerWeights{3,1}.learn = 1;
net.layerWeights{4,2}.learn = 1;
net.layerWeights{5,3}.learn = 1;
net.layerWeights{5,4}.learn = 1;

net.layerWeights{2,1}.weightFcn = 'dotprod';
net.layerWeights{3,1}.weightFcn = 'dotprod';
net.layerWeights{4,2}.weightFcn = 'dotprod';
net.layerWeights{5,3}.weightFcn = 'dotprod';
net.layerWeights{5,4}.weightFcn = 'dotprod';

net.layerWeights{5,3}.delays = 1;
net.layerWeights{2,5}.delays = 1;
net.layerWeights{3,5}.delays = 1;








net.initFcn = 'initlay';
net.performFcn = 'mse';
net.trainFcn = 'trainlm';
net.divideFcn  = 'dividerand' ;


  % LOADING DATA FROM FILES 
inn = ddd(:,1:15);  % TAKE TRAIN DATA FROM TRAIN_DATA FILE 
ouu = outt(:,1:3);  % TAKE target DATA FROM target_data FILE 

 TrainData = inn';
 TargetData = outt';

TestData = inn'; % LET 


% TrainData =[[1 1;3 2;2 1;1 2;3 2]];   sample 
% TargetData = [[1 1;1 2;2 1]]; 

[net,tr] = train(net,TrainData,TargetData);
plottrainstate(tr);


Y = sim(net,TestData);

 Y  % TEST RESULT HOLDS HERE , SEE COMMAND LINE , FOR TEST = TRAIN DATA WILL RESULTS = TO TARGET DATA , SO EXPECTED.

% PLOT ONLY 
net.plotFcns = {'plotperform','plottrainstate'};
% plotconfusion(TargetData,Y);







 plotregression(TargetData,Y);

plotperform(tr);
% plotroc(TargetData,Y);
% gensim(net)

%view(net)









