%function A = findPCA(ecg)
clc
clear point;
clear ecg;
clear meanECG;
clear ecg_Mean;
clear ecg_PCA;

[ECG, ANN, TYPE] = readPhysionetSignal();

%%
clc
clear C;
C = [];
for i = 1:20
    Ctemp = unique(TYPE{i});
    C = [C;Ctemp];
end
Cfinal = unique(C);
Cfinal(Cfinal == '+') = [];
Cfinal(Cfinal == '~') = [];
Cfinal(Cfinal == '|') = [];
Cfinal(Cfinal == 'x') = [];

save("selectedBeat.mat",'Cfinal');
%%
clear ecg;
sCount = 1;
for selSig = 1:20
  count = 1;
  for i = 5:length(ANN{1,selSig}) 
    if (TYPE{1,selSig}(i) == 'N')
        
    elseif (sum(TYPE{1,selSig}(i) == Cfinal) == 1)
        point(count) = ANN{1,selSig}(i);
        count = count + 1;
    end
  end
  for i = 1:min(500,length(point))
    ecg(sCount,:) = ECG{selSig}(point(i)-89:point(i)+162)';%rand(1,252);
    sCount = sCount+1;
  end
end


%%

SB = findPCA(ecg', 20);
%%
figure(3)
plot(1:252,SB(:,1:2))
%%
rangeP = 1:12;
A(rangeP,:) = findPCAcoeff(ecg(rangeP,:)',SB)

figure(3)
plot(1:252,ecg(rangeP,:),'b');
figure(4)
for i = rangeP
    plot(1:252,(A(i,:)*SB'),'r');
    hold on;
end


%%
save("PCAof_ABeats.mat",'SB');

