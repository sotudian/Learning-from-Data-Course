
% Shahab Sotudian
clear
clc
% DATA and parameters
load('a9a.mat')
T = 100;
SPLIT =[2,3,10,25,100];
Counter = 0;
figure;
for S = SPLIT
[alpha, DTCell] = train_boosted_dt(Xtr,ytr,T,S);
    for j = 1:T
        [ypred_test, scores_Dt_test] = test_boosted_dt(Xte, alpha(1:j), DTCell(1:j));
        [ypred_train, scores_Dt_train] = test_boosted_dt(Xtr, alpha(1:j), DTCell(1:j));
        [X_ts,Y_ts,T_ts,AUC_ts]= perfcurve(yte,scores_Dt_test,1,'XVals',[0:0.01:1]);
        [X_tr,Y_tr,T_tr,AUC_tr]= perfcurve(ytr,scores_Dt_train,1,'XVals',[0:0.01:1]);
        Err_test(j,:) = 1-AUC_ts;    
        Err_train(j,:) = 1-AUC_tr;
    end
L=1:T;
plot(L,Err_train(:,1),'LineWidth',1.1)
hold on
plot(L,Err_test(:,1),'LineWidth',1.1,'LineStyle','--')
hold off
title(strcat('Number of Splits = ', string(S)));
xlabel('T');
ylabel('Error');
legend('Train Result','Test Result');
end






