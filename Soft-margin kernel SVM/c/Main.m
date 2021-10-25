clear
clc
%% DATA 
load('arrhythmia.mat')
for i=1:279
  Median_column=median(X(:,i),'omitnan');
  for j=1:452
    if isnan(X(j,i))
     X(j,i)=Median_column;
    end
  end
end
% change y 0 to -1
Y = double(Y);
Zeros_of_Y=find(Y==0);
Y(Zeros_of_Y)=-1;

L=randperm(length(X));
% Test Data
Test_X=X(L(1:89),:);
Test_Y=Y(L(1:89),:);
% Train Data
Train_X=X(L(90:452),:);
Train_Y=Y(L(90:452),:);

%% Tunning C and T for Sparse SVM
[Best_T_C,Tuning_Results_SPARSE] = Tuning_SPARSE_svm(Train_X,Train_Y)
Num = Tuning_Results_SPARSE(:,1);
BoxConstraint =Tuning_Results_SPARSE(:,2);
T = Tuning_Results_SPARSE(:,3);
Misclassification_rate=Tuning_Results_SPARSE(:,4);
T1 = table(Num,BoxConstraint,T,Misclassification_rate)
fprintf('\n =======================================================================')
fprintf('\n                 Best parameter for sparse SVM ')
fprintf('\n =======================================================================')
Num = Best_T_C(:,1);
BoxConstraint =Best_T_C(:,2);
T = Best_T_C(:,3);
Misclassification_rate=Best_T_C(:,4);
T2 = table(Num,BoxConstraint,T,Misclassification_rate)
fprintf('\n #######################################################################\n\n')


% % % Traing with best parameters
% % [w_SPARSE,b_SPARSE,si_SPARSE,w_plus]= SPARSE_train_svm_primal(Train_X, Train_Y,0.5,Best_T_C(:,2),Best_T_C(:,3));

%% Eliminating features
E_features=find(abs(w_SPARSE)<(max(abs(w_SPARSE)))/1e5);
Reduced_Train_X=Train_X;
Reduced_Test_X=Test_X;
Reduced_Train_X(:,E_features)=[];
Reduced_Test_X(:,E_features)=[];
% Training a linear SVM without the sparsity constraint with the reduced features
Reduced_features_SVM_Linear = fitcsvm(Reduced_Train_X,Train_Y,'KernelFunction','linear','BoxConstraint',Best_T_C(:,2),'Standardize',true);

%% Model Test
[label_Linear,score_Linear] = predict(Reduced_features_SVM_Linear,Reduced_Test_X);
%  ROC
[Xsvm_Linear,Ysvm_Linear,Tsvm_Linear,AUCsvm_Linear] = perfcurve(double(Test_Y),double(score_Linear(:,2)),Reduced_features_SVM_Linear.ClassNames(2));
AUCsvm_Linear


% Not reduced
All_features_SVM_Linear = fitcsvm(Train_X,Train_Y,'KernelFunction','linear','BoxConstraint',Best_T_C(:,2),'Standardize',true);
[All_label_Linear,All_score_Linear] = predict(All_features_SVM_Linear,Train_X);
[All_Xsvm_Linear,All_Ysvm_Linear,All_Tsvm_Linear,All_AUCsvm_Linear] = perfcurve(double(Test_Y),double(All_score_Linear(:,2)),All_features_SVM_Linear.ClassNames(2));
All_AUCsvm_Linear

% Plot  ROC
plot(All_Xsvm_Linear,All_Ysvm_Linear,'--','LineWidth',2)
hold on
plot(Xsvm_Linear,Ysvm_Linear,'LineWidth',2)
legend('All features','Reduced features')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Linear SVM')
hold off



