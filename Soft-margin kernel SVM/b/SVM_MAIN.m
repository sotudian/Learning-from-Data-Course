clear
clc
%% DATA preprocessing
load('arrhythmia.mat')
for i=1:279
  Median_column=median(X(:,i),'omitnan');
  for j=1:452
    if isnan(X(j,i))
     X(j,i)=Median_column;
    end
  end
end
Y = double(Y);  
Zeros_of_Y=find(Y==0);     % change y which are 0 to -1
Y(Zeros_of_Y)=-1;
L=randperm(length(X));
% Test Data
Test_X=X(L(1:89),:);
Test_Y=Y(L(1:89),:);
% Train Data
Train_X=X(L(90:452),:);
Train_Y=Y(L(90:452),:);



%% Train the SVM Classifier

%% Polynomial kernel-SVM
% Results of parameter tuning for Polynomial kernel-SVM using 3-fold cross-validation
[ Best_C_PO,Tuning_Results_pol] = Tuning_Polynomial(Train_X,Train_Y);
Num = Tuning_Results_pol(:,1);
BoxConstraint =Tuning_Results_pol(:,2);
PolynomialOrder = Tuning_Results_pol(:,3);
Misclassification_rate1=Tuning_Results_pol(:,4);
T1 = table(Num,BoxConstraint,PolynomialOrder,Misclassification_rate1)
fprintf('\n =======================================================================')
fprintf('\n                 Best parameter for Polynomial kernel-SVM ')
fprintf('\n =======================================================================')
Num = Best_C_PO(:,1);
BoxConstraint =Best_C_PO(:,2);
PolynomialOrder = Best_C_PO(:,3);
Misclassification_rate2=Best_C_PO(:,4);
T2 = table(Num,BoxConstraint,PolynomialOrder,Misclassification_rate2)
fprintf('\n #######################################################################\n\n')




%% RBF kernel-SVM
% Results of parameter tuning for RBF kernel-SVM using 3-fold cross-validation
[ Best_C_RS,Tuning_Results_rbf ] = Tuning_RBF(Train_X,Train_Y);
Num = Tuning_Results_rbf(:,1);
BoxConstraint =Tuning_Results_rbf(:,2);
Gamma = Tuning_Results_rbf(:,3);
Misclassification_rate3=Tuning_Results_rbf(:,4);
T3 = table(Num,BoxConstraint,Gamma,Misclassification_rate3)
fprintf('\n\n =======================================================================')
fprintf('\n                 Best parameter for RBF kernel-SVM ')
fprintf('\n =======================================================================')
Num = Best_C_RS(:,1);
BoxConstraint =Best_C_RS(:,2);
Gamma = Best_C_RS(:,3);
Misclassification_rate4=Best_C_RS(:,4);
T4 = table(Num,BoxConstraint,Gamma,Misclassification_rate4)
fprintf('\n #######################################################################\n\n')



%% Linear-SVM
% Results of parameter tuning for Linear-SVM using 3-fold cross-validation
[ Best_C,Tuning_Results_Linear ] = Tuning_Linear(Train_X,Train_Y);
Num = Tuning_Results_Linear(:,1);
BoxConstraint =Tuning_Results_Linear(:,2);
Misclassification_rate5=Tuning_Results_Linear(:,3);
T5 = table(Num,BoxConstraint,Misclassification_rate5)
fprintf('\n\n =======================================================================')
fprintf('\n                 Best parameter for Linear-SVM ')
fprintf('\n =======================================================================')
Num = Best_C(:,1);
BoxConstraint =Best_C(:,2);
Misclassification_rate6=Best_C(:,3);
T6 = table(Num,BoxConstraint,Misclassification_rate6)
fprintf('\n #######################################################################\n\n')






%% Traing with best parameters
Validated_SVM_POl=fitcsvm(Train_X,Train_Y,'KernelFunction','Polynomial','PolynomialOrder',Best_C_PO(:,3),'BoxConstraint',Best_C_PO(:,2),'Standardize',true);
Validated_SVM_RBF = fitcsvm(Train_X,Train_Y,'KernelFunction','rbf','KernelScale',1/sqrt(Best_C_RS(:,3)),'BoxConstraint',Best_C_RS(:,2),'Standardize',true);
Validated_SVM_Linear = fitcsvm(Train_X,Train_Y,'KernelFunction','linear','BoxConstraint',Best_C(:,2),'Standardize',true);

%% Model Test
fprintf('\n\n\n\n\n #######################################################################')
fprintf('\n                           Model test Results ')
fprintf('\n #######################################################################\n')
% Polynomial
[label_POL,score_POL]= predict(Validated_SVM_POl,Test_X);
Number_Misclassified_Polinomial=sum(Test_Y~= label_POL)
% RBF
[label_RBF,score_RBF] = predict(Validated_SVM_RBF,Test_X);
Number_Misclassified_RBF=sum(Test_Y~= label_RBF)
% Linear
[label_Linear,score_Linear] = predict(Validated_SVM_Linear,Test_X);
Number_Misclassified_Linear=sum(Test_Y~= label_Linear)
%  ROC
[Xsvm_POl,Ysvm_POl,Tsvm_POl,AUCsvm_POl] = perfcurve(double(Test_Y),double(score_POL(:,2)),Validated_SVM_POl.ClassNames(2));
[Xsvm_RBF,Ysvm_RBF,Tsvm_RBF,AUCsvm_RBF] = perfcurve(double(Test_Y),double(score_RBF(:,2)),Validated_SVM_RBF.ClassNames(2));
[Xsvm_Linear,Ysvm_Linear,Tsvm_Linear,AUCsvm_Linear] = perfcurve(double(Test_Y),double(score_Linear(:,2)),Validated_SVM_Linear.ClassNames(2));
Num = [1;2;3];
Kernel =["Polynomial";"RBF";"Linear"];
AUC=[AUCsvm_POl;AUCsvm_RBF;AUCsvm_Linear];
T7 = table(Num,Kernel,AUC)                  % The AUC values for each kernel
% Plot  ROC
plot(Xsvm_POl,Ysvm_POl,'--','LineWidth',2)
hold on
plot(Xsvm_RBF,Ysvm_RBF,':','LineWidth',2)
hold on
plot(Xsvm_Linear,Ysvm_Linear,'LineWidth',2)
legend('Polynomial','RBF','Linear')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Polynomial, RBF and Linear kernel SVM')
hold off

