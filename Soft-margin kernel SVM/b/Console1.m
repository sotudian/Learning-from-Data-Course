% Shahabeddin Sotudian-94125091
% Question 2
%  SVM Classifier
clear;
clc;
%------------------------------------------------------------------------------------------------------------------------------------------------------------
%% DATA

load('Dataset_WBCD.mat');
% Training Data
Training_Data=cat(1,Dataset_WBCD(1:222,:),Dataset_WBCD(445:563,:));
% Test Data
Test_Data=cat(1,Dataset_WBCD(223:333,:),Dataset_WBCD(564:623,:));
% Validation Data
Validation_Data=cat(1,Dataset_WBCD(334:444,:),Dataset_WBCD(624:683,:));

%------------------------------------------------------------------------------------------------------------------------------------------------------------
%% Train the SVM Classifier
fprintf('\n #######################################################################')
fprintf('\n                           Training Results ')
fprintf('\n #######################################################################\n')

% 1-Polynomial

fprintf('\n\n\n\n =======================================================================')
fprintf('\n                      1- Polynomial kernel-SVM ')
fprintf('\n =======================================================================')
SVMModel_train_Polynomial=fitcsvm(Training_Data(:,1:9),Training_Data(:,10),'KernelFunction','Polynomial',...
               'PolynomialOrder',5,'BoxConstraint',10,'KernelScale',10); 
cv1 = crossval(SVMModel_train_Polynomial);
%        Estimate the out-of-sample misclassification rate.
       ap1=kfoldLoss(cv1);
       fprintf(['\n Out-of-sample misclassification rate for training data Polynomial kernel-SVM = %.4f%\n'], ap1)
       
    
       

% 2-RBF

fprintf('\n\n\n\n =======================================================================')
fprintf('\n                         2- RBF kernel-SVM ')
fprintf('\n =======================================================================')
            
 SVMModel_train_RBF=fitcsvm(Training_Data(:,1:9),Training_Data(:,10),'KernelFunction','rbf',...
               'BoxConstraint',10,'KernelScale',10); 
cv2 = crossval(SVMModel_train_RBF);
%        Estimate the out-of-sample misclassification rate.
       ap2=kfoldLoss(cv2);
       fprintf(['\n Out-of-sample misclassification rate for training data RBF kernel-SVM = %.4f%\n'], ap2)
       
%------------------------------------------------------------------------------------------------------------------------------------------------------------       
%% Validation
fprintf('\n\n\n\n\n #######################################################################')
fprintf('\n                           Validation Results ')
fprintf('\n #######################################################################\n')


% 1-Polynomial

[ Best_BC_PO_KS,Tuning_Results_pol ] = Tuning_Polynomial(Validation_Data(:,1:9),Validation_Data(:,10));
fprintf('\n\n\n\n =======================================================================')
fprintf('\n        Results of parameter tuning for Polynomial kernel-SVM ')
fprintf('\n =======================================================================')
Num = Tuning_Results_pol(:,1);
BoxConstraint =Tuning_Results_pol(:,2);
PolynomialOrder = Tuning_Results_pol(:,3);
KernelScale =Tuning_Results_pol(:,4);
Misclassification_rate=Tuning_Results_pol(:,5);
T1 = table(Num,BoxConstraint,PolynomialOrder,KernelScale,Misclassification_rate)
fprintf('\n =======================================================================')
fprintf('\n                 Best parameter for Polynomial kernel-SVM ')
fprintf('\n =======================================================================')
Num = Best_BC_PO_KS(:,1);
BoxConstraint =Best_BC_PO_KS(:,2);
PolynomialOrder = Best_BC_PO_KS(:,3);
KernelScale =Best_BC_PO_KS(:,4);
Misclassification_rate=Best_BC_PO_KS(:,5);
T2 = table(Num,BoxConstraint,PolynomialOrder,KernelScale,Misclassification_rate)
fprintf('\n #######################################################################\n\n')



% 2-RBF


[ Best_BC_RS_KS,Tuning_Results_rbf ] = Tuning_RBF(Validation_Data(:,1:9),Validation_Data(:,10));
fprintf('\n\n\n\n =======================================================================')
fprintf('\n            Results of parameter tuning for RBF kernel-SVM ')
fprintf('\n =======================================================================')
Num = Tuning_Results_rbf(:,1);
BoxConstraint =Tuning_Results_rbf(:,2);
Rbf_sigma = Tuning_Results_rbf(:,3);
KernelScale =Tuning_Results_rbf(:,4);
Misclassification_rate=Tuning_Results_rbf(:,5);
T3 = table(Num,BoxConstraint,Rbf_sigma,KernelScale,Misclassification_rate)
fprintf('\n\n =======================================================================')
fprintf('\n                 Best parameter for RBF kernel-SVM ')
fprintf('\n =======================================================================')
Num = Best_BC_RS_KS(:,1);
BoxConstraint =Best_BC_RS_KS(:,2);
Rbf_sigma = Best_BC_RS_KS(:,3);
KernelScale =Best_BC_RS_KS(:,4);
Misclassification_rate=Best_BC_RS_KS(:,5);
T4 = table(Num,BoxConstraint,Rbf_sigma,KernelScale,Misclassification_rate)
fprintf('\n #######################################################################\n\n')



%------------------------------------------------------------------------------------------------------------------------------------------------------------
%% Model test
fprintf('\n\n\n\n\n #######################################################################')
fprintf('\n                           Model test Results ')
fprintf('\n #######################################################################\n')


% 1-Polynomial

fprintf('\n\n\n\n =======================================================================')
fprintf('\n                 Model test for Polynomial kernel-SVM ')
fprintf('\n =======================================================================')

Validated_SVM_POl=fitcsvm(Test_Data(:,1:9),Test_Data(:,10),'KernelFunction','Polynomial',...
               'PolynomialOrder',Best_BC_PO_KS(:,3),'BoxConstraint',Best_BC_PO_KS(:,2),'KernelScale',Best_BC_PO_KS(:,4)); 
label = predict(Validated_SVM_POl,Test_Data(:,1:9));
Numerr = sum(Test_Data(:,10)~= label);
fprintf(['\n Number of Misclassified test data(Polynomial) = %.3f%\n'], Numerr)
% the confusion matrix
fprintf('\n\n\n ===============    Confusion Matrix    ================ \n')
conMat = confusionmat(Test_Data(:,10),label);
disp(conMat)
fprintf('\n #######################################################################\n\n')



% 1-RBF


fprintf('\n\n\n\n =======================================================================')
fprintf('\n                 Model test for RBF kernel-SVM ')
fprintf('\n =======================================================================')

Validated_SVM_RBF=fitcsvm(Test_Data(:,1:9),Test_Data(:,10),'KernelFunction','rbf',...
               'BoxConstraint',Best_BC_RS_KS(:,2),'KernelScale',Best_BC_RS_KS(:,4)); 
label2 = predict(Validated_SVM_RBF,Test_Data(:,1:9));
Numerr2 = sum(Test_Data(:,10)~= label2);
fprintf(['\n Number of Misclassified test data(RBF) = %.3f%\n'], Numerr2)
% the confusion matrix
fprintf('\n\n\n ===============    Confusion Matrix    ================ \n')
conMat2 = confusionmat(Test_Data(:,10),label2);
disp(conMat2)
fprintf('\n #######################################################################\n\n')





%------------------------------------------------------------------------------------------------------------------------------------------------------------
%% Question 2.2 error rate for test and training data
fprintf('\n\n\n\n =======================================================================')
fprintf('\n      Error rate for test and training data - Polynomial kernel-SVM ')
fprintf('\n =======================================================================')

Validated_SVM_POl_2_1=fitcsvm(Training_Data(:,1:9),Training_Data(:,10),'KernelFunction','Polynomial',...
               'PolynomialOrder',Best_BC_PO_KS(:,3),'BoxConstraint',Best_BC_PO_KS(:,2),'KernelScale',Best_BC_PO_KS(:,4)); 
label_POl_2_1 = predict(Validated_SVM_POl_2_1,Training_Data(:,1:9));
Err_rate1=sum(Training_Data(:,10)~= label_POl_2_1)/size(Training_Data(:,10),1);

fprintf(['\n Error rate of test data (Polynomial) = %.3f%\n'], (Numerr/size(Test_Data(:,10),1)))
fprintf(['\n Error rate of training data (Polynomial) = %.3f%\n'], Err_rate1)

fprintf('\n\n\n\n =======================================================================')
fprintf('\n         Error rate for test and training data - RBF kernel-SVM ')
fprintf('\n =======================================================================')
Validated_SVM_RBF_2_1=fitcsvm(Training_Data(:,1:9),Training_Data(:,10),'KernelFunction','rbf',...
               'BoxConstraint',Best_BC_RS_KS(:,2),'KernelScale',Best_BC_RS_KS(:,4)); 
label2_1 = predict(Validated_SVM_RBF_2_1,Training_Data(:,1:9));
Err_rate2 = sum(Training_Data(:,10)~= label2_1)/size(Training_Data(:,10),1);
fprintf(['\n Error rate of test data(RBF) = %.3f%\n'], (Numerr2/size(Test_Data(:,10),1)))
fprintf(['\n Error rate of training data(RBF) = %.3f%\n'], Err_rate2)

fprintf('\n #######################################################################\n\n')





%% Question 2.3 Number of support vectors
fprintf('\n\n\n\n =======================================================================')
fprintf('\n                     Number of support vectors ')
fprintf('\n =======================================================================')
fprintf(['\n \nNumber of support vectors for Polynomial = %d%\n\n'], size(Validated_SVM_POl.SupportVectors,1))

fprintf(['\n Number of support vectors for RBF = %d%\n\n'], size(Validated_SVM_RBF.SupportVectors,1))




