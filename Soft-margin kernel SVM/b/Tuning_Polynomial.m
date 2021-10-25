function [ Best_C_PO,Tuning_Results ] = Tuning_Polynomial(X,Y)
box=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000];  % BoxConstraint
PolynomialOrder=[1,2,3];                                  % Polynomial Order
Best_C_PO=[0,0,0,100];
f=1;
A1=X(1:121,:);    B1=Y(1:121,:);                          % Data splitting for 3-fold cross validation
A2=X(122:242,:);  B2=Y(122:242,:);
A3=X(243:363,:);  B3=Y(243:363,:);
for j=1:size(box,2)
    for p=1:size(PolynomialOrder,2)
%            3-fold cross validation.
svm1 = fitcsvm(cat(1,A1,A2),cat(1,B1,B2),'KernelFunction','Polynomial','PolynomialOrder',PolynomialOrder(p),'BoxConstraint',box(j),'Standardize',true);
Predicted1 = predict(svm1,A3);
E(1)=sum(B3~= Predicted1)/size(B3,1);
svm2 = fitcsvm(cat(1,A1,A3),cat(1,B1,B3),'KernelFunction','Polynomial','PolynomialOrder',PolynomialOrder(p),'BoxConstraint',box(j),'Standardize',true);
Predicted2 = predict(svm2,A2);
E(2)=sum(B2~= Predicted2)/size(B3,1);
svm3 = fitcsvm(cat(1,A2,A3),cat(1,B2,B3),'KernelFunction','Polynomial','PolynomialOrder',PolynomialOrder(p),'BoxConstraint',box(j),'Standardize',true);
Predicted3 = predict(svm3,A1);
E(3)=sum(B1~= Predicted3)/size(B3,1);
   Tuning_Results(f,1)=f;
   Tuning_Results(f,2)=box(j);
   Tuning_Results(f,3)=PolynomialOrder(p);
   Tuning_Results(f,4)=mean(E);                          % Estimate the misclassification rate.

           if Best_C_PO(1,4)>Tuning_Results(f,4)         % Finding the best parameters
               Best_C_PO=Tuning_Results(f,:);
           end
      f=f+1;
   end
end
end

