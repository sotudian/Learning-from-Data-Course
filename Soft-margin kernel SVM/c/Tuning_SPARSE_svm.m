function [Best_T_C,Tuning_Results] = Tuning_SPARSE_svm(X,Y)
C=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000];        % BoxConstraint
T =[5,33,61,89,117,145,173,201,229,257];                      % Sparsity 
Best_T_C=[0,0,0,100];
f=1;              lambda=0.5;
A1=X(1:121,:);    B1=Y(1:121,:);                              % Data splitting for 3-fold cross validation
A2=X(122:242,:);  B2=Y(122:242,:);
A3=X(243:363,:);  B3=Y(243:363,:);

for j=1:size(C,2)
    for p=1:size(T,2)
        %            3-fold cross validation   
[w_SPARSE1,b_SPARSE1,si_SPARSE1,w_plus1]= SPARSE_train_svm_primal(cat(1,A1,A2),cat(1,B1,B2), lambda,C(j),T(p));
[w_SPARSE2,b_SPARSE2,si_SPARSE2,w_plus2]= SPARSE_train_svm_primal(cat(1,A1,A3),cat(1,B1,B3), lambda,C(j),T(p));
[w_SPARSE3,b_SPARSE3,si_SPARSE3,w_plus3]= SPARSE_train_svm_primal(cat(1,A2,A3),cat(1,B2,B3), lambda,C(j),T(p));
f_SPARSE1=0;      f_SPARSE2=0;     f_SPARSE3=0;
for i=1:size(A3,1)
            if   sign(w_SPARSE1'*A3(i,:)'+b_SPARSE1)~=B3(i)
               f_SPARSE1=f_SPARSE1+1;
            end
            if   sign(w_SPARSE2'*A2(i,:)'+b_SPARSE2)~=B2(i)
               f_SPARSE2=f_SPARSE2+1;
            end
            if   sign(w_SPARSE3'*A1(i,:)'+b_SPARSE3)~=B1(i)
               f_SPARSE3=f_SPARSE3+1;
            end
end
   Tuning_Results(f,1)=f;
   Tuning_Results(f,2)=C(j);
   Tuning_Results(f,3)=T(p);
   Tuning_Results(f,4)=(f_SPARSE1+f_SPARSE2+f_SPARSE3)/size(Y,1);                 % Estimate the misclassification rate.
   
       if Best_T_C(1,4)>Tuning_Results(f,4)                                       % Finding the best parameters
           Best_T_C=Tuning_Results(f,:);
       end
      f=f+1;
    end
end
end

