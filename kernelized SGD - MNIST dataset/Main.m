% % % clear
% % % clc
% % % % DATA 
% % % load('mnist.mat')
% % % 
% % % %% Cost matrix like Question 5.2.a
% % % p=5;  % Degree of polynomial kernel
% % % Delta_a=(ones(10)-eye(10));  % Cost matrix
% % % [alpha_a, Xsv_a] = train_mhinge_krnel_sgd(Xtr, ytr, Delta_a, p);
% % % [ypred_a] = test_mhinge_kernel_sgd(alpha_a, Xsv_a, Xte, p);
% % % % test error (0/1 Loss)
% % % Loss_a=numel(find(ypred_a~=yte(1:size(ypred_a,2))))
% % % Err_percent_a=Loss_a/size(ypred_a,2)
% % % C_a = confusionmat(yte(1:size(ypred_a,2)),ypred_a)
% % % 
% % % %% Cost matrix like Question 5.2.b
% % % p=5;  % Degree of polynomial kernel
% % % % Cost matrix
% % % Delta_b=zeros(10);
% % % for d1=1:10
% % %     for d2=1:10
% % %          if d1==d2
% % %             Delta_b(d1,d2)=0;    
% % %          elseif abs(d1-d2)==1
% % %             Delta_b(d1,d2)=1;   
% % %          else
% % %             Delta_b(d1,d2)=2; 
% % %          end
% % %     end
% % % end
% % % 
% % % 
% % % [alpha_b, Xsv_b] = train_mhinge_krnel_sgd(Xtr, ytr, Delta_b, p);
% % % [ypred_b] = test_mhinge_kernel_sgd(alpha_b, Xsv_b, Xte, p);
% % % % test error (0/1 Loss)
% % % Loss_b=numel(find(ypred_b~=yte(1:size(ypred_b,2))))
% % % Err_percent_b=Loss_b/size(ypred_b,2)
% % % C_b = confusionmat(yte(1:size(ypred_b,2)),ypred_b)
% % % 
% % % 
% % % 


%% Hazf
clear
clc
% DATA 
load('mnist.mat')
p=5;  % Degree of polynomial kernel
% Cost matrix
Delta_b=zeros(10);
for d1=1:10
    for d2=1:10
         if d1==d2
            Delta_b(d1,d2)=0;    
         elseif abs(d1-d2)==1
            Delta_b(d1,d2)=1000000;   
         else
            Delta_b(d1,d2)=2000000; 
         end
    end
end


[alpha_b, Xsv_b] = train_mhinge_krnel_sgd(Xtr, ytr, Delta_b, p);
[ypred_b] = test_mhinge_kernel_sgd(alpha_b, Xsv_b, Xte, p);
% test error (0/1 Loss)
Loss_b=numel(find(ypred_b~=yte(1:size(ypred_b,2))))
Err_percent_b=Loss_b/size(ypred_b,2)
C_b = confusionmat(yte(1:size(ypred_b,2)),ypred_b)



