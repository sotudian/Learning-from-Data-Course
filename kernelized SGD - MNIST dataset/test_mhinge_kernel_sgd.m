function [ypred] = test_mhinge_kernel_sgd(alpha, Xsv, Xte, p)

for tt=1:size(Xte,2)
% for tt=1:1500
    
    
    FR=zeros(1,10);
    for i=1:10
        for j=1:size(Xsv,2)
            
         FR(i)=FR(i)+ alpha(j,i)*((Xsv(:,j)'*Xte(:,tt))^p);
        
        end

    end
    
    [a,label]=max(FR);
    ypred(tt)=label-1;
    
end 
% ypred

end

