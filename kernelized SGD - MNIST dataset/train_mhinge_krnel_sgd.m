function [alpha, Xsv] = train_mhinge_krnel_sgd(Xtr, ytr, Delta, p)
tic
k=10;
Eta=1/sqrt(size(Xtr,2));
% alpha=zeros(1000,k);
alpha=zeros(size(Xtr,2),k);
% alpha = sparse(size(Xtr,2),k);
rr=1;
zero_rows=[];
for t=1:size(Xtr,2)
% for t=1:1000
%     t
% Xtr(:,t)
% ytr(:,t)
    

    %     Subgradient
    for i=1:10    % Y_prime
        Y=ytr(:,t);
        P1=0;
        P2=0;

        if t~=1
        
            v=1:(t-1);
%             zero_rows
            if isempty(zero_rows)
               
            else
            v(zero_rows)=[];
            end
            
            for j=v
                M1=( (Xtr(:,j)'*Xtr(:,t))^p );
            P1=P1+   alpha(j,i)*M1;
            P2=P2+   alpha(j,Y+1)*M1;
            end
        end

        LOSS(i)=Delta(i,Y+1)+P1-P2;
    end
    [M,Y_prime_star] = max(LOSS);



    if Y_prime_star == (Y+1) 

        alpha(t,:)=zeros(1,k);
        zero_rows(rr)=t;
        rr=1+rr;
    else
        A1=zeros(1,k);
        A1(1,Y_prime_star)=-Eta;
        A1(1,Y+1)= Eta;
        alpha(t,:)=A1;
    end



end


% Rem=1:size(Xtr,2);
Rem=1:size(Xtr,2);
% Rem=1:1000;
Rem(zero_rows)=[];
Xsv=Xtr(:,Rem);

alpha(zero_rows,:)=[];

toc

end

