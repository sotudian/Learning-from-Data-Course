function [w,b] = train_svm_dual(X,y,lambda)
m=size(y,1);
% Solving soft SVM in the dual form using QP solver
H1=zeros(m,m);
        for i=1:m
            for j=1:m
                H1(i,j)=y(i)*y(j)*(X(i,:)*X(j,:)');
            end
        end
H=(1/(2*lambda))*H1;
f = -ones (1,m);
A=[];
b=[];
Aeq = y';
beq = 0;
lb = zeros(m,1);
ub = (1/m)*ones(m,1);
Alfa  = quadprog(H,f,A,b,Aeq,beq,lb,ub);
% Setting almost zero Alfa equal to zero
AlmostZero = (abs(Alfa)< max(abs(Alfa))/1e6);
Alfa(AlmostZero) = 0;

Is=find(Alfa >0);                      % Recovering w
w1=zeros(size(Is,1),size(X,2));
for i=1:size(Is,1)
    w1(i,:)=((Alfa(Is(i))*y(Is(i))).*X(Is(i),:));
end
w=(1/(2*lambda))*sum(w1);

Im= find(Alfa >0 & Alfa < 1/m);        % Recovering b
b1=y(Im)-X(Im,:)*w';
b=mean(b1);






end

