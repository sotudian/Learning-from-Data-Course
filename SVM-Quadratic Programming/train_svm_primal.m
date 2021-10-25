function [w,b] = train_svm_primal(X, y, lambda)
m = size(X,1);
d = size(X,2);
% Solving soft SVM in the primal form using QP solver
H   = diag([zeros(1,m+1),(2*lambda)*ones(1,d)]);
f   = [(1/m)*ones(1,m),zeros(1,d+1)]';
A   = [-eye(m) -y -diag(y)*X];
b   = -ones(m,1);
Aeq = [];
beq = [];
lb  = [zeros(m,1);-inf*ones(d+1,1)];
ub  = [];
W   = quadprog(H,f,A,b,Aeq,beq,lb,ub);


si  = W(1:m,:);
w   = W(m+2:d+m+1,:);
b   = W(m+1,:);

end




