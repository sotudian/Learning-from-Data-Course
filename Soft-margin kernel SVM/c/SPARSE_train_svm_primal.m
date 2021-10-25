function [w,b,si,w_plus] = SPARSE_train_svm_primal(X, y, lambda,C,T)
m = size(X,1);
d = size(X,2);

H   = diag([zeros(1,m+1),(2*lambda)*ones(1,d),zeros(1,d)]);
f   = [(C)*ones(1,m),zeros(1,2*d+1)]';

% A


A   = [-eye(m)      -y            -diag(y)*X        zeros(m,d);
       zeros(d,m)   zeros(d,1)    eye(d)            -eye(d)   ;
       zeros(d,m)   zeros(d,1)    -eye(d)           -eye(d)   ;
       zeros(1,m)    0            zeros(1,d)        ones(1,d)];

b   = [-ones(m,1);
       zeros(d,1);
       zeros(d,1);
               T];

Aeq=[];
beq = [];

lb  = [zeros(m,1);-inf*ones(d+1,1);zeros(d,1)];
ub  = [];
W   = quadprog(H,f,A,b,Aeq,beq,lb,ub);

si  = W(1:m,:);
w   = W(m+2:d+m+1,:);
b   = W(m+1,:);
w_plus=W(m+d+2:2*d+m+1,:);


end
