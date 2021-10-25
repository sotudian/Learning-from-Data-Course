lambda=2;
y=[-1;1;-1;-1];
X=[7 7;8 8;9 9;10 10];

n = size(X,1);
d = size(X,2);

H   = diag([zeros(1,n+1),lambda*ones(1,d)]);
f   = [(1/n)*ones(1,n),zeros(1,d+1)]';
A   = [-eye(n) -y -diag(y)*X]
b   = -ones(n,1);
Aeq = [];
beq = [];
lb  = [zeros(n,1);-inf*ones(d+1,1)];
ub  = [];
W   = quadprog(H,f,A,b,Aeq,beq,lb,ub);

eps = W(1:n,:)
w   = W(n+2:d+n+1,:)
b   = W(n+1,:)
