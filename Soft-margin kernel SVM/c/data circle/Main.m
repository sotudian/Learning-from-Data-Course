clear
clc
% Data generation in two circle
N=70;%random points
r=1.1;%radius
x1=2;y1=3;%center at (x,y).
x2=4;y2=3;%center at (x,y).
Ns = round(1.28*N + 2.5*sqrt(N) + 100); % 4/pi = 1.2732
X = rand(Ns,1)*(2*r) - r;
Y = rand(Ns,1)*(2*r) - r;
I = find(sqrt(X.^2 + Y.^2)<=r);
X1 = X(I(1:N)) + x1;
Y1 = Y(I(1:N)) + y1;
X2 = X(I(1:N)) + x2;
Y2 = Y(I(1:N)) + y2;
Class1=cat(2,X1,Y1);
Y_Class1=ones(size(X1));
Class2=cat(2,X2,Y2);
Y_Class2=-ones(size(X1));

%% SVM
X=cat(1,Class1,Class2);
y=cat(1,Y_Class1,Y_Class2);
lambda=0.5;
[w1,b1,si] = train_svm_primal(X,y,lambda);
C=size(X,1);
T=2;
[w,b,si,w_plus]= SPARSE_train_svm_primal(X, y, lambda,C,T);
w1
b1
w
b
% w(1)/w1(1)
% w(2)/w1(2)
% b1/b
% SVMModel=fitcsvm(X,y,'BoxConstraint',1/size(X,1)); 
% SVMModel.Beta
% SVMModel.Bias
