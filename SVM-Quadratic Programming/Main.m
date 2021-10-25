% Shahab Sotudian
clear
clc
% Data generation in two circle
N=70;                                       % Number of random points
r=1.1;                                      %radius
x1=2;y1=3;                                  %center of first circle
x2=4;y2=3;                                  %center of second circle
Ns = round(1.28*N + 2.5*sqrt(N) + 100);     % 4/pi = 1.2732
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
% DATA Plotting
scatter(Class1(:,1),Class1(:,2),'MarkerFaceColor',[0 1 0],'LineWidth',1.5)
hold on
scatter(Class2(:,1),Class2(:,2),'MarkerFaceColor',[0 0 1],'LineWidth',1.5)

%% SVM
X=cat(1,Class1,Class2);
y=cat(1,Y_Class1,Y_Class2);
lambda=0.2;
[w_primal,b_primal] = train_svm_primal(X,y,lambda)
[w_dual,b_dual] = train_svm_dual(X,y,lambda)



% TEST
N=20;%random points
r=0.9
Ns = round(1.28*N + 2.5*sqrt(N) + 100); % 4/pi = 1.2732
X_test = rand(Ns,1)*(2*r) - r;
Y_test = rand(Ns,1)*(2*r) - r;
I_test = find(sqrt(X_test.^2 + Y_test.^2)<=r);
X1_test = X_test(I_test(1:N)) + x1;
Y1_test = Y_test(I_test(1:N)) + y1;
X2_test = X_test(I_test(1:N)) + x2;
Y2_test = Y_test(I_test(1:N)) + y2;
Class1_test=cat(2,X1_test,Y1_test);
Y_Class1_test=ones(size(X1_test));
Class2_test=cat(2,X2_test,Y2_test);
Y_Class2_test=-ones(size(X1_test));
%% Perceptron
X_test=cat(1,Class1_test,Class2_test);
y_test=cat(1,Y_Class1_test,Y_Class2_test);
f=0;
for i=1:size(X_test,1)
            if   sign(w_primal'*X_test(i,:)'+b_primal)~=y_test(i)
               f=f+1;
            end
end
f





