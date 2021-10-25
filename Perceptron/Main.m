% Shahab Sotudian
clear
clc
% Data generation in two circle
N=70;%random points
r=0.95;%radius
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

%% Perceptron
X=cat(1,Class1,Class2);
y=cat(1,Y_Class1,Y_Class2);

X=cat(1,Class1,Class2);
y=cat(1,Y_Class1,Y_Class2);

% % % a=cat(2,X,y);
% % % random_x = a(randperm(size(a, 1)), :);
% % % X=random_x(:,1:2);
% % % y=random_x(:,3);
[w,b] = train_perceptron(X, y)
% % A(1)=0;
% % A(2)=5;
% % B(1)=w*[5;5]+b;
% % B(2)=w*[0;0]+b;
% DATA Plotting
% scatter(Class1(:,1),Class1(:,2),'MarkerFaceColor',[0 1 0],'LineWidth',1.5)
% hold on
% scatter(Class2(:,1),Class2(:,2),'MarkerFaceColor',[0 0 1],'LineWidth',1.5)


for i=1:size(X,1)
            if   sign(w*X(i,:)'+b)~=y(i)
               i
            end
end

for i=1:size(X,1)
            RESULT(i,1)=sign(w*X(i,:)'+b);
            RESULT(i,2)=y(i);
             
end


% TEST
N=20;%random points
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



for i=1:size(X_test,1)
            if   sign(w*X_test(i,:)'+b)~=y_test(i)
               i
            end
end




% DATA Plotting
scatter(Class1(:,1),Class1(:,2),'MarkerFaceColor',[0 1 0],'LineWidth',1.5)
hold on
scatter(Class2(:,1),Class2(:,2),'MarkerFaceColor',[0 0 1],'LineWidth',1.5)

% DATA Plotting
figure
scatter(Class1(:,1),Class1(:,2),'MarkerFaceColor',[0 1 0],'LineWidth',1)
hold on
scatter(Class2(:,1),Class2(:,2),'MarkerFaceColor',[0 0 1],'LineWidth',1)
hold on
scatter(X_test(:,1),X_test(:,2),'MarkerFaceColor',[0 0 0],'LineWidth',2)






