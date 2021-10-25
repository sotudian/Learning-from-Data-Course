% Generate Gaussian points

clear;
mu=[1; 2; 3];      %mean
L=[10; 5; 0.01];   % eigenvalues of covariance
Cov=rotx(20)*roty(30)*rotz(45)*eye(3)*diag(L)*transpose(rotx(20)*roty(30)*rotz(45)*eye(3)); %covariance
 
[V,D]=eig(Cov);   %eigen-decomposition

% generate an nx3 matrix of points and plot them 
n=300;
X=mvnrnd(transpose(mu),Cov,n); 
scatter3(X(:,1),X(:,2),X(:,3));
xlabel('x');
ylabel('y');
zlabel('z');

sample_mu=transpose(X)*ones(n,1)/n;
%X_center=X-ones(n,1)*transpose(sample_mu);
%sample_Cov=transpose(X_center)*X_center/n;

[V_s,Coeff,d_s]=pca(X);

hold on; 

%plot eigenvectors
quiver3(sample_mu(1),sample_mu(2),sample_mu(3),V_s(1,1),V_s(2,1),V_s(3,1),15,'r');
quiver3(sample_mu(1),sample_mu(2),sample_mu(3),V_s(1,2),V_s(2,2),V_s(3,2),8,'g');
quiver3(sample_mu(1),sample_mu(2),sample_mu(3),V_s(1,3),V_s(2,3),V_s(3,3),8,'b');

% compute reduced 2D representation and plot these points
X_red=Coeff(:,1:2)*transpose(V_s(:,1:2))+ones(n,1)*transpose(sample_mu);
scatter3(X_red(:,1),X_red(:,2),X_red(:,3));

%compute approximation error per element
error=norm(X-X_red,'fro')/n;
