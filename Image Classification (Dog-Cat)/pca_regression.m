%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the PCA basis. 
%It also takes in a test data matrix Xtest and a dimension k
%and uses the top-k vectors in the PCA basis to reduce the 
%dimension of Xtrain and Xtest. Finally, it uses the reduced data
%as inputs to the linear_regression function to produce 
%a vector of label guesses yguess.

function yguess = pca_regression(Xtrain,ytrain,Xtest,k)

[coeff,score,latent,tsquared,explained,mu]= pca(Xtrain,'Centered',true,'NumComponents',k);

Xtrain_reduced=(Xtrain-ones(size(Xtrain,1),1)*(mu))*coeff;                % Reduced-dimension training data
Mu_tset=(transpose(Xtest)*ones(size(Xtest,1),1))/size(Xtest,1);           % Test mean 
Xtest_reduced=(Xtest-ones(size(Xtest,1),1)*(Mu_tset'))*coeff;             % Reduced-dimension test data

yguess = linear_regression(Xtrain_reduced,ytrain,Xtest_reduced);


end



% % X_rec=score*transpose(coeff) + ones(size(Xtrain,1),1)*(mu);
% % 
% % show_image(X_rec,1)
% % show_image(Xtrain,1)
% % show_image(coeff,1)

