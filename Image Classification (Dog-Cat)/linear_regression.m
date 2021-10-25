%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute ordinary-least-squares
%vector b. It also takes in a test data matrix Xtest and 
%produces a vector of label guesses yguess, corresponding to the sign
%of the linear prediction.

function yguess = linear_regression(Xtrain,ytrain,Xtest)
% OLS Solution
XX=(Xtrain')*Xtrain;
b=pinv(XX)*(Xtrain')*ytrain;  
    for j=1:size(Xtest,1)
        yguess(j,1)= sign(Xtest(j,:)*b);
        if yguess(j,1)==0
           yguess(j,1)= -1; 
        end
    end
end




