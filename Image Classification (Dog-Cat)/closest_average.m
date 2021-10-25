%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the average cat
%and dog vectors. It also takes in a test data matrix Xtest and 
%produces a vector of label guesses yguess, corresponding to whether
%each row of Xtest is closer to the average cat or average dog.

function [yguess] = closest_average(Xtrain,ytrain,Xtest)
    [avgcat,avgdog] = average_pet(Xtrain,ytrain);
    for j=1:size(Xtest,1)
        if norm((Xtest(j,:)-avgcat),2)<= norm((Xtest(j,:)-avgdog),2)
            yguess(j,1)=-1;
        else
            yguess(j,1)=1;
        end
    end
end


