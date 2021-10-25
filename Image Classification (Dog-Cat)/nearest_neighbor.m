%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the average cat
%and dog vectors. It also takes in a test data matrix Xtest and 
%produces a vector of label guesses yguess. Each guess is found
%by searching through Xtrain to find the closest row, and then 
%outputting its label.

function yguess = nearest_neighbor(Xtrain,ytrain,Xtest)

    for j=1:size(Xtest,1)
            for m=1:size(Xtrain,1)
              Dist(m)= norm((Xtest(j,:)-Xtrain(m,:)),2);
            end
            i_closest=find(Dist==min(Dist));
            yguess(j,1)= ytrain(i_closest(1));
    end
end

