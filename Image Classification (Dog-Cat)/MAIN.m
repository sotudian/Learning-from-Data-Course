% Shahabeddin Sotudian

% clear
% clc
% [X, y] = read_data();
% [avgcat avgdog] = average_pet(X,y);
% show_image(avgdog,1)
% show_image(avgcat,1)
% 

% 80% of our data for training and 20% for testing
% [Xtrain, ytrain, Xtest, ytest] = split_data(X,y,20);


% %% Q 7.2
% [yguess_Test] = closest_average(Xtrain,ytrain,Xtest);
% Accuracy_Test = calculate_accuracy(ytest,yguess_Test)
% [yguess_Train] = closest_average(Xtrain,ytrain,Xtrain);
% Accuracy_Train = calculate_accuracy(ytrain,yguess_Train)



%% Q 7.3
% [yguess_Test] = nearest_neighbor(Xtrain,ytrain,Xtest);
% Accuracy_Test = calculate_accuracy(ytest,yguess_Test)
% [yguess_Train] = nearest_neighbor(Xtrain,ytrain,Xtrain);
% Accuracy_Train = calculate_accuracy(ytrain,yguess_Train)

% %% Q 7.4
% [yguess_Test] = linear_regression(Xtrain,ytrain,Xtest);
% Accuracy_Test = calculate_accuracy(ytest,yguess_Test)
% [yguess_Train] = linear_regression(Xtrain,ytrain,Xtrain);
% Accuracy_Train = calculate_accuracy(ytrain,yguess_Train)


% %% Q 7.5
[coeff11,score11,latent11] = pca(Xtrain);
for i=1:10
%     show_image(coeff11',i)
show_image(score11,i)
%     show_image(Xtrain,i)
end


%% Q 7.6


for k=[10, 20, 50, 100]
    k
[yguess_Test] = pca_regression(Xtrain,ytrain,Xtest,k);
Accuracy_Test = calculate_accuracy(ytest,yguess_Test)
[yguess_Train] = pca_regression(Xtrain,ytrain,Xtrain,k);
Accuracy_Train = calculate_accuracy(ytrain,yguess_Train)
end
% 


