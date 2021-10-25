%This function takes in a data matrix X and a label
%vector y and outputs the average cat image and average dog image.

function [avgcat,avgdog] = average_pet(X,y)
avgdog=zeros(1,size(X,2));
avgcat=zeros(1,size(X,2));
for i=1:size(X,1)
    if y(i)==1
     avgdog=avgdog+X(i,:);
    else
     avgcat=avgcat+X(i,:);   
    end
end
avgdog=avgdog/size(find(y==1),1);
avgcat=avgcat/size(find(y==-1),1);
end


