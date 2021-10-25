function [w,b] = train_perceptron(X,y)
X_Tilde=cat(2,ones(size(X,1),1),X);
W_Tilde=zeros(1,size(X_Tilde,2))*rand();
N_error=1;
while(N_error~=0)   % Continue until the number of error becomes zero
        for i=1:size(X,1)
            if   sign(W_Tilde*X_Tilde(i,:)')~=y(i)
               W_Tilde=W_Tilde+y(i)*X_Tilde(i,:);

            end
        end
 % Finding the number of misclassified data points
N_error=0;
        for i=1:size(X,1)
            if   sign(W_Tilde(1,2:size(W_Tilde,2))*X(i,:)'+W_Tilde(1,1))~=y(i)
               N_error=N_error+1;
            end
        end
end
w=W_Tilde(1,2:size(W_Tilde,2));
b=W_Tilde(1,1);
end




