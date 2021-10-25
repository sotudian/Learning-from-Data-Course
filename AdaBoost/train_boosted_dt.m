function [alpha,DTCell] = train_boosted_dt(Xtr,ytr,T,S)
N = size(Xtr,1);
D = ones(N,1).*(1/size(ytr,1));
DTCell = {};
alpha = zeros(T,1);
        for t = 1:T
            DTCell{t} = fitctree(Xtr,ytr,'Weights',D,'MaxNumSplits',S);
            Y_Prime = predict(DTCell{t}, Xtr);
            Ep = D'*((1-ytr.*Y_Prime)./2);
            alpha(t)=log((1/Ep)-1)/2;
            D= D.*exp(-alpha(t)* ytr.*Y_Prime);
            D= D./sum(D);
        end
end





