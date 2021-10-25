function [ypred, scores] = test_boosted_dt(Xte,alpha,DTCell)
T = size(DTCell,2);
SIGMA = zeros(size(Xte,1),1);
        for t = 1:T
            SIGMA = SIGMA + alpha(t)*predict(DTCell{t}, Xte);
        end
SIGMA = SIGMA./sum(abs(alpha));
scores = SIGMA;
ypred = sign(SIGMA);
end









