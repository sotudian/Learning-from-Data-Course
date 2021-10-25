% Shahab Sotudian
load('arrhythmia.mat')
for i=1:279
  Median_column=median(X(:,i),'omitnan');
  for j=1:452
    if isnan(X(j,i))
     X(j,i)=Median_column;
    end
  end
end

L=randperm(length(X));
% Test
Test_X=X(L(1:90),:);
Test_Y=Y(L(1:90),:);
% Train
Train_X=X(L(91:452),:);
Train_Y=Y(L(91:452),:);





