function Bagged = random_forests(numBags,trImg,trLabs)
%https://uk.mathworks.com/help/stats/classification-ensembles.html

%% Two Steps:
%Bootstrap sample: create new training sets ?^(??)  by random sub-sampling with replacement the initial training set N? ? N times
%Some observations may end up several times in the training set, while others are not included (“out of bag”)
%Aggregation: parallel combination of classifiers ?^(??), independently trained on distinct bootstrap samples 
%(on each different subsampled dataset)
%Final prediction is class with maximum votes (for classification) or average (for regression). 
%Technique of ensemble learning: It can be used by itself to build an Ensemble of Classifiers

%Parameters - 
%	Input	
%		trImg - training images
%		trLabs - training labs
%		numBags - number of bags to use for boostrapping

%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error

%%
Bagged = TreeBagger(numBags,trImg,trLabs,'OOBPrediction','on')
%%
%% plot out of bag prediction error
oobErrorBagged = oobError(Bagged);
plot(oobErrorBagged)
xlabel 'No. Trees Grown';
ylabel 'Out-Of-Bag Classification Error';
title('Bag Prediction Error');

oobPredict(Bagged)
%%
%% view trees
view(Bagged.Trees{1}) % text description
view(Bagged.Trees{1},'mode','graph') % graphic description

end