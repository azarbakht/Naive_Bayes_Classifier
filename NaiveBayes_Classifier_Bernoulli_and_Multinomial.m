% Naive Bayes classifier
% Authors: Mandana Hamidi and Amir Azarbakht
% azarbaam@eecs.oregonstate.edu
% 2014-04-25

clc;
clear all;
close all;

% HOW TO:
% Bernoulli or multinomial Naive Bayes classifier
% set the following testMethod variable for the method you wanna use to 0 or 1, for Bernoulli or multinomial respectively;
constTestMethod_Bernoulli = 0;
constTestMethod_multinomial = 1;
% method = {'Bernoulli', 'multinomial'};
testMethod = 0;

TrainData = textscan(fopen('data/train.data'),'%d %d %d');% read in training data
TrainLabels = textscan(fopen('data/train.label'),'%d');% read in train labels
Vocabs = textscan(fopen('data/vocabulary.txt'),'%s');% read dictionary vocabulary set

% Train data
% documentLabel: in this matrix, line number is docID
docID(:,1) = TrainData{1};
wordID(:,1) = TrainData{2};
wordCount(:,1) = TrainData{3};% word count in training examples
documentLabel(:,1) = TrainLabels{1}(docID(:));

% Test data
% documentLabelTest: in this matrix, line number is docID
TestData = textscan(fopen('data/test.data'),'%d %d %d');% read in test data
TestLabels = textscan(fopen('data/test.label'),'%d'); % read in test label

docIDTest(:,1) = TestData{1};% Test data
wordIDTest(:,1) = TestData{2}; % worID in the test data
wordCountTest(:,1) = TestData{3}; % word count

clear TestData fTestLabel;% TestLabels;

numberOfVocabulary = size(Vocabs{1},1);
numberOfDocs = size(TrainLabels{1},1);
numberOfCategories = max(TrainLabels{1});

clear TrainData fTrainLabel TrainLabels;

% *********************   LEARNING Phase ****************************************
numberOfEachWordInCat = zeros(numberOfCategories,numberOfVocabulary);

% computes number of words in a documents
if testMethod == constTestMethod_Bernoulli
    for i = 1: size(documentLabel,1)
        numberOfEachWordInCat(documentLabel(i,1), wordID(i,1)) = numberOfEachWordInCat(documentLabel(i,1), wordID(i,1)) + 1;
    end
    
elseif testMethod == constTestMethod_multinomial
    for i = 1: size(documentLabel,1)
        numberOfEachWordInCat(documentLabel(i,1), wordID(i,1)) = numberOfEachWordInCat(documentLabel(i,1), wordID(i,1)) + wordCount(i,1);
    end
end

prior = zeros(numberOfCategories, 1);
probEachWordInCat = zeros(size(numberOfEachWordInCat));
sumAll = sum(sum(numberOfEachWordInCat(:,:)));

for i = 1: numberOfCategories
    sumRow = sum(numberOfEachWordInCat(i,:));
    prior(i) = sumRow/sumAll;
    
    if testMethod == constTestMethod_Bernoulli
        % with laplacian smoothing
        probEachWordInCat(i,:) = double(numberOfEachWordInCat(i,:) + 1) / double(sumRow + 2);
        
    elseif testMethod == constTestMethod_multinomial
        % with laplacian smoothing
        probEachWordInCat(i,:) = double(numberOfEachWordInCat(i,:) + 1) / double(sumRow + numberOfVocabulary);
    end
end

% *******************   TEST Phase  ********************************
save  trainNaiveBayesTrain
clear docID wordID  wordCount documentLabel numberOfEachWordInCat 
numberOfEachWordInDocTest = zeros(numberOfVocabulary,1);

predictedLabel = zeros(1, max(docIDTest));  % to see if the predicted label was predicted correctly or not
accuracy = Inf(1, max(docIDTest));
accuracyWithinCat = zeros(numberOfCategories,1);

for docIDTestIndex =1:max(docIDTest) % loop over each document
    docIDTestIndex
    docBounds = (find (docIDTest(:,1)  == docIDTestIndex));
    % computes number of words in a documents
    
    if testMethod == constTestMethod_Bernoulli
        for i = min(docBounds):max(docBounds)
            numberOfEachWordInDocTest(wordIDTest(i,1)) = numberOfEachWordInDocTest(wordIDTest(i,1)) + 1;
        end
    elseif testMethod == constTestMethod_multinomial
        for i = min(docBounds):max(docBounds)
            numberOfEachWordInDocTest(wordIDTest(i,1)) = numberOfEachWordInDocTest(wordIDTest(i,1)) + wordCountTest(i,1);
        end
    end
    
    posterior = zeros(1, numberOfCategories);
    % compute the likelihood and posterior for this doc
    for counterCategory = 1:numberOfCategories
        likelihood = 0;
        for wordIndex = 1:size(numberOfEachWordInDocTest)
            likelihood = likelihood + ...
                numberOfEachWordInDocTest(wordIndex) * log(probEachWordInCat(counterCategory, wordIndex)) + ...
                (1 - numberOfEachWordInDocTest(wordIndex)) * log(1 - probEachWordInCat(counterCategory, wordIndex));
        end
        
        posterior(counterCategory) = likelihood + log(prior(counterCategory));
    end
    
    % predicte the category
    [val, predictedLabel(docIDTestIndex)] = max(posterior);
    
    if predictedLabel(docIDTestIndex) == TestLabels{1}(docIDTestIndex)%documentLabelTest(docBounds(1),1)
        accuracy(docIDTestIndex) = 1;
        accuracyWithinCat(TestLabels{1}(docIDTestIndex),1) = accuracyWithinCat(TestLabels{1}(docIDTestIndex),1) + 1;
        
    elseif predictedLabel(docIDTestIndex) ~= TestLabels{1}(docIDTestIndex)
        accuracy(docIDTestIndex) = 0;
    end
    numberOfEachWordInDocTest = zeros(numberOfVocabulary,1);
end

accuracy = sum(accuracy) / size(accuracy,2);

% confusion matrix
 documentLabelTest(:,1) = TestLabels{1}(docIDTest(:));
confusionMatrix = zeros(numberOfCategories, numberOfCategories);
for i = 1: numberOfCategories
    catBounds = (find (documentLabelTest(:,1)  == i));
    for k = min(catBounds):max(catBounds)
        confusionMatrix(i,predictedLabel(docIDTest(k,1))) = confusionMatrix(i,predictedLabel(docIDTest(k,1))) + 1;
    end
end

%compute total accuracy
maxVal = max(docIDTest);
accuracyTotal = double( sum(accuracyWithinCat))/double( maxVal);


