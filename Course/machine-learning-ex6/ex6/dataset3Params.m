function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

max_error = 999999999.9;
C_test = 0.01;
while(C_test<30)
	sigma_test = 0.01; 
	while(sigma_test < 30)
			model = svmTrain(X,y,C_test, @(x1,x2) gaussianKernel(x1,x2,sigma_test));
			preds = svmPredict(model,Xval);
			error = mean(double(preds ~= yval));
			if(error < max_error)
				max_error = error;
				C = C_test;
				sigma = sigma_test
			end
			sigma_test = sigma_test * 3;
	end
	C_test = C_test * 3;
end

% Best seems to be 
% C = 1, sigma = 0.09


% =========================================================================

end
