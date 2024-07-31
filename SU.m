function [ su ] = SU( X,Y )

% % % % % % % %% Entropy H(x), H(y)
% % % % % % % Hx = entropy(X);
% % % % % % % Hy = entropy(Y);
% % % % % % %
% % % % % % % %% Conditional entropy H(x|y)
% % % % % % % Hx_y = condEntropy(X,Y);
% % % % % % % %% Information Gain
% % % % % % % IGx_y = Hx - Hx_y;
% % % % % % % %% Symmetrical uncertainty
% % % % % % % su= 2*(IGx_y/(Hx+Hy));

%% Rouhi
%function [score] = SU(firstVector,secondVector)
%
%calculates SU = 2 * (I(X;Y)/(H(X) + H(Y)))

hX = h(X);
hY = h(Y);
iXY = mi(X,Y);

su = (2 * iXY) / (hX + hY);
end



