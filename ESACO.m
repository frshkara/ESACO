function [pheromones] = ESACO(X, fCorr, pheromones)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       X           - n × d matrix, d dimensional training set with n patterns
%       nCycle      - maximum number of cycles that algorithm repeated.
% Output:
%       ph          - final 1×d pheromone vector for each feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, nFeautures] = size(X);
nAnt = 30;
decay = 0.1;
nCycle = 30;

if nargin < 5 || isempty(pheromones)
    pheromones = zeros(1, nFeautures) + 0.5;    
end

% NF - the number of features selected by each agent in each cycle.
NF = 50;%floor(nFeautures / 20);

for c=1:nCycle
    fprintf('cycle : %i \n', c);
    randomAnts = randperm(nFeautures, nAnt);
    FC = zeros(nAnt, nFeautures);

    
    parfor ant=1:nAnt
        FC(ant, :) = moveAnt(randomAnts(ant), pheromones, ...
            NF, nFeautures, fCorr);
    end 
    
    FC = sum(FC);
    pheromones = ( (1 - decay) * pheromones) + ( FC ./ sum(FC) );
  
end
end

function [featureCounter] = moveAnt (ant, pheromone, NF, nFeautures, fCorr)    

visitedFeatures = zeros(1, nFeautures);
visitedFeatures(ant) = 1;
featureCounter = zeros(1, nFeautures);
% exploreExploitCoeff = 1;
beta = 0.7;
explore = 0;
exploit = 0;
counter = 0;

while counter < NF    
    exploreExploitCoeff = 1 * (1 - counter/NF)^0.7;

    if rand() > exploreExploitCoeff
        exploit = exploit + 1;
        next = heuristic(ant, pheromone, visitedFeatures, fCorr, beta);
    else 
        explore = explore + 1;
        next = probability(ant, pheromone, visitedFeatures, fCorr, beta);
    end
    
    if isempty(ant) || isempty(next)
        disp("HERE");
        error("Cannot find next feature!");
    end
    
    visitedFeatures(next) = 1;
    ant = next;
    featureCounter(next) = featureCounter(next) + 1;
    counter = counter + 1;
end

end

function [next] = heuristic (ant, p, visited, fCorr, beta) 
p(~~visited) = 0;
sim = fCorr;
sim = (1 ./ sim) .^ beta;
temp = p .* sim ;
[~, next] = max(temp);
end

function [next] = probability(ant, p, visited, fCorr, beta)
p(~~visited) = 0;
sim = fCorr;
sim = ( 1 ./ sim) .^ beta;
prod = p .* sim ;
prob = (prod) ./ sum(prod);
cs = cumsum(prob);
next = find(cs>rand,1);
end

