
function [H, p] = entropy_production(states_vector,k)  
    % Author: Sebastian Geli: gelisebastianm@gmail.com
    % Adapted from Lynn et al. 2021: https://www.pnas.org/doi/abs/10.1073/pnas.2109889118
    
    %K number of states (unique clusters)
    
    transitions = [states_vector(1:end-1); states_vector(2:end)]'; % Transition matrix
    
%     % Rows to delete in order to avoid transitions between subjects or tasks
%     % Example: if Tmax = 176 and the time series is 528 (176*3) units long, 
%     % the rows to delete will be [176,352] (the first 2 multiples of 176)
%     rows2delete = (1:((length(states_vector)/Tmax)-1)).*Tmax;
%     transitions(rows2delete,:) = [];

    p = zeros(k);     % Probability matrix dim = K x K
    p = p + 0.000001; % To avoid dividing by 0 when calculating entropy
    
    for i=1:length(transitions)  %Calculates probability matrix from transition matrix
    
        p(transitions(i,1),transitions(i,2)) = p(transitions(i,1),transitions(i,2)) ...
        + 1/length(transitions);
    
    end
    H = sum(p.*log2(p./p'), "all"); 
    % Entropy production (Lynn): Bigger H, more entropy, more irreversible
end