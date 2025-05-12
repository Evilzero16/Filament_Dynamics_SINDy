clear all
close all;

% initial variables
num = 1; % number of filaments in pool
r = 0.3; % Growth rate
d = 225; % Severing rate
Ntot = 1000; % Total number of available monomers
MaxS = 300000; % Max steps
MaxTraj = 5000; % Number of trajectories
t1_values = 0:1:40; % t1 values for Fil1 0:1:30;
Max_mean_values = zeros(1, length(t1_values)); % for storing mean values
Max_variance_values = zeros(1, length(t1_values)); % for storing variance values
Min_mean_values = zeros(1, length(t1_values)); % for storing mean values
Min_variance_values = zeros(1, length(t1_values)); % for storing variance values
Single_mean_values = zeros(1, length(t1_values)); % for storing mean values
Single_variance_values = zeros(1, length(t1_values)); % for storing variance values
Single_SD_values = zeros(1, length(t1_values)); % for storing SD values

Tot_mean_values = zeros(1, length(t1_values)); % for storing mean values
Tot_variance_values = zeros(1, length(t1_values)); % for storing variance values
Tot_SD_values = zeros(1, length(t1_values)); % for storing SD values


for t = 1:length(t1_values)
    t1 = t1_values(t); % current t1 value

    % code for actual simulation

    pMax = zeros(1, Ntot * 1); % probability
    pMin = zeros(1, Ntot * 1); % probability
    p1 = zeros(1, Ntot * 1);
    pTot = zeros(1, Ntot * 1);

    for j = 1:MaxTraj

        % initial stuff for each trajectory

        m = nan(num, MaxS); % initialize matrix that holds lengths of all filaments at each step
        mtot = nan(1, MaxS); % track sum of lengths at each step
        mmax = nan(1, MaxS); % track maximum length at each step
        mmin = nan(1, MaxS); % track maximum length at each step
        m(:, 1) = 1; % how long are the filaments at the beginning?
        mtot(1) = sum(m(:, 1));
        mmax(1) = max(m(:, 1));
        mmin(1) = min(m(:, 1));

        monomers = Ntot; % available monomers in pool
        T = zeros(1, MaxS); % time array
        T(1) = 0; % initial time

        for i = 1:MaxS

            fil = randi([1 num]); % pick a filament at random
            k2 = r*(monomers-mtot(1)); % limited pool
            %k2 = r*Ntot;
            k1 = d;
                                
            % Determine time spent
    
            k0=k1+k2;
            CoinFlip1=rand;
            tau(i)=(1/k0)*log(1/CoinFlip1); %also, tau(i)=exprnd(1/k0);
            
            T(i+1)= T(i)+tau(i);
            
            % Determine reaction
            CoinFlip2=rand;

            if CoinFlip2<=(k1/k0)
                if m(fil,i)==1 % filament cannot decay past one unit
                    m(fil,i+1)= m(fil,i);
                else
                    m(fil,i+1) = m(fil,i)-1; % lose one unit
                    monomers = monomers + 1;
                end
             
            else
                m(fil,i+1)=m(fil,i)+1; % grow one unit
                monomers = monomers - 1;
                          
            end

            for newind = 1:num
                if newind ~= fil % keep all other filament at their previous length
                    m(newind, i + 1) = m(newind, i);
                end
            end

            mtot(i + 1) = sum(m(:, i + 1), 1);
            mmax(i + 1) = max(m(:, i + 1));
            mmin(i + 1) = min(m(:, i + 1));

            if T(i + 1) >= t1
                break;
            end

        end
    p1(m(1,i+1)) = p1(m(1,i+1))+1; %calculating probability of single filament lengths
    pMax(mmax(1, i + 1)) = pMax(mmax(1, i + 1)) + 1; % calculating probability of single filament lengths
    pMin(mmin(1, i + 1)) = pMin(mmin(1, i + 1)) + 1;
    pTot(mtot(1, i + 1)) = pTot(mtot(1, i + 1)) + 1;

    end
%
    % Probability distribution of the Maximum filament
    pMax = pMax / sum(pMax);
    x = 1:1:Ntot* 1;
    MaxAvg = sum(x .* pMax);
    MaxVariance = sum((x .^ 2) .* pMax) - (sum(x .* pMax))^2;
    
    % Store mean and variance values
    Max_mean_values(t) = MaxAvg;
    Max_variance_values(t) = MaxVariance;
    
    % Probability distribution of the Minimum filament
    pMin = pMin / sum(pMin);
    x = 1:1:Ntot * 1;
    MinAvg = sum(x .* pMin);
    MinVariance = sum((x .^ 2) .* pMin) - (sum(x .* pMin))^2;
    
    % Store mean and variance values
    Min_mean_values(t) = MinAvg;
    Min_variance_values(t) = MinVariance;
% 
    % Probability distribution of the Average/Single filament
    p1 = p1 / sum(p1);
    x = 1:1:Ntot * 1;
    SingleAvg = sum(x .* p1);
    SingleVariance = sum((x .^ 2) .* p1) - (sum(x .* p1))^2;
    SingleSD = sqrt(SingleVariance);
    % Store mean and variance values of Single Fil
    Single_mean_values(t) = SingleAvg;
    Single_variance_values(t) = SingleVariance;
    Single_SD_values(t) = SingleSD;
    
    % Probability distribution of the Average/Single filament
    pTot = pTot / sum(pTot);
    x = 1:1:Ntot * 1;
    TotAvg = sum(x .* pTot);
    TotVariance = sum((x .^ 2) .* pTot) - (sum(x .* pTot))^2;
    TotSD = sqrt(TotVariance);

    % Store mean and variance values of Single Fil
    Tot_mean_values(t) = TotAvg;
    Tot_variance_values(t) = TotVariance;
    Tot_SD_values(t) = TotSD;
end



figure(1);
% Define the exponential function to fit
exponential_function = @(params, t1_values) params(1) * (1 - exp(-params(2) * t1_values));

% Initial guess for parameters
initial_guess = [Single_mean_values(end), 0.1];

% Fit the data
params_fit = lsqcurvefit(@(params, t1_values) exponential_function(params, t1_values), initial_guess, t1_values, Single_mean_values-1);

% Extracting the fitted parameters
Lk_fit = params_fit(1);
k_fit = params_fit(2);

% Plot the original data
plot(t1_values, Single_mean_values-1, '.','MarkerSize',25, 'LineWidth', 2, 'Color', 'b', 'DisplayName', 'Sev Sim');
hold on;
% Plot the fitted curve
plot(t1_values, exponential_function(params_fit, t1_values), '-','MarkerSize',25, 'LineWidth', 2, 'Color', 'b', 'DisplayName', 'Fitted Curve');

xlabel('time');
ylabel('Length (monomers)');
%title('Fitting Your Data');
legend('Location', 'best');
%xlim([0 30])
%ylim([0 30])
fprintf('Fitted LAvg: %f\n', Lk_fit);
fprintf('Fitted kAvg: %f\n', k_fit);

figure(2);
% Define the exponential function to fit
exponential_function1 = @(params, t1_values) params(1) * (1 - exp(-params(2) * t1_values));

% Initial guess for parameters
initial_guess1 = [Max_mean_values(end), 0.1];

% Fit the data
params_fit1 = lsqcurvefit(@(params, t1_values) exponential_function1(params, t1_values), initial_guess1, t1_values, Max_mean_values-1);

% Extracting the fitted parameters
Lk_fit1 = params_fit1(1);
k_fit1 = params_fit1(2);

% Plot the original data
plot(t1_values, Max_mean_values-1, '.','MarkerSize',25, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Sev Sim');
hold on;

% Plot the fitted curve
plot(t1_values, exponential_function1(params_fit1, t1_values), '-','MarkerSize',25, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Fitted Curve');
xlabel('time');
ylabel('Bundle Length');
%title('Fitting Your Data');
legend('Location', 'best');
%xlim([0 6])
fprintf('Fitted Lk1Max: %f\n', Lk_fit1);
fprintf('Fitted k1Max: %f\n', k_fit1);

%{
filename = 'C:\RIT\LPFil1T.dat';
writematrix(t1_values, filename);
%filename = 'E:\RIT\Researcher\Hrishit\LP\Fil6SM.dat';
%writematrix(Single_mean_values, filename);
filename = 'C:\RIT\LPFil1M.dat';
writematrix(Max_mean_values, filename);
%}
