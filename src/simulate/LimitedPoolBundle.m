qqclear all
close all;

% initial variables
num = 6; % number of filaments in pool
r = 0.004; % rate of growth
d = 320.005; % degradation rate
Ntot = 80000; % total number of available monomers
MaxS = 2000050; % max steps
t1 = 2000000; % get out at this time
MaxTraj = 1; % number of trajectories
pMax= zeros(1, Ntot); %probability
p1= zeros(1, Ntot); %probability
NumLags = 40;
figure(1) % trajectory

% code for actual simulation
for j = 1:MaxTraj
    
    % initial stuff for each trajectory

    m = nan(num, MaxS); % initialize matrix that holds lengths of all filaments at each step
    mtot = nan(1, MaxS); % track sum of lengths at each step
    mmax = nan(1, MaxS); % track maximum length at each step
    m(:, 1) = 1; % how long are the filaments at the beginning?
    mtot(1) = sum(m(:, 1));
    mmax(1) = max(m(:, 1));

    monomers = Ntot; % available monomers in pool
    T = zeros(1, MaxS); % time array
    T(1) = 0; % initial time
    
    for i = 1:MaxS
        fil = randi([1 num]); % pick a filament at random
       
            k2 = r*(monomers-mtot(1)); % limited pool
            %k2 = r*Ntot; %grow free

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

        if T(i + 1) >= t1
            break;
        end
        
    end
    p1(m(1,i+1)) = p1(m(1,i+1))+1; % calculating probability of single filament lengths
    pMax(mmax(1,i+1)) = pMax(mmax(1,i+1))+1; %calculating probability of single filament lengths    ta = T(800:i);


end

% Plot individual trajectories with labels
legend_labels = cell(1, num);
for k = 1:num
    plot(T, m(k, :) - 1, '-', 'MarkerSize', 15, 'LineWidth', 2);
    legend_labels{k} = ['Fil' num2str(k)];
    hold on;
end

T1= 0:0.1:600;
%L0 = 0;
%L1 = L0*(exp(-r.*T1));
%Lss = (Ntot-d/r)/num;
%L = L1+Lss*(1- exp(-r.*T1));
%plot(T1,L,'--','LineWidth', 2, 'Color', 'k');
%legend_labels{num + 1} = 'Avg The';
% Plot the maximum length among all filaments
plot(T, mmax - 1, '-r', 'MarkerSize', 15, 'LineWidth', 2); % Adjust line properties as needed
legend_labels{num+1} = 'Max'; % Add label for the maximum length
plot(T, mtot-num, '-', 'MarkerSize', 15, 'LineWidth', 2); 
legend_labels{num + 2} = 'Tot'; 
%L01 = 0;
%L11 = L01*(exp(-r.*T1));
%Lss1 = (Ntot-d/r);
%L1 = L11+Lss1*(1- exp(-r.*T1));
%plot(T1,L1,'-','LineWidth', 2, 'Color', 'k');
legend_labels{num + 3} = 'Tot The';
% Add legend and box
legend(legend_labels, 'Location', 'Best');
xlabel('Time (s)');
ylabel('Length (monomers)');
box on;
%xlim([0 500]);
%ylim([0 60]);

% Probability distribution of the Maximum filament
pMax = pMax/sum(pMax);
%filename = 'E:\RIT\SeveringCodeFinal\Fil6LPPD\Fil6LPPDSim3.dat';
%writematrix(pMax, filename);
p1 = p1/sum(p1);
x = 1:1:Ntot;
MaxAvg = sum(x.*pMax);
Maxvariance= sum((x.^2).*pMax)-(sum(x.*pMax))^2;
SingleAvg = sum(x.*p1);
SingleVariance= sum((x.^2).*p1)-(sum(x.*p1))^2;



% Define the range of L values
L = 1:1000;

% Calculate the probability for each value of L
p = (num / SingleAvg) * (1 - exp(-L / SingleAvg)).^(num - 1) .* exp(-L / SingleAvg);

% Plot the probability as a function of L
figure(2)
plot(x,pMax,'-','LineWidth', 2, 'Color', 'r','DisplayName', 'LP Bundle Sim');
hold on
plot(L, p, 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'LP Bundle The');
xlim([0 1000])
xlabel('Length (monomers)');
ylabel('Probability distribution');
legend('Location', 'best');
% Save the figure as a PDF
%saveas(gcf, 'BundleLPPD.pdf');
print('BundleLPPD', '-dpdf', '-r300'); % '-r300' sets the resolution to 300 DPI


figure(3)
plot(x,p1,'-','LineWidth', 2, 'Color', 'r','DisplayName', 'LP Indvdl Fil');
xlim([0 1000])
xlabel('Length (monomers)');
ylabel('Probability distribution');
legend('Location', 'best');

