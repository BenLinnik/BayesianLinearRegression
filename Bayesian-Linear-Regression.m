%% Benjamin Linnik
%% Recreate Bayesian Linear Regression plots from Christopher M. Bishop - Pattern Recognition and Machine Learning p. 157-158
%% Part 1

clear; close all; % clear everything

%rng(5); % start seed, make deterministic
n = 20; % number of data points
s = 0.1; % used for gaussian basis function width
alpha = 100; % initial spread
betaInv = (0.3)^2; % beta^-1, used as sigma for noise estimation
beta = 1/betaInv; 

XAxisPrecision = 200 ;

%[X, t, Xsin, Ysin] = genData(n, L)
X = rand(n,1); % data sampling, evenly distributed
X = sort(X) % sort ascending (for convinience)
noise = randn(n,1); % sample n normal distributed variables
t = sin(2*pi*X)+sqrt(betaInv)*noise % n targets get generated, with noise added
Xsin = linspace(0,1,XAxisPrecision); % used for plotting, evenly distributed X's
Ysin = sin(2*pi*Xsin); % used for plotting of underlying function without noise

% plot sampled data and underlying function
plotCurveBar( Xsin, Ysin, ones(1,XAxisPrecision)*sqrt(betaInv) );
hold on;
plot(X,t,'ob');
title("Data sampled");
ylabel("t");
hold off;

% get dimension of input
%[n,L] = size(X);
M = 10; % 10 basis functions
% take only first column, if matrix was given as input
%X = X(:,1)
xbar = mean(X,1); % x mean
tbar = mean(t,1); % t mean

XM = ones(n,M-1).*X; % help matrix to construct Phi
% help matrix to construct Phi, eveny distributed mean values of gaussianbasis functions
mu = ones(n,M-1).*linspace(0,1,M-1); 

Phi = exp(-((XM-mu).^2)/(2*s^2)); % part of Phi matrix
Phi0 = ones(n,1)*tbar; % constant part of Phi matrix

Phi = [Phi0 Phi] % complete Phi matrix

figure
hold on;
phi = [];
for index = 1:M-1
    phi = [phi exp(-((Xsin'-mu(1,index)).^2)/(2*s^2))];
    plot(Xsin,phi(:,index)); % plot basis functions for control
end
titlestr = sprintf('Gaussian basis functions, without weights, s = %.1f',s);
title(titlestr);
phi = [tbar*ones(XAxisPrecision,1) phi];
hold off;
SNInv = alpha * diag(M)+ beta*(Phi'*Phi); %S_N^-1
SN = inv(SNInv) %S_N 
mN = beta * SN * Phi' *t  %m_N, Bayesian method for weight calclation 
%mN = beta * ( SNInv \ Phi' *t ) %m_N same result, just debugging
mNphi = (mN'.*ones(XAxisPrecision,M)).*phi; % weights * basis functions
estfctmNphi = []; % sum of all basis functions
for index = 1:XAxisPrecision
    estfctmNphi = [estfctmNphi sum(mNphi(index,:))]; % sum each basis function at given x
end

figure
plot (Xsin, mNphi) % plot each weighted basis function
title("Weighted basis functions")

sigmaSquared = []; % calculate sigma squared

for index = 1:n
    %sigmaSquaredAtData = [sigmaSquaredAtData 1/beta + Phi(index,:)*SN*Phi(index,:)'];
    sigmaSquared = [sigmaSquared 1/beta + phi(index,:)*SN*phi(index,:)'];
end
sigmaSquared % control and debugging
sigmaSquaredi = pchip(X',sigmaSquared,Xsin); % make smoother for plotting

figure
% plot sampled data and underlying function
plotCurveBar( Xsin, estfctmNphi, sqrt(sigmaSquaredi) ); % make plot like Bishop
hold on;
titlestr = sprintf('n = %d, s = %.1f',n,s);
plot( Xsin, Ysin, 'g-', 'linewidth',2); % plot sampling function without noise
plot( X, t, 'bo', 'linewidth',2); % plot the data
xlim([0 1]) % set limits for axis
ylim([-2 2])
xlabel("x")
ylabel("t")
title(titlestr)
hold off;

figure % same plot again, but without limits on axis
% plot sampled data and underlying function
plotCurveBar( Xsin, estfctmNphi, sqrt(sigmaSquaredi) );
hold on;
plot( Xsin, Ysin, 'g-', 'linewidth',2);
plot( X, t, 'bo', 'linewidth',2);
xlim([0 1])
xlabel("x")
ylabel("t")
titlestr = sprintf('Unzoomed, n = %d, s = %.1f',n,s);
title(titlestr)
hold off;

%%

%% other method to plot
cholSN = chol(SN); % used to calculate sigma of posterior distributions
mNphiLower = ((mN'-(ones(1,M)*cholSN)).*ones(XAxisPrecision,M)).*phi; % m_N * phi - sigma
estfctmNphiLower = []; % sum of all basis functions ; % m_N * phi - sigma
for index = 1:XAxisPrecision
    % sum each basis function at given x ; % m_N * phi - sigma
    estfctmNphiLower = [estfctmNphiLower sum(mNphiLower(index,:))]; 
end
estfctmNphiLower;

 % weights * basis functions; % m_N * phi + sigma
mNphiUpper = ((mN'+(ones(1,M)*cholSN)).*ones(XAxisPrecision,M)).*phi;
estfctmNphiUpper = []; % sum of all basis functions % m_N * phi + sigma
for index = 1:XAxisPrecision
     % sum each basis function at given x % m_N * phi + sigma
    estfctmNphiUpper = [estfctmNphiUpper sum(mNphiUpper(index,:))];
end
estfctmNphiUpper;


figure
% plot sampled data and underlying function
color = [255,228,225]/255; %pink
fill([Xsin,fliplr(Xsin)],[estfctmNphiUpper,fliplr(estfctmNphiLower)],color,'LineStyle','none');
hold on;
plot( Xsin, estfctmNphi, 'r-', 'linewidth',2);
plot( Xsin, Ysin, 'g-', 'linewidth',2);
plot( X, t, 'bo', 'linewidth',2);
xlabel("x")
ylabel("t")
titlestr = sprintf('Other method, posterior bases, n = %d, s = %.1f',n,s);
title(titlestr)
xlim([0 1])
ylim([-4 4])
hold off;


%% Part 2
%%


w = randn(n,1); % sample n normal distributed variables
mN' % for comparison, show average mN
% repmatrepmat(A,n, m) returns an array containing n x m copies of A
% in the row and column dimensions
mN_array = repmat(mN',5,1) + randn(5,M)*cholSN % generate n-dimensional matrix
% first 'randn'
% creates a normal distribution with 5 x M values
% Afterwards scale this distribution to the correct covariance with cholSN
% Finally move it to the correct mean value by adding the mN vector

estfctmNphiArray = []; % sum of all basis functions
for mN_i = 1:5
     % weights * basis functions
    mNphi = (mN_array(mN_i,:).*ones(XAxisPrecision,M)).*phi;
    estfctmNphi = []; % sum of all basis functions
    for index = 1:XAxisPrecision
         % sum each basis function at given x
        estfctmNphi = [estfctmNphi sum(mNphi(index,:))];
    end
    estfctmNphi';
    estfctmNphiArray = [estfctmNphiArray estfctmNphi'];
end
estfctmNphiArray = estfctmNphiArray';

figure % same plot again, but without limits on axis
% plot sampled data and underlying function
%plotCurveBar( Xsin, estfctmNphi, sqrt(sigmaSquaredi) );
plot( Xsin, estfctmNphi );
hold on;
plot( Xsin, estfctmNphiArray, 'r-', 'linewidth',1);
plot( Xsin, Ysin, 'g-', 'linewidth',2);
plot( X, t, 'bo', 'linewidth',2);
xlim([0 1])
xlabel("x")
ylabel("t")
ylim([-3 3])
titlestr = sprintf('Part 2, n = %d, s = %.1f',n,s);
title(titlestr)
hold off;






%% 
%
