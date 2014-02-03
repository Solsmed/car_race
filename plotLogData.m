path = 'car_race/';



timeEpsilonLogStatic = dlmread([path 'timeLogs_10cars_e159.log']);
rewardEpsilonLogStatic = dlmread([path 'rewardLogs_10cars_e159.log']);
timeEpsilonLogStatic2 = dlmread([path 'timeLogs_10cars_e05.log']);
rewardEpsilonLogStatic2 = dlmread([path 'rewardLogs_10cars_e05.log']);

timeEpsilonLogDynamic = dlmread([path 'timeLogs_10cars_eChange_159.log']);
rewardEpsilonLogDynamic = dlmread([path 'rewardLogs_10cars_eChange_159.log']);

timeLogOptimal = dlmread([path 'timeLogs_optimal.log']);
rewardLogOptimal = dlmread([path 'rewardLogs_optimal.log']);


%timeEpsilonLogDynamic = dlmread([path 'timeLogs.log']);
%rewardEpsilonLogDynamic = dlmread([path 'rewardLogs.log']);
clf

numTrials = 1000;
numEpislonsStatic = 3;
numEpislonsDyanmic = 3;
numCars = 10;

telstatic = reshape(timeEpsilonLogStatic',[numTrials numEpislonsStatic numCars]);
relstatic = reshape(rewardEpsilonLogStatic',[numTrials numEpislonsStatic numCars]);
telmeanstatic = nanmean(telstatic,3);
relmeanstatic = nanmean(relstatic,3);

telstatic2 = reshape(timeEpsilonLogStatic2',[1000 1 10]);
relstatic2 = reshape(rewardEpsilonLogStatic2',[1000 1 10]);
telmeanstatic2 = nanmean(telstatic2,3);
relmeanstatic2 = nanmean(relstatic2,3);

teldynamic = reshape(timeEpsilonLogDynamic',[numTrials numEpislonsDyanmic numCars]);
reldynamic = reshape(rewardEpsilonLogDynamic',[numTrials numEpislonsDyanmic numCars]);
telmeandynamic = nanmean(teldynamic,3);
relmeandynamic = nanmean(reldynamic,3);

teloptimal = reshape(timeLogOptimal',[numTrials numCars]);
reloptimal = reshape(rewardLogOptimal',[numTrials numCars]);
telmeanoptimal = nanmean(teloptimal,2);
relmeanoptimal = nanmean(reloptimal,2);

% nan patch
telmeanstatic(3,3) = mean([telmeanstatic(2,3) telmeanstatic(4,3)]);
telmeandynamic(24,1) = mean([telmeandynamic(23,1) telmeandynamic(25,1)]);



B = fspecial('gaussian',[31 1]',9);
Bhalf = floor(length(B) / 2);
xt = linspace(1,numTrials,1000)';
At = [ones(size(xt)) 1./(xt.^(1/7))];
cts = (At\telmeanstatic);
ctd = (At\telmeandynamic);
cto = (At\telmeanoptimal);

% Bad fits for dynamic!
xr = linspace(1,numTrials,1000)';
Ar = [ones(size(xr)) 1./(xr.^(1/7))];
crs = (Ar\relmeanstatic);
crd = (At\relmeandynamic);
cro = (At\relmeanoptimal);


clf
figure(1)
subplot(1,2,1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,[telmeanstatic telmeanstatic2],'valid'),'-','LineWidth',1)

subtitle = 'Static epsilons .1 .5 .9 (b,g,r)';
title(sprintf(['Times\n' subtitle]))
xlim([1 500])
ylim([1 600])
ylabel('Time steps until finish line')
xlabel('Trial')

subplot(1,2,2)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,[relmeanstatic relmeanstatic2],'valid'),'-','LineWidth',1)
title(sprintf(['Rewards\n' subtitle]))
xlim([1 500])
ylim([-100 4])
ylabel('Total reward at finish line')
xlabel('Trial')



figure(2)
subplot(1,2,1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeandynamic,'valid'),'-')
%subtitle = 'Dynamic epsilons exp(-1 -5 -9 * t) (b,g,r)';
%title(sprintf(['Times\n' subtitle]))
%xlim([1 500])
ylim([1 600])
ylabel('Time steps until finish line')
xlabel('Trial')
subplot(1,2,2)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeandynamic,'valid'),'-')
hold on
%title(sprintf(['Rewards\n' subtitle]))
%xlim([1 500])
%ylim([-100 4])
ylabel('Total reward at finish line')
xlabel('Trial')





figure(3)
subplot(1,2,1)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeanstatic,'valid') ./ filter2(B,telmeanstatic,'valid'),'LineWidth',1)
title('Ratio rewards / times (static epsilon)')
xlabel('Trial')
ylim([-1 1])
subplot(1,2,2)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeandynamic,'valid') ./ filter2(B,telmeandynamic,'valid'))
title('Ratio rewards / times (dynamic epsilon)')
xlabel('Trial')
ylim([-1 1])


figure(4)
subplot(1,2,1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeandynamic(:,3),'valid'),'r-')
hold on
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeanstatic(:,1),'valid'),'b-','LineWidth',1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeanoptimal,'valid'),'k-','LineWidth',1)
ylabel('Time steps until finish line')
xlabel('Trial')
title('Time')

subplot(1,2,2)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,relmeandynamic(:,3),'valid'),'r-')
hold on
plot(xt((Bhalf+1):end-Bhalf), filter2(B,relmeanstatic(:,1),'valid'),'b-','LineWidth',1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,relmeanoptimal,'valid'),'k-','LineWidth',1)
title('Reward')
ylabel('Total reward at finish line')
xlabel('Trial')

figure(5)

%subplot(1,2,1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeanstatic(:,1),'valid'),'-','LineWidth',1)
hold on
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeanstatic(:,3),'valid'),'r-','LineWidth',1)
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeandynamic(:,1),'valid'),'k--')
plot(xt((Bhalf+1):end-Bhalf), filter2(B,telmeandynamic(:,3),'valid'),'g--')
%subtitle = 'Static epsilons .1 .5 .9 (b,g,r)';
%title(sprintf(['Times\n' subtitle]))
%xlim([1 500])
ylim([0 800])
ylabel('Time steps until finish line')
xlabel('Trial')

figure(6)
%subplot(1,2,2)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeanstatic(:,1),'valid'),'-','LineWidth',1)
hold on
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeanstatic(:,3),'valid'),'r-','LineWidth',1)
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeandynamic(:,1),'valid'),'k--')
plot(xr((Bhalf+1):end-Bhalf), filter2(B,relmeandynamic(:,3),'valid'),'g--')
%title(sprintf(['Rewards\n' subtitle]))
%xlim([1 500])
ylim([-150 10])
ylabel('Total reward at finish line')
xlabel('Trial')

