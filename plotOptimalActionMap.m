path = '';

optimalActions = dlmread([path 'optimalActions_100x100x50.log']);
res = 100;
numFrames = length(optimalActions) / (res^2);
numTrials = 1000;

angle = (0:-45:-359)';
angle = angle([7 8 1:6]);
directions(1,:) = [0 0];
%directions(2:9,:) = [cos(deg2rad(angle)) sin(deg2rad(angle))];
directions(2:9,:) = [cos(angle*pi/180) sin(angle*pi/180)];

%{
lightSource = 90;
shade(1,:) = [0.5 0.5 0.5];
angDist = min(lightSource - angle, mod(angle - lightSource,360));
shadeCoef = 1 - angDist/180;
shade(2:9,:) = repmat(shadeCoef,[1 3]);
%}

optimalActions2D = reshape(optimalActions,res,res,[]);
%optimalActions2D = permute(optimalActions2D, [1 2 3]);

qres = 1:4:res;
[X, Y] = meshgrid(qres,qres);
index_num = 0;
%for i = 1:numFrames
    clf
for i = [3,10,20,30,40,47] %3,10,20,30,40,47
    index_num = index_num + 1;
    subplot(3,2,index_num)
    image(optimalActions2D(:,:,i))
    colormap(jet(8))
    hold on
    
    alist = reshape(optimalActions2D(:,:,i),[],1);
    ulist = directions(alist+1,1);
    vlist = directions(alist+1,2);
    
    U = reshape(ulist,res,res);
    V = reshape(vlist,res,res);
    
    quiver(X,Y,U(qres,qres),V(qres,qres),'k')
    
    x = linspace(0,1,100*res);
    y = sin(pi*x);
    ww = 0.01;
    wpy = 0.5;
    y(((x > wpy-ww) .* (x < wpy+ww) == 1)) = 0.85;
    fill(res*[0 x 1.1 0],[1 y 1.1 1.1]*res,[0.8 0.8 0.8],'LineWidth',1)
    %plot(res*[0.5,0.5],res*[0.85,1.0],'LineWidth',5,'Color',[0.8 0.8 0.8])
    
    x = linspace(0,1,100*res);
    y = (sin(pi*x)-0.3);
    wpy = 0.4;
    y(((x > wpy-ww) .* (x < wpy+ww) == 1)) = 0.80;
    fill(x*res,y*res,[0.8 0.8 0.8],'LineWidth',1)
    %plot(res*[0.4, 0.4],res*[0.8, 0.64],'LineWidth',5,'Color',[0.8 0.8 0.8])
    
    
    
    
    title(sprintf('Optimal-action map\nAfter %d trials, epsilon = 0.1',i*numTrials/numFrames))
    
    axis xy
    pause(0)%3 * exp(-5 * i/numFrames))
    drawnow
end