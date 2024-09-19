%% Using Gaussian Mixture Model to do clustering
% Input:    data        - data,
%           k           - the number of Gaaussians,
%           threshold   - the precision of the stopping threshold
% Output:   lambda      - the weight for Gaussians
%           mu          - the means for Gaussians
%           sigma       - the covariance matrix for Gaussians
function [lambda, mu, sigma] = GMM_Clustering(data, k, precision)

    [num,dim] = size(data);   % Get the size and dimension of data
    lambda = repmat(1/k,k,1); % Initialize weight for k-th Gaussian to 1/k
    
    randIdx = randperm(num);   % do randomly permutation process
    mu = data(randIdx(1:k),:); % Initialize k means for Gaaussians randomly 
    mu = [12 25; 28 35; 47 25];
    dataVariance =  cov(data,1);    % Obtain the variance of dataset ∑(x-mu)'*(x-mu)/ num
    sigma = cell (1, k);            % Store covariance matrices
    % sigma is initialized as the covariance of the whole dataset
    for i = 1 : k
        sigma{i} = dataVariance;
    end
    % x,y is used to draw pdf of Gaussians
    x=0:0.05:60;
    y=0:0.05:60;
    
    iter = 0; precious_L = 100000;
    while iter < 100
        
        % E-step (Expectation)
        gauss = zeros(num, k); % gauss - stores data generated from Gaussian distribution given mu & sigma
        for idx = 1: k
            gauss(:,idx) = lambda(idx)*mvnpdf(data, mu(idx,:), sigma{idx});
        end
        respons = zeros(num, k); % respons - stores responsibilities
        
        total = sum(gauss, 2);
        for idx = 1:num
            respons(idx, :) = gauss(idx,:) ./ total(idx);
        end

       % M-step (Maximization)
       responsSumedRow = sum(respons,1);
       responsSumedAll = sum(responsSumedRow,2);
       for i = 1 : k
          % Updata lambda
          lambda(i) =  responsSumedRow(i) / responsSumedAll;
          
          % Updata mu
          newMu = zeros(1, dim);
          for j = 1 : num
              newMu = newMu + respons(j,i) * data(j,:);
          end
          mu(i,:) = newMu ./ responsSumedRow(i);
          
          % Updata sigma
          newSigma = zeros(dim, dim);
          for j = 1 : num
              diff = data(j,:) - mu(i,:);
              diff = respons(j,i) * (diff'* diff);
              newSigma = newSigma + diff;
          end
          sigma{i} = newSigma ./ responsSumedRow(i);
       end
       
        subplot(2,2,2)
        title('Expectation Maxmization');
        hold on
        [X,Y]=meshgrid(x,y);

        stepHandles = cell(1,k);
        ZTot = zeros(size(X));

        for idx = 1 : k
           Z = getGauss(mu(idx,:), sigma{idx}, X, Y);
           Z = lambda(idx)*Z;
           [~,stepHandles{idx}] = contour(X,Y,Z);
           ZTot = ZTot + Z;
        end
        hold off
       
        subplot(2,2,3) % image 3 - PDF for 2D MOG/GMM
        mesh(X,Y,ZTot),title('PDF for 2D MOG/GMM');

        subplot(2,2,4) % image 4 - Projection of MOG/GMM
        surf(X,Y,ZTot),view(2),title('Projection of MOG/GMM')
        shading interp
        %colorbar
        drawnow();
       
       % Compute the log likelihood L
       temp = zeros(num, k);
       for idx = 1 : k
          temp(:,idx) = lambda(idx) *mvnpdf(data, mu(idx, :), sigma{idx}); 
       end
       temp = sum(temp,2);
       temp = log(temp);
       L = sum(temp);
       
       iter = iter + 1;
       preciousTemp = abs(L-precious_L);
       if  preciousTemp < precision
           break;
       else
           % delete plot handles in image 2
           for idx = 1 : k
                set(stepHandles{idx},'LineColor','none')
           end
           %set(MStepHamdle,'LineColor','none')
       end
       precious_L = L;
       
    end
    %save respons.mat respons;
    save('respons.mat', 'respons');
end 

function [Z] = getGauss(mean, sigma, X, Y)
    dim = length(mean);

    weight = 1/sqrt((2*pi).^dim * det(sigma));
    [~,row] = size(X);
    [~,col] = size(Y);
    Z = zeros(row, col);
    for i = 1 : row
        sampledData = [X(i,:); Y(i,:)]';
        sampleDiff = sampledData - mean;
        inner = -0.5 * (sampleDiff / sigma .* sampleDiff);
        Z(i,:) =  weight * exp(sum(inner,2));
    end

end