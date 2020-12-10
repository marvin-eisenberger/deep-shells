function X = computeLBeigenpairs(X,kMax)
    % computeLBeigenpairs - compute the spectrum of the LB operator
    %
    % X: input shape
    % kMax: basis size
    
    [X.evecs,X.evals,X.W,X.A] = HeatKernels(X,1:X.n,kMax + 10);

    startIndexX = find(X.evals>1,1)-1;

    if startIndexX ~= 1
        fixSpectrumOverheadX = startIndexX-1;
        
        [X.evecs,X.evals,X.W,X.A] = HeatKernels(X,1:X.n,kMax + 10 + fixSpectrumOverheadX);
        X.evals = X.evals((1:kMax + 10)+fixSpectrumOverheadX);
        X.evecs = X.evecs(:,(1:kMax + 10)+fixSpectrumOverheadX);
    end


    function [ Q,beta,W1,A1,rmVert ] = HeatKernels( surface ,sample, n_evecs )
    
        [W1, A1]=laplacian(surface.vert,surface.triv);

        rmVert = find(diag(A1)<1e-14);
        vertCurr = 1:size(A1,1);
        vertCurr(rmVert) = [];

        [surface.evecs,surface.evals]=eigs(W1(vertCurr,vertCurr),A1(vertCurr,vertCurr),n_evecs,1e-5);

        surface.evals = -diag(surface.evals);

        [~,sorted_idx] = sort(surface.evals,'ascend');
        surface.evals = surface.evals(sorted_idx);
        surface.evecs = surface.evecs(:,sorted_idx);

        if ~isempty(rmVert)
            evecsNew = zeros(size(surface.vert,1),n_evecs);
            evecsNew(vertCurr,:) = surface.evecs;
            evecsNew(rmVert,:) = repmat(mean(surface.evecs,1),length(rmVert),1);
            surface.evecs = evecsNew;
        end

        Q = surface.evecs(sample,:);
        beta = surface.evals;


    end
end
















