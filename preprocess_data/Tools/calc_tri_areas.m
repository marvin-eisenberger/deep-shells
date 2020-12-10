function S_tri = calc_tri_areas(M)

    getDiff  = @(a,b)M.vert(M.triv(:,a),:) - M.vert(M.triv(:,b),:);
    getTriArea  = @(X,Y).5*sqrt(sum(cross(X,Y).^2,2));
    S_tri = getTriArea(getDiff(1,2),getDiff(1,3));


end
