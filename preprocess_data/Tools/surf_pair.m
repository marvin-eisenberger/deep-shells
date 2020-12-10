function surf_pair(X,Y,opt)
    % surf_pair: display two shapes
    % 
    % X,Y: shapes
    % opt: plotting options like 'same' or 'normals'

    if ~exist('opt','var')
       opt = ''; 
    end

    if strcmp(opt,"same")
        hold off
        trisurf(X.triv,X.vert(:,1),X.vert(:,2),X.vert(:,3));
        hold on
        trisurf(Y.triv,Y.vert(:,1),Y.vert(:,2),Y.vert(:,3));
        axis equal
    else
        subplot(1,2,1)
        hold off
        trisurf(X.triv,X.vert(:,1),X.vert(:,2),X.vert(:,3));
        axis equal

        subplot(1,2,2)
        hold off
        trisurf(Y.triv,Y.vert(:,1),Y.vert(:,2),Y.vert(:,3));
        axis equal
    end

    if strcmp(opt,"normals")
        subplot(1,2,1)
        hold on
        quiver3(X.vert(:,1),X.vert(:,2),X.vert(:,3),X.normal(:,1),X.normal(:,2),X.normal(:,3),2,'Color','red');
        
        subplot(1,2,2)
        hold on
        quiver3(Y.vert(:,1),Y.vert(:,2),Y.vert(:,3),Y.normal(:,1),Y.normal(:,2),Y.normal(:,3),2,'Color','red');
    end

    drawnow

end




























