function X = compute_shot_feats(X)
    Xnormalize = X;
    refarea = 1.93e+04;
    Xnormalize.vert = Xnormalize.vert ./ sqrt(sum(calc_tri_areas(Xnormalize))) .* sqrt(refarea);

    Xnormalize.area = sum(calc_tri_areas(Xnormalize));

    opts = struct;
    opts.shot_num_bins = 10; % number of bins for shot
    opts.shot_radius = 5; % percentage of the diameter used for shot

    X.SHOT = calc_shot(Xnormalize.vert', Xnormalize.triv', 1:Xnormalize.n, opts.shot_num_bins, opts.shot_radius*sqrt(Xnormalize.area)/100, 3)';
end