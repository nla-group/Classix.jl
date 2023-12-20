clear all
ari = @(a,b) rand_index(double(a),double(b),'adjusted');
load Phoneme_z.mat

% MATLAB CLASSIX
no_mex = struct('use_mex',0);
tic
[label, explain, out] = classix(data,0.445,8,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))

% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.445,8);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))

%% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.445, verbose=0, minPts=int32(8));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
