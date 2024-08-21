dims = [128, 64, 32];                               % these are the dimensions of the data

transformParams.type          = 'dft';              % let's do a DFT
transformParams.direction     = 'forward';          % in the forward direction
transformParams.precision     = 'double';           % of double precision
transformParams.shape         = dims;               % of size dims
transformParams.axes          = 1:ndims(dims);      % along all axes (could be removed, all axes are implict)
transformParams.normalization = 'none';             % without normalization
transformParams.type          = 'complexToComplex'; % on complex data (could be specified as 'c2c' as well)

targetParams.type             = 'cpu';              % the transform will be executed on the CPU
targetParams.threadLimit      = 4;                  % using maximum 4 threads

dftPlan = afft.Plan(transformParams, targetParams); % create the plan

X = rand(dims, 'like', 1i);                         % create input data
Y = dftPlan.execute(X);                             % execute the plan

disp(Y);                                            % display the result
