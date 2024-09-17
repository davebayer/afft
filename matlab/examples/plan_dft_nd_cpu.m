% these are the dimensions of the data
dims = [128, 64, 32];

% let's do a DFT
transformParams.transform     = 'dft';
% in the forward direction
transformParams.direction     = 'forward';
% of double precision
transformParams.precision     = 'double';
% of size dims
transformParams.shape         = dims;
% along all axes (could be removed, all axes are implict)
transformParams.axes          = 1:ndims(dims);
% without normalization
transformParams.normalization = 'none';
% on complex data (could be specified as 'c2c' as well)
transformParams.type          = 'complexToComplex';

% the transform will be executed on the CPU (could be removed, the CPU is the default target)
targetParams.target = 'cpu';

% using maximum 4 threads
backendParams.threadLimit       = 4;
% use the FFTW_ESTIMATE planner flag
backendParams.fftw3.plannerFlag = 'estimate';

% select the first available backend (could be removed, the first available backend is the default)
selectParams.strategy = 'first';

% create the plan
dftPlan = afft.Plan(transformParams, targetParams, backendParams, selectParams);

% create input data
X = rand(dims, 'like', 1i);

% execute the plan
Y = dftPlan.execute(X);

% display the result
disp(Y);
