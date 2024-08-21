dims = [128, 64, 32];

transformParams.type          = 'dft';
transformParams.direction     = 'forward';
transformParams.precision     = 'double';
transformParams.shape         = dims;
transformParams.normalization = 'none';
transformParams.type          = 'complexToComplex'; % or 'c2c'

targetParams.type             = 'cpu';
targetParams.threadLimit      = 4;


dftPlan = afft.Plan(transformParams, targetParams);

X = rand(dims, 'like', 1i);
Y = dftPlan.execute(X);

disp(Y);
