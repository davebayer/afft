function X = ifft2(Y, varargin)
% afft.ifft2 - Computes the inverse 2D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   X = afft.ifft2(Y)
%   X = afft.ifft2(Y, m, n)
%   X = afft.ifft2(___, symflag)
%
% Description:
%   X = afft.ifft2(Y) returns the two-dimensional discrete inverse Fourier transform of a matrix using a fast Fourier
%       transform algorithm. If Y is a multidimensional array, then afft.ifft2 takes the 2-D inverse transform of each
%       dimension higher than 2. The output X is the same size as Y.
%
%   X = afft.ifft2(Y, m, n) truncates Y or pads Y with trailing zeros to form an m-by-n matrix before computing the
%       inverse transform. X is also m-by-n. If Y is a multidimensional array, then afft.ifft2 shapes the first two
%       dimensions of Y according to m and n.
%
%   X = afft.ifft2(___, symflag) specifies the symmetry of Y in addition to any of the input argument combinations in
%       previous syntaxes. For example, afft.ifft2(Y, 'symmetric') treats Y as conjugate symmetric.
%
% Input Arguments:
%   Y       - Input array, specified as a floating-point matrix or a floating-point multidimensional array.
%
%   m       - Number of inverse transform rows, specified as a positive integer scalar.
%
%   n       - Number of inverse transform columns, specified as a positive integer scalar.
%
%   symflag - Symmetry type, specified as 'nonsymmetric' or 'symmetric'. When Y is not exactly conjugate symmetric due
%             to round-off error, afft.ifft2(Y, 'symmetric') treats Y as if it were conjugate symmetric by ignoring the
%             second half of its elements (that are in the negative frequency spectrum). For more information on
%             conjugate symmetry, see Algorithms.
%
% Output Arguments:
%   X - Time domain representation returned as a matrix or a multidimensional array. If Y is conjugate symmetric, then X
%       is real. Otherwise, X is complex.
% Example:
%   >> Y = magic(3);
%   >> X = afft.ifft2(Y, [4, 4]);
%   >> disp(X)
%      2.8125 + 0.0000i   0.0000 + 0.9375i   0.9375 + 0.0000i   0.0000 - 0.9375i
%      0.0000 + 0.9375i  -0.3125 - 0.7500i   1.0000 + 0.3125i   0.3125 + 0.2500i
%      0.9375 + 0.0000i   0.5000 + 0.3125i   0.3125 + 0.0000i   0.5000 - 0.3125i
%      0.0000 - 0.9375i   0.3125 - 0.2500i   1.0000 - 0.3125i  -0.3125 + 0.7500i
%
% See also:
%   ifft2, afft.fft2, fft2
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">GitHub</a>.

  if afft.internal.isAccelerated
    X = afft.internal.afft_matlab(uint32(2004), Y, varargin{:});
    return;
  end

  if ~isfloat(Y)
    error('Input array must be a floating-point array.');
  end

  if ~iscomplex(Y)
    error('Input array must be a complex array.');
  end

  if hasSymmetricFlag && hasNonsymmetricFlag
    error('Specify either ''symmetric'' or ''nonsymmetric'', but not both.');
  elseif hasSymmetricFlag
    symFlag = 'symmetric';
  else
    symFlag = 'nonsymmetric';
  end

  ip = inputParser;
  addOptional(ip, 'm', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isreal(x) && x >= 0));
  addOptional(ip, 'n', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isreal(x) && x >= 0));
  addParameter(ip, 'normalization', 'none', @afft.internal.isNormalization);
  addParameter(ip, 'threadLimit', 0, @afft.internal.isThreadLimit);
  addParameter(ip, 'backend', [], @afft.internal.isBackend);
  addParameter(ip, 'selectStrategy', [], @afft.internal.isSelectStrategy);

  parse(ip, varargin{:});

  if isempty(ip.Results.m) && isempty(ip.Results.n)
    X = ifft2(Y, symFlag);
  elseif ~isempty(ip.Results.m) && ~isempty(ip.Results.n)
    X = ifft(Y, ip.Results.m, ip.Results.n, symFlag);
  else
    error('Either none or both of m and n must be specified.');
  end

  transformSize = size(X, 1) * size(X, 2);

  if strcmp(ip.Results.normalization, 'none')
    X = X * transformSize;
  elseif strcmp(ip.Results.normalization, 'orthogonal')
    X = X * sqrt(transformSize);
  end
end
