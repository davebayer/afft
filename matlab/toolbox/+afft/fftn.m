function Y = fftn(X, varargin)
% afft.fftn - Computes the forward N-D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   Y = afft.fftn(X)
%   Y = afft.fftn(X, sz)
%
% Description:
%   Y = afft.fftn(X) returns the multidimensional Fourier transform of an N-D array using a fast Fourier transform
%       algorithm. The N-D transform is equivalent to computing the 1-D transform along each dimension of X. The output
%       Y is the same size as X.
%
%   Y = afft.fftn(X, sz) truncates X or pads X with trailing zeros before taking the transform according to the elements
%       of the vector sz. Each element of sz defines the length of the corresponding transform dimensions. For example,
%       if X is a 5-by-5-by-5 array, then Y = afft.fftn(X, [8, 8, 8]) pads each dimension with zeros resulting in an
%       8-by-8-by-8 transform Y.
%
% Inputs:
%   X  - Input array, specified as a floating-point matrix or a floating-point multidimensional array. If X is an empty
%        0-by-0 matrix, then afft.fftn(X) returns an empty 0-by-0 matrix.
%
%   sz - Length of the transform dimensions, specified as a vector of positive integers. The elements of sz correspond
%        to the transformation lengths of the corresponding dimensions of X. length(sz) must be at least ndims(X).
%
% Outputs:
%   Y - Frequency domain representation returned as a multidimensional array.
%
% Example:
%   >> X = magic(3);
%   >> sz = [4 4];
%   >> Y = afft.fftn(X, sz);
%   >> disp(Y)
%     45.0000 + 0.0000i   0.0000 -15.0000i  15.0000 + 0.0000i   0.0000 +15.0000i
%      0.0000 -15.0000i  -5.0000 +12.0000i  16.0000 - 5.0000i   5.0000 - 4.0000i
%     15.0000 + 0.0000i   8.0000 - 5.0000i   5.0000 + 0.0000i   8.0000 + 5.0000i
%      0.0000 +15.0000i   5.0000 + 4.0000i  16.0000 + 5.0000i  -5.0000 -12.0000i
%
% See also:
%   fftn, afft.ifftn, ifftn
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">GitHub</a>.

  if afft.internal.isAccelerated
    Y = afft.internal.afft_matlab(uint32(2002), X, varargin{:});
    return;
  end

  if ~isfloat(X)
    error('Input array must be a floating-point array.');
  end

  ip = inputParser;
  addOptional(ip, 'sz', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(isreal(x)) && all(x >= 0)));
  addParameter(ip, 'normalization', 'none', @afft.internal.isNormalization);
  addParameter(ip, 'threadLimit', 0, @afft.internal.isThreadLimit);
  addParameter(ip, 'backend', [], @afft.internal.isBackend);
  addParameter(ip, 'selectStrategy', [], @afft.internal.isSelectStrategy);

  parse(ip, varargin{:});

  Y = fftn(X, ip.Results.sz);

  transformSize = numel(Y);

  if strcmp(ip.Results.normalization, 'unitary')
    Y = Y / transformSize;
  elseif strcmp(ip.Results.normalization, 'orthogonal')
    Y = Y / sqrt(transformSize);
  end
end
