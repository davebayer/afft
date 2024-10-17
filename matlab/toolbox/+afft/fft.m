function Y = fft(X, varargin)
% afft.fft - Computes the forward 1D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   Y = afft.fft(X)
%   Y = afft.fft(X, n)
%   Y = afft.fft(X, n, dim)
%
% Description:
%   Y = afft.fft(X) computes the discrete Fourier transform (DFT) of X using a fast Fourier transform (FFT) algorithm.
%       Y is the same size as X.
%   - If X is a vector, then afft.fft(X) returns the Fourier transform of the vector.
%   - If X is a matrix, then afft.fft(X) treats the columns of X as vectors and returns the Fourier transform of each
%     column.
%   - If X is a multidimensional array, then afft.fft(X) treats the values along the first array dimension whose size
%     does not equal 1 as vectors and returns the Fourier transform of each vector.
%
%   Y = afft.fft(X, n) returns the n-point DFT.
%   - If X is a vector and the length of X is less than n, then X is padded with trailing zeros to length n.
%   - If X is a vector and the length of X is greater than n, then X is truncated to length n.
%   - If X is a matrix, then each column is treated as in the vector case.
%   - If X is a multidimensional array, then the first array dimension whose size does not equal 1 is treated as in the
%     vector case.
%
%   Y = afft.fft(X, n, dim) returns the Fourier transform along the dimension dim. For example, if X is a matrix, then
%       afft.fft(X, n, 2) returns the n-point Fourier transform of each row.
%
% Inputs:
%   X   - Input array, specified as a floating-point vector, matrix, or multidimensional array. If X is an empty 0-by-0
%         matrix, then afft.fft(X) returns an empty 0-by-0 matrix.
%
%   n   - Transform length, specified as [] or a nonnegative integer scalar. Specifying a positive integer scalar for
%         the transform length can improve the performance of afft.fft. The length is typically specified as a power
%         of 2 or a value that can be factored into a product of small prime numbers (with prime factors not greater
%         than 7). If n is less than the length of the signal, then afft.fft ignores the remaining signal values past
%         the nth entry and returns the truncated result. If n is 0, then afft.fft returns an empty matrix.
%
%   dim - Dimension to operate along, specified as a positive integer scalar. If you do not specify the dimension, then
%         the default is the first array dimension of size greater than 1. If dim is greater than ndims(X), then
%         afft.fft(X, [], dim) returns X. When n is specified, afft.fft(X, n, dim) pads or truncates X to length n along
%         dimension dim.
%
% Outputs:
%   Y - Frequency domain representation returned as a vector, matrix, or multidimensional array. If X is real, then Y is
%       conjugate symmetric, and the number of unique points in Y is ceil((n+1)/2).
%
% Example:
%   >> X   = magic(3);
%   >> n   = 4;
%   >> dim = 2;
%   >> Y = afft.fft(X, n, dim);
%   >> disp(Y)
%     15.0000 + 0.0000i   2.0000 - 1.0000i  13.0000 + 0.0000i   2.0000 + 1.0000i
%     15.0000 + 0.0000i  -4.0000 - 5.0000i   5.0000 + 0.0000i  -4.0000 + 5.0000i
%     15.0000 + 0.0000i   2.0000 - 9.0000i  -3.0000 + 0.0000i   2.0000 + 9.0000i
%
% See also:
%   fft, afft.ifft, ifft
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">GitHub</a>.

  if afft.internal.isAccelerated
    Y = afft.internal.afft_matlab(uint32(2000), X, varargin{:});
    return;
  end

  if ~isfloat(X)
    error('Input array must be a floating-point array.');
  end

  ip = inputParser;
  addOptional(ip, 'n', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isreal(x) && x >= 0));
  addOptional(ip, 'dim', 1, @afft.internal.isDimension);
  addParameter(ip, 'normalization', 'none', @afft.internal.isNormalization);
  addParameter(ip, 'threadLimit', 0, @afft.internal.isThreadLimit);
  addParameter(ip, 'backend', [], @afft.internal.isBackend);
  addParameter(ip, 'selectStrategy', [], @afft.internal.isSelectStrategy);

  parse(ip, varargin{:});

  Y = fft(X, ip.Results.n, ip.Results.dim);

  transformSize = size(Y, ip.Results.dim);

  if strcmp(ip.Results.normalization, 'unitary')
    Y = Y / transformSize;
  elseif strcmp(ip.Results.normalization, 'orthogonal')
    Y = Y / sqrt(transformSize);
  end
end
