function Y = fft2(varargin)
% afft.fft2 - Computes the 2D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   Y = afft.fft2(X)
%   Y = afft.fft2(X, m, n)
%
% Description:
%   Y = afft.fft2(X) returns the two-dimensional Fourier transform of a matrix X using a fast Fourier transform algorithm,
%       which is equivalent to computing afft.fft(afft.fft(X).').'.
%
%       When X is a multidimensional array, afft.fft2 computes the 2-D Fourier transform on the first two dimensions of
%       each subarray of X that can be treated as a 2-D matrix for dimensions higher than 2. For example, if X is an
%       m-by-n-by-1-by-2 array, then Y(:, :, 1, 1) = afft.fft2(X(:, :, 1, 1)) and
%       Y(:, :, 1, 2) = afft.fft2(X(:, :, 1, 2)). The output Y is the same size as X.
%
%   Y = afft.fft2(X, m, n) truncates X or pads X with trailing zeros to form an m-by-n matrix before computing the
%       transform. If X is a matrix, then Y is an m-by-n matrix. If X is a multidimensional array, then afft.fft2 shapes
%       the first two dimensions of X according to m and n.
%
% Inputs:
%   X - Input array, specified as a floating-point matrix or a multidimensional array. If X is an empty 0-by-0 matrix,
%       then afft.fft2(X) returns an empty 0-by-0 matrix.
%
%   m - Number of transform rows, specified as a positive integer scalar.
%
%   n - Number of transform columns, specified as a positive integer scalar.
%
% Outputs:
%   Y - Frequency domain representation returned as a matrix or a multidimensional array. If X is real, then Y is
%       conjugate symmetric, and the number of unique points in Y is m-by-ceil((n+1)/2).
%
% Example:
%   >> X = magic(3);
%   >> m = 4;
%   >> n = 4;
%   >> Y = afft.fft2(X, m, n);
%   >> disp(Y)
%     45.0000 + 0.0000i   0.0000 -15.0000i  15.0000 + 0.0000i   0.0000 +15.0000i
%      0.0000 -15.0000i  -5.0000 +12.0000i  16.0000 - 5.0000i   5.0000 - 4.0000i
%     15.0000 + 0.0000i   8.0000 - 5.0000i   5.0000 + 0.0000i   8.0000 + 5.0000i
%      0.0000 +15.0000i   5.0000 + 4.0000i  16.0000 + 5.0000i  -5.0000 -12.0000i
%
% See also:
%   fft2, afft.ifft2, ifft2
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">afft GitHub</a>.

  Y = afft_matlab(uint32(2001), varargin{:});
end
