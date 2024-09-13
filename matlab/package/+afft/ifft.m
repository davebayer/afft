function X = ifft(Y, varargin)
% afft.ifft - Computes the inverse 1D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   X = afft.ifft(Y)
%   X = afft.ifft(Y, n)
%   X = afft.ifft(Y, n, dim)
%   X = afft.ifft(___, symflag)
%
% Description:
%   X = afft.ifft(Y) computes the inverse discrete Fourier transform of Y using a fast Fourier transform algorithm. X is
%       the same size as Y.
%   - If Y is a vector, then afft.ifft(Y) returns the inverse transform of the vector.
%   - If Y is a matrix, then afft.ifft(Y) returns the inverse transform of each column of the matrix.
%   - If Y is a multidimensional array, then afft.ifft(Y) treats the values along the first dimension whose size does
%     not equal 1 as vectors and returns the inverse transform of each vector.
%
%   X = afft.ifft(Y, n) returns the n-point inverse Fourier transform of Y by padding Y with trailing zeros to length n.
%
%   X = afft.ifft(Y, n, dim) returns the inverse Fourier transform along the dimension dim. For example, if Y is a
%       matrix, then afft.ifft(Y, n, 2) returns the n-point inverse transform of each row.
%
%   X = afft.ifft(___, symflag) specifies the symmetry of Y in addition to any of the input argument combinations in
%       previous syntaxes. For example, afft.ifft(Y, 'symmetric') treats Y as conjugate symmetric.
%
% Inputs:
%   Y       - Input array, specified as a floating-point vector, a matrix, or a multidimensional array.
%
%   n       - Inverse transform length, specified as [] or a nonnegative integer scalar. Padding Y with zeros by
%             specifying a transform length larger than the length of Y can improve the performance of afft.ifft. The
%             length is typically specified as a power of 2 or a product of small prime numbers. If n is less than the
%             length of the signal, then afft.ifft ignores the remaining signal values past the nth entry and returns
%             the truncated result. If n is 0, then afft.ifft returns an empty matrix.
%   dim     - Dimension to operate along, specified as a positive integer scalar. By default, dim is the first array
%             dimension whose size does not equal 1. For example, consider a matrix Y.
%   symflag - Symmetry type, specified as 'nonsymmetric' or 'symmetric'. When Y is not exactly conjugate symmetric due
%             to round-off error, afft.ifft(Y, 'symmetric') treats Y as if it were conjugate symmetric by ignoring the
%             second half of its elements (that are in the negative frequency spectrum). For more information on
%             conjugate symmetry, see Algorithms.
%
% Outputs:
%   X - Time domain representation returned as a vector, matrix, or multidimensional array. If Y is conjugate symmetric,
%       then X is real.
%
% Example:
%   >> Y = [1 2 3 4];
%   >> n = 6;
%   >> X = afft.ifft(Y, n);
%   >> disp(X)
%      1.6667 + 0.0000i  -0.5833 + 0.7217i   0.4167 - 0.1443i  -0.3333 + 0.0000i   0.4167 + 0.1443i  -0.5833 - 0.7217i
%
% See also:
%   ifft, afft.fft, fft
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">GitHub</a>.

  X = afft_matlab(uint32(2003), Y, varargin{:});
end
