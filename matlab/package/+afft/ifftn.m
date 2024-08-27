function X = ifftn(Y, varargin)
% afft.ifftn - Computes the inverse N-D Fast Fourier Transform (FFT) of a given signal.
%
% Syntax:
%   X = afft.ifftn(Y)
%   X = afft.ifftn(Y, sz)
%   X = afft.ifftn(___, symflag)
%
% Description:
%   X = afft.ifftn(Y) returns the multidimensional discrete inverse Fourier transform of an N-D array using a fast
%       Fourier transform algorithm. The N-D inverse transform is equivalent to computing the 1-D inverse transform
%       along each dimension of Y. The output X is the same size as Y.
%
%   X = afft.ifftn(Y, sz) truncates Y or pads Y with trailing zeros before taking the inverse transform according to the
%       elements of the vector sz. Each element of sz defines the length of the corresponding transform dimension. For
%       example, if Y is a 5-by-5-by-5 array, then X = afft.ifftn(Y, [8, 8, 8]) pads each dimension with zeros,
%       resulting in an 8-by-8-by-8 inverse transform X.
%
%   X = afft.ifftn(___, symflag) specifies the symmetry of Y in addition to any of the input argument combinations in
%       previous syntaxes. For example, afft.ifftn(Y, 'symmetric') treats Y as conjugate symmetric.
%
% Inputs:
%   Y       - Input array, specified as a floating-point matrix or a floating-point multidimensional array.
%
%   sz      - Lengths of inverse transform dimensions, specified as a vector of positive integers.
%
%   symflag - Symmetry type, specified as 'nonsymmetric' or 'symmetric'. When Y is not exactly conjugate symmetric due
%             to round-off error, afft.ifftn(Y, 'symmetric') treats Y as if it were conjugate symmetric by ignoring the
%             second half of its elements (that are in the negative frequency spectrum). For more information on
%             conjugate symmetry, see Algorithms.
%
% Outputs:
%   X - Time domain representation returned as a multidimensional array. If Y is conjugate symmetric, then X is real.
%
% Example:
%   >> Y = magic(3);
%   >> X = afft.ifftn(Y, [4, 4]);
%   >> disp(X)
%      2.8125 + 0.0000i   0.0000 + 0.9375i   0.9375 + 0.0000i   0.0000 - 0.9375i
%      0.0000 + 0.9375i  -0.3125 - 0.7500i   1.0000 + 0.3125i   0.3125 + 0.2500i
%      0.9375 + 0.0000i   0.5000 + 0.3125i   0.3125 + 0.0000i   0.5000 - 0.3125i
%      0.0000 - 0.9375i   0.3125 - 0.2500i   1.0000 - 0.3125i  -0.3125 + 0.7500i
%
% See also:
%   ifftn, afft.fftn, fftn
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">afft GitHub</a>.

  X = afft_matlab(uint32(3002), Y, varargin{:});
end
