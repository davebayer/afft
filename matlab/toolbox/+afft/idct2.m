function X = idct2(Y, varargin)
  X = afft_matlab(uint32(4004), Y, varargin{:});
end
