function X = idht2(Y, varargin)
  X = afft.internal.afft_matlab(uint32(3004), Y, varargin{:});
end
