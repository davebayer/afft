function X = idst(Y, varargin)
  X = afft.internal.afft_matlab(uint32(5003), Y, varargin{:});
end
