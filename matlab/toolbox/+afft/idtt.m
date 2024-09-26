function X = idtt(Y, varargin)
  X = afft.internal.afft_matlab(uint32(6003), Y, varargin{:});
end
