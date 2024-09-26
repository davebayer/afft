function X = idhtn(Y, varargin)
  X = afft.internal.afft_matlab(uint32(3005), Y, varargin{:});
end
