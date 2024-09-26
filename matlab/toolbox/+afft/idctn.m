function X = idctn(Y, varargin)
  X = afft.internal.afft_matlab(uint32(4005), Y, varargin{:});
end
