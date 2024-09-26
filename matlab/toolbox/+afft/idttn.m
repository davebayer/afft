function X = idttn(Y, varargin)
  X = afft.internal.afft_matlab(uint32(6005), Y, varargin{:});
end
