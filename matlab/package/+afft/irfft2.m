function X = irfft2(Y, rm, rn, varargin)
  X = afft.irfft2(uint32(2010), Y, rm, rn, varargin{:});
end
