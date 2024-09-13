function Y = irfft2(X, rm, rn, varargin)
  Y = afft.irfft2(uint32(3004), X, rm, rn, varargin{:});
end
