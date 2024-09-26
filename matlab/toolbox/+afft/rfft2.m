function Y = rfft2(X, varargin)
  Y = afft.internal.afft_matlab(uint32(2007), X, varargin{:});
end
