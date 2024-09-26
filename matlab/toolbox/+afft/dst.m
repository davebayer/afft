function Y = dst(X, varargin)
  Y = afft.internal.afft_matlab(uint32(5000), X, varargin{:});
end
