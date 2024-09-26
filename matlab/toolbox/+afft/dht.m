function Y = dht(X, varargin)
  Y = afft.internal.afft_matlab(uint32(3000), X, varargin{:});
end
