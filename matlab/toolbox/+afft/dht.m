function Y = dht(X, varargin)
  Y = afft_matlab(uint32(3000), X, varargin{:});
end
