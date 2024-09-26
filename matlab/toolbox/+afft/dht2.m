function Y = dht2(X, varargin)
  Y = internal.afft_matlab(uint32(3001), X, varargin{:});
end
