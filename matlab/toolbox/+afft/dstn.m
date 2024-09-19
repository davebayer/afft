function Y = dstn(X, varargin)
  Y = afft_matlab(uint32(5002), X, varargin{:});
end
