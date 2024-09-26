function Y = dstn(X, varargin)
  Y = internal.afft_matlab(uint32(5002), X, varargin{:});
end
