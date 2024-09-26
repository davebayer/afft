function Y = dst(X, varargin)
  Y = internal.afft_matlab(uint32(5000), X, varargin{:});
end
