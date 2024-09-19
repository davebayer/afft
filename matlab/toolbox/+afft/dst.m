function Y = dst(X, varargin)
  Y = afft_matlab(uint32(5000), X, varargin{:});
end
