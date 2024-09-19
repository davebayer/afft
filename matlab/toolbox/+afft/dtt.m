function Y = dtt(X, varargin)
  Y = afft_matlab(uint32(6000), X, varargin{:});
end
