function Y = dttn(X, varargin)
  Y = afft_matlab(uint32(6002), X, varargin{:});
end
