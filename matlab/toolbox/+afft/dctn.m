function Y = dctn(X, varargin)
  Y = afft.internal.afft_matlab(uint32(4002), X, varargin{:});
end
