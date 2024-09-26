function Y = dct(X, varargin)
  Y = afft.internal.afft_matlab(uint32(4000), X, varargin{:});
end
