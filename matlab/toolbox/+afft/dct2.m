function Y = dct2(X, varargin)
  Y = afft.internal.afft_matlab(uint32(4001), X, varargin{:});
end
