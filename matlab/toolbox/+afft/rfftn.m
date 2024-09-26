function Y = rfftn(X, varargin)
  Y = afft.internal.afft_matlab(uint32(2008), X, varargin{:});
end
