function Y = dst2(X, varargin)
  Y = afft.internal.afft_matlab(uint32(5001), X, varargin{:});
end
