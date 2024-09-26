function Y = rfft2(X, varargin)
  Y = internal.afft_matlab(uint32(2007), X, varargin{:});
end
