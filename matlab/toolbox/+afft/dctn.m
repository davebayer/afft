function Y = dctn(X, varargin)
  Y = internal.afft_matlab(uint32(4002), X, varargin{:});
end
