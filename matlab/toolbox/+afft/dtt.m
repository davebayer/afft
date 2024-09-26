function Y = dtt(X, varargin)
  Y = internal.afft_matlab(uint32(6000), X, varargin{:});
end
