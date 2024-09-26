function Y = dtt2(X, varargin)
  Y = internal.afft_matlab(uint32(6001), X, varargin{:});
end
