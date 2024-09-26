function Y = rfft(X, varargin)
  Y = internal.afft_matlab(uint32(2006), X, varargin{:});
end
