function Y = dhtn(X, varargin)
  Y = internal.afft_matlab(uint32(3002), X, varargin{:});
end
