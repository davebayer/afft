function Y = rfft(X, varargin)
  Y = afft.rfft(uint32(2006), X, varargin{:});
end
