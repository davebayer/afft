function Y = rfft(X, varargin)
  Y = afft.rfft(uint32(3000), X, varargin{:});
end
