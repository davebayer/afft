function Y = rfft2(X, varargin)
  Y = afft.rfft2(uint32(2007), X, varargin{:});
end
