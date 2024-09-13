function Y = rfft2(X, varargin)
  Y = afft.rfft2(uint32(3001), X, varargin{:});
end
