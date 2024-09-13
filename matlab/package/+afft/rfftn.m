function Y = rfftn(X, varargin)
  Y = afft.rfftn(uint32(3002), X, varargin{:});
end
