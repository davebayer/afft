function Y = rfftn(X, varargin)
  Y = afft.rfftn(uint32(2008), X, varargin{:});
end
