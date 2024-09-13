function Y = irfftn(X, rsz, varargin)
  Y = afft.irfftn(X, rsz, varargin{:});
end
