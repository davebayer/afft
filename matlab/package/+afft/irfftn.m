function Y = irfftn(X, rsz, varargin)
  Y = afft.irfftn(uint32(3005), X, rsz, varargin{:});
end
