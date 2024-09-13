function X = irfftn(Y, rsz, varargin)
  X = afft.irfftn(uint32(2011), Y, rsz, varargin{:});
end
