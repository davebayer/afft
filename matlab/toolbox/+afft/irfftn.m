function X = irfftn(Y, rsz, varargin)
  X = internal.afft_matlab(uint32(2011), Y, rsz, varargin{:});
end
