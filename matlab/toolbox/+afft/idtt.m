function X = idtt(Y, varargin)
  X = afft_matlab(uint32(6003), Y, varargin{:});
end
