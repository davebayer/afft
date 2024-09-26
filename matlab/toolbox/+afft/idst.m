function X = idst(Y, varargin)
  X = internal.afft_matlab(uint32(5003), Y, varargin{:});
end
