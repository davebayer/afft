function X = idct(Y, varargin)
  X = internal.afft_matlab(uint32(4003), Y, varargin{:});
end
