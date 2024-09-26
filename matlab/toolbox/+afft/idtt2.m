function X = idtt2(Y, varargin)
  X = internal.afft_matlab(uint32(6004), Y, varargin{:});
end
