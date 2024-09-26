function X = idstn(Y, varargin)
  X = internal.afft_matlab(uint32(5005), Y, varargin{:});
end
