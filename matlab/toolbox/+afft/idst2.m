function X = idst2(Y, varargin)
  X = internal.afft_matlab(uint32(5004), Y, varargin{:});
end
