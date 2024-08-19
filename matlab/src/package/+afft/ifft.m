function dst = ifft(src, varargin)
  narginchk(1,4)
  dst = afft_matlab(uint32(3000), src, varargin);
end
