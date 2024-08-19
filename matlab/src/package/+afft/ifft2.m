function dst = ifft2(src, varargin)
  narginchk(1,4)
  dst = afft_matlab(uint32(3001), src, varargin);
end
