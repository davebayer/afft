function dst = ifftn(src, varargin)
  narginchk(1,3)
  dst = afft(uint32(3002), src, varargin);
end
