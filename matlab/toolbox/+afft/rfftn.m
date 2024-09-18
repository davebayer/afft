function Y = rfftn(X, varargin)
  Y = afft_matlab(uint32(2008), X, varargin{:});
end
