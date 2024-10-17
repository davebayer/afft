function res = isAccelerated()
  persistent cachedRes;
  if isempty(cachedRes)
    cachedRes = exist(strcat('+afft/+internal/afft_matlab.', mexext), 'file') == 3;
  end

  res = cachedRes;
end
