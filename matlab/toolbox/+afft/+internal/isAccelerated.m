function res = isAccelerated()
  persistent cachedRes;
  if isempty(cachedRes)
    cachedRes = isfile(strcat('afft_matlab.', mexext));
  end

  res = cachedRes;
end
