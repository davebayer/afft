function res = isThreadLimit(limit)
  res = isnumeric(limit) && isreal(limit) && limit >= 0;
end
