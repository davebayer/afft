function res = isSelectStrategy(str)
  res = ischar(str) && ismember(str, {'first', 'best'});
end
