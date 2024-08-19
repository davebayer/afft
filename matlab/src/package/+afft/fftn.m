function dst = fftn(src, size)
  arguments
    src
    size = []
  end

  dst = afft(uint32(2002), src, size);
end
