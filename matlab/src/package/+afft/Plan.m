classdef Plan
  properties
    ptr (1, 1) uint64 {} = 0
  end
  methods
    function obj = Plan(transformParams, targetParams)
      arguments
        transformParams (1, 1) struct
        targetParams    (1, 1) struct = struct()
      end
      obj.ptr = afft_matlab(uint32(1000), transformParams, targetParams);
    end
    function dst = execute(obj, src)
      dst = afft_matlab(uint32(1001), obj.ptr, src);
    end
  end
end
