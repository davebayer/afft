classdef Plan
  properties
    ptr (1, 1) uint64 {} = 0
  end
  methods
    function obj = Plan(transformParams, targetParams)
      arguments
        transformParams (1, 1) struct
        targetParams    (1, 1) struct = struct('target', 'cpu')
      end
      obj.ptr = afft_matlab(uint32(1000), transformParams, targetParams);
    end
    function dst = execute(obj, src)
      dst = afft_matlab(uint32(1001), obj.ptr, src);
    end
    function transformParams = getTransformParameters(obj)
      transformParams = afft_matlab(uint32(1002), obj.ptr);
    end
    function targetParams = getTargetParameters(obj)
      targetParams = afft_matlab(uint32(1003), obj.ptr);
    end
  end
end
