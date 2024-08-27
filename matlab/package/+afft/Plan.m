classdef Plan
  % afft.Plan Standalone class for transform plans
  % This class wraps the C++ afft::Plan object and enables to execute the transforms with minimum overhead.
  %
  % afft.Plan Methods:
  %   Plan                   - Constructor for the Plan class
  %   execute                - Execute the transform
  %   getTransformParameters - Get the transform parameters
  %   getTargetParameters    - Get the target parameters

  properties (Access = private)
    data % oblique data property
  end
  
  methods
    function obj = Plan(transformParams, targetParams)
      arguments
        transformParams (1, 1) struct
        targetParams    (1, 1) struct = struct('target', 'cpu')
      end
      obj.data = afft_matlab(uint32(1000), transformParams, targetParams);
    end
    function Y = execute(obj, X)
      % execute Execute the transform plan
      %   Executes the plan on the input data X. The X must match the transform description provided in the constructor.
      Y = afft_matlab(uint32(1001), obj.data, X);
    end
    function transformParams = getTransformParameters(obj)
      transformParams = afft_matlab(uint32(1002), obj.data);
    end
    function targetParams = getTargetParameters(obj)
      targetParams = afft_matlab(uint32(1003), obj.data);
    end
  end
end
