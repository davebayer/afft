classdef Plan < matlab.mixin.Copyable
  % afft.Plan Standalone class for transform plans
  % This class wraps the C++ afft::Plan object and enables to execute the transforms with minimum overhead.
  %
  % afft.Plan Methods:
  %   Plan                   - Constructor for the Plan class
  %   execute                - Execute the transform
  %   getTransformParameters - Get the transform parameters
  %   getTargetParameters    - Get the target parameters

  properties (Access = private)
    mData            % oblique data property
    mTransformParams % transform parameters
    mTargetParams    % target parameters
    mBackendParams   % backend parameters
    mSelectParams    % select parameters
  end
  
  methods
    function obj = Plan(transformParams, targetParams, backendParams, selectParams)
      arguments
        transformParams (1, 1) struct
        targetParams    (1, 1) struct = struct()
        backendParams   (1, 1) struct = struct()
        selectParams    (1, 1) struct = struct()
      end

      % Create the plan
      obj.mData = internal.afft_matlab(uint32(1000), transformParams, targetParams, backendParams, selectParams);

      % Store the parameters
      obj.mTransformParams = transformParams;
      obj.mTargetParams    = targetParams;
      obj.mBackendParams   = backendParams;
      obj.mSelectParams    = selectParams;
    end

    function Y = execute(obj, X)
      % execute Execute the transform plan
      %   Executes the plan on the input data X. The X must match the transform description provided in the constructor.
      Y = internal.afft_matlab(uint32(1001), obj.mData, X);
    end

    function transformParams = getTransformParameters(obj)
      transformParams = obj.mTransformParams;
    end

    function targetParams = getTargetParameters(obj)
      targetParams = obj.mTargetParams;
    end

    function backendParams = getBackendParameters(obj)
      backendParams = obj.mBackendParams;
    end

    function selectParams = getSelectParameters(obj)
      selectParams = obj.mSelectParams;
    end
  end
end
