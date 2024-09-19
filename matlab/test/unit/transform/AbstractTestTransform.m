%% AbstractTestTransform
% *Toolbox:* afft
% 
% Superclass for transform unit tests.
%
%% Description
% This abstract class provides a framework for testing matrix functions in
% different grid sizes, precisions, and complexities. Derived classes
% should specify the testFunction, referenceFunction, and appropriate test
% parameters.
%
%% Examples
%   % Example of a derived class
%   classdef MyTestTransform < AbstractTestTransform
%       properties (TestParameter)
%           gridSize = [AbstractTestTransform.GridSizes1D, AbstractTestTransform.GridSizes2D, AbstractTestTransform.GridSizes3D];
%           precision = {'single', 'double'};
%           isComplex = [true, false];
%       end
%       properties
%           testFunction = @(varargin) myTestFunction(varargin{:});
%           referenceFunction = @(varargin) myReferenceFunction(varargin{:});
%       end
%   end
%
%   % Creating an instance of the derived class and running tests
%   testCase = MyTestTransform();
%   results = run(testCase);
%
%% Properties (Abstract)
% * |testFunction| - (function_handle) Function handle for the test
%   function, which must take one input.
% * |referenceFunction| - (function_handle) Function handle for the
%   reference, which must take one input.
%
%% Properties
% * |GridSizes1D| - (vector) 1D grid sizes for testing.
% * |GridSizes2D| - (cell array) 2D grid sizes for testing.
% * |GridSizes3D| - (cell array) 3D grid sizes for testing.
% * |singleTolerance| - (double) Tolerance for single precision
%   comparisons. 
% * |doubleTolerance| - (double) Tolerance for double precision
%   comparisons. 
%
%% Methods
% * |testTransform| - Runs a test comparing the output of testFunction and
%   referenceFunction using specified grid size, precision, and complexity.
%
%% Methods (Static)
% * |rfftnRef| - Reference function for afft.rfftn.
% * |rfft2Ref| - Reference function for afft.rfft2.
% * |dctnRef| - Reference function for afft.dctn.
% * |dstnRef| - Reference function for afft.dstn.
%
%% See Also
% * |matlab.unittest.TestCase|
% * |matlab.unittest.parameters.TestParameter|

classdef (Abstract) AbstractTestTransform < matlab.unittest.TestCase
  properties (Constant)
    GridSizes0D     = {[0, 0]};
    GridSizes1D     = {[10], [15]};
    GridSizes2D     = {[10, 10], [15, 15], [10, 11], [11, 10], [12, 14], [13, 15]};
    GridSizes3D     = {[10, 10, 10], [15, 15, 15], [10, 11, 11], [11, 12, 11], [11, 11, 12], [10, 11, 12], ...
                       [11, 12, 12], [12, 12, 11], [12, 10, 12]};
    
    singleTolerance = 3e-5;
    doubleTolerance = 3e-12;
  end

  methods (Static)
    function src = generateSrcArray(gridSize, precision, complexity)
      src = rand(gridSize, precision);

      if strcmp(complexity, 'real')
      elseif strcmp(complexity, 'complex')
        src = src + 1i * rand(gridSize, precision);
      else
        error('Invalid complexity');
      end
    end
  end
  methods
    function compareResults(testCase, precision, dstRef, dst)
      if strcmp(precision, 'single')
        tolerance = testCase.singleTolerance;
      elseif strcmp(precision, 'double')
        tolerance = testCase.doubleTolerance;
      else
        error('Invalid precision');
      end

      testCase.verifyEqual(double(gather(dst)), double(gather(dstRef)), 'RelTol', tolerance);
    end
  end
end
