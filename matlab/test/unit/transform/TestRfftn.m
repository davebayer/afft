classdef TestRfftn < AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    normalization = {'none'}; % todo: add 'unitary', 'orthogonal' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fftn function.
      dstRef = fftn(src);

      if not(isempty(dstRef))
        % Reduce the x dimension to (x / 2 + 1)
        xReduced = uint64(fix(size(dstRef, 1) / 2)) + 1;
        dstRef   = dstRef(1:xReduced, :, :);
      end

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / numel(src);
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef / sqrt(numel(src));
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, precision, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real', 'cpu');

      dstRef = TestRfftn.computeReference(src, normalization);
      dst    = afft.rfftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real', 'gpu');

      dstRef = TestRfftn.computeReference(src, normalization);
      dst    = afft.rfftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end