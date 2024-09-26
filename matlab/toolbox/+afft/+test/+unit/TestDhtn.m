classdef TestDhtn < AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    dhtType       = {'separable'};
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    normalization = {'none'}; % todo: add 'unitary', 'orthogonal' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, dhtType, normalization)
      if strcmp(dhtType, 'separable')
        tmp = src;
        
        % Compute the reference using the built-in fft function.
        for dim = 1:ndims(src)
          tmp = fft(tmp, [], dim);
          tmp = real(tmp) - imag(tmp);
        end

        dstRef = tmp;
      else
        error('Invalid DHT type');
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
    function testCpu(testCase, precision, dhtType, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real');

      dstRef = TestDhtn.computeReference(src, dhtType, normalization);
      dst    = afft.dhtn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end

    % None of the gpu backends support the DHT transform yet.
    % function testGpu(testCase, precision, dhtType, gridSize, normalization)
    %   src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real');

    %   dstRef = TestDhtn.computeReference(src, dhtType, normalization);
    %   dst    = afft.dhtn(src); % todo: implement normalization

    %   compareResults(testCase, precision, dstRef, dst);
    % end
  end
end