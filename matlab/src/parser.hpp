/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef PARSER_HPP
#define PARSER_HPP

#include <afft/afft.hpp>
#include <matlabw/mx/mx.hpp>

/// @brief Transform parser.
class TransformParser
{
  public:
    /**
     * @brief Parse DFT parameters.
     * @param array The array to parse.
     * @return DFT parameters.
     */
    [[nodiscard]] afft::Transform operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform type must be a char array");
      }

      std::u16string_view strView{mx::CharArrayCref{array}};

      if (strView == u"dft")
      {
        return afft::Transform::dft;
      }
      else if (strView == u"dht")
      {
        return afft::Transform::dht;
      }
      else if (strView == u"dtt")
      {
        return afft::Transform::dtt;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }
};

/// @brief Direction parser.
class DirectionParser
{
  public:
    /**
     * @brief Parse direction.
     * @param array The array to parse.
     * @return Direction.
     */
    [[nodiscard]] afft::Direction operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "direction must be a char array");
      }

      std::u16string_view strView{matlabw::mx::CharArrayCref{array}};

      if (strView == u"forward")
      {
        return afft::Direction::forward;
      }
      else if (strView == u"inverse" || strView == u"backward")
      {
        return afft::Direction::inverse;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid direction");
      }
    }
};

/// @brief Precision parser.
class PrecisionParser
{
  public:
    /**
     * @brief Parse precision.
     * @param array The array to parse.
     * @return Precision.
     */
    [[nodiscard]] afft::PrecisionTriad operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "precision must be a char array");
      }

      std::u16string_view strView{matlabw::mx::CharArrayCref{array}};

      if (strView == u"single")
      {
        return afft::makePrecision<float>();
      }
      else if (strView == u"double")
      {
        return afft::makePrecision<double>();
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid precision");
      }
    }
};

/// @brief Shape parser.
class ShapeParser
{
  public:
    /**
     * @brief Parse shape.
     * @param array The array to parse.
     * @return Shape.
     */
    [[nodiscard]] afft::View<afft::Size> operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isNumeric() || array.isComplex())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform shape must be a real numeric array");
      }

      const std::size_t shapeRank = array.getSize();

      if (shapeRank > afft::maxDimCount)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform shape exceeds maximum dimension count");
      }

      mx::visit(array, [&](auto&& typedArray)
      {
        using TypedArrayT = std::decay_t<decltype(typedArray)>;

        if constexpr (matlabw::mx::isRealNumeric<typename TypedArrayT::value_type>)
        {
          std::transform(typedArray.begin(), typedArray.end(), mShapeStorage, [](auto value)
          {
            if (value <= typename TypedArrayT::value_type{})
            {
              throw mx::Exception("afft:planCreate:invalidArgument", "transform shape must be a positive number");
            }

            return static_cast<afft::Size>(value);
          });

          std::reverse(mShapeStorage, mShapeStorage + shapeRank);
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "transform shape must be a numeric array");
        }
      });

      return afft::View<afft::Size>{mShapeStorage, shapeRank};
    }

  private:
    afft::Size mShapeStorage[afft::maxDimCount]; ///< Shape of the transform.
};

/// @brief Axes parser.
class AxesParser
{
  public:
    /**
     * @brief Parse axes.
     * @param array The array to parse.
     * @return Axes.
     */
    [[nodiscard]] afft::View<afft::Axis> operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isNumeric() || array.isComplex())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform axes must be a real numeric array");
      }

      const std::size_t transformRank = array.getSize();

      if (transformRank > afft::maxDimCount)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform axes exceeds maximum dimension count");
      }

      mx::visit(array, [&](auto&& typedArray)
      {
        using TypedArrayT = std::decay_t<decltype(typedArray)>;

        if constexpr (matlabw::mx::isRealNumeric<typename TypedArrayT::value_type>)
        {
          std::transform(typedArray.begin(), typedArray.end(), mAxesStorage, [transformRank](auto value)
          {
            if (value <= typename TypedArrayT::value_type{})
            {
              throw mx::Exception("afft:planCreate:invalidArgument", "transform axes must be a positive number");
            }

            return static_cast<afft::Axis>(transformRank) - static_cast<afft::Axis>(value) - 1;
          });

          std::reverse(mAxesStorage, mAxesStorage + transformRank);
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "transform axes must be a numeric array");
        }
      });

      return afft::View<afft::Axis>{mAxesStorage, transformRank};
    }

  private:
    afft::Axis mAxesStorage[afft::maxDimCount]; ///< Axes of the transform.
};

/// @brief Normalization parser.
class NormalizationParser
{
  public:
    /**
     * @brief Parse normalization.
     * @param array The array to parse.
     * @return Normalization.
     */
    [[nodiscard]] afft::Normalization operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "normalization must be a char array");
      }

      std::u16string_view strView{matlabw::mx::CharArrayCref{array}};

      if (strView == u"none")
      {
        return afft::Normalization::none;
      }
      else if (strView == u"unitary")
      {
        return afft::Normalization::unitary;
      }
      else if (strView == u"ortho" || strView == u"orthogonal")
      {
        return afft::Normalization::orthogonal;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid normalization");
      }
    }
};

/// @brief DFT type parser.
class DftTypeParser
{
  public:
    /**
     * @brief Parse DFT type.
     * @param array The array to parse.
     * @return DFT type.
     */
    [[nodiscard]] afft::dft::Type operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "DFT type must be a char array");
      }

      std::u16string_view strView{matlabw::mx::CharArrayCref{array}};

      if (strView == u"complexToComplex" || strView == u"c2c")
      {
        return afft::dft::Type::complexToComplex;
      }
      else if (strView == u"realToComplex" || strView == u"r2c")
      {
        return afft::dft::Type::realToComplex;
      }
      else if (strView == u"complexToReal" || strView == u"c2r")
      {
        return afft::dft::Type::complexToReal;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }
};

/// @brief DHT type parser.
class DhtTypeParser
{
  public:
    /**
     * @brief Parse DHT type.
     * @param array The array to parse.
     * @return DHT type.
     */
    [[nodiscard]] afft::dht::Type operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "DHT type must be a char array");
      }

      std::u16string_view strView{matlabw::mx::CharArrayCref{array}};

      if (strView == u"separable")
      {
        return afft::dht::Type::separable;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }
};

/// @brief DTT types parser.
class DttTypesParser
{
  public:
    /**
     * @brief Parse DTT types.
     * @param array The array to parse.
     * @return DTT types.
     */
    [[nodiscard]] afft::View<afft::dtt::Type> operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "DTT types must be a char array");
      }

      const std::size_t rank = array.getDimM();

      if (rank > afft::maxDimCount)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "DTT types exceeds maximum dimension count");
      }

      // TODO: add support for specifying DTT type for each dimension, do not forget to change the order of the types
      if (rank != 1)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "DTT types must be a row vector");
      }

      const std::u16string_view strView{mx::CharArrayCref{array}};

      mDttTypesStorage[0] = parseSingleDttType(strView);

      return afft::View<afft::dtt::Type>{mDttTypesStorage, rank};
    }

  private:
    /**
     * @brief Parse single DTT type.
     * @param strView The string view to parse.
     * @return DTT type.
     */
    [[nodiscard]] afft::dtt::Type parseSingleDttType(std::u16string_view strView)
    {
      if (strView.size() != 3 && strView.size() != 4)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }

      bool isDct{};

      if (strView.substr(0, 3) == u"dct")
      {
        isDct = true;
      }
      else if (strView.substr(0, 3) == u"dst")
      {
        isDct = false;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }

      if (strView.size() == 3)
      {
        return (isDct) ? afft::dtt::Type::dct : afft::dtt::Type::dst;
      }
      
      if (auto numStrView = strView.substr(3, 1); numStrView == u"1")
      {
        return (isDct) ? afft::dtt::Type::dct1 : afft::dtt::Type::dst1;
      }
      else if (numStrView == u"2")
      {
        return (isDct) ? afft::dtt::Type::dct2 : afft::dtt::Type::dst2;
      }
      else if (numStrView == u"3")
      {
        return (isDct) ? afft::dtt::Type::dct3 : afft::dtt::Type::dst3;
      }
      else if (numStrView == u"4")
      {
        return (isDct) ? afft::dtt::Type::dct4 : afft::dtt::Type::dst4;
      }
      else if (numStrView == u"5")
      {
        return (isDct) ? afft::dtt::Type::dct5 : afft::dtt::Type::dst5;
      }
      else if (numStrView == u"6")
      {
        return (isDct) ? afft::dtt::Type::dct6 : afft::dtt::Type::dst6;
      }
      else if (numStrView == u"7")
      {
        return (isDct) ? afft::dtt::Type::dct7 : afft::dtt::Type::dst7;
      }
      else if (numStrView == u"8")
      {
        return (isDct) ? afft::dtt::Type::dct8 : afft::dtt::Type::dst8;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }

    afft::dtt::Type mDttTypesStorage[afft::maxDimCount]; ///< DTT types of the transform.
};

/// @brief Transform parameters parser.
class TransformParametersParser
{
  public:
    /**
     * @brief Parse transform parameters.
     * @param transformParamsArray Transform parameters array.
     * @return Transform parameters.
     */
    [[nodiscard]] afft::TransformParametersVariant operator()(matlabw::mx::ArrayCref transformParamsArray)
    {
      if (!transformParamsArray.isStruct())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform parameters must be a struct array");
      }

      matlabw::mx::StructArrayCref transformParamsStruct{transformParamsArray};

      const auto transformArray = transformParamsStruct.getField("transform");

      if (!transformArray)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "missing transform field");
      }

      TransformParser transfromParser{};

      switch (transfromParser(*transformArray))
      {
      case afft::Transform::dft:
        return parseDftTransformParameters(transformParamsStruct);
      case afft::Transform::dht:
        return parseDhtTransformParameters(transformParamsStruct);
      case afft::Transform::dtt:
        return parseDttTransformParameters(transformParamsStruct);
      default:
        throw mx::Exception("afft:planCreate:internal", "invalid transform type");
      }
    }

  private:
    /**
     * @brief Parse direction.
     * @param directionArray The direction array.
     * @return Direction.
     */
    [[nodiscard]] afft::Direction parseDirection(std::optional<matlabw::mx::ArrayCref> directionArray)
    {
      if (!directionArray)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "missing direction field");
      }

      return mDirectionParser(*directionArray);
    }

    /**
     * @brief Parse precision.
     * @param precisionArray The precision array.
     * @return Precision.
     */
    [[nodiscard]] afft::PrecisionTriad parsePrecision(std::optional<matlabw::mx::ArrayCref> precisionArray)
    {
      if (!precisionArray)
      {
        return afft::makePrecision<double>();
      }

      return mPrecisionParser(*precisionArray);
    }

    /**
     * @brief Parse shape.
     * @param shapeArray The shape array.
     * @return Shape.
     */
    [[nodiscard]] afft::View<afft::Size> parseShape(std::optional<matlabw::mx::ArrayCref> shapeArray)
    {
      if (!shapeArray)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "missing shape field");
      }

      return mShapeParser(*shapeArray);
    }

    /**
     * @brief Parse axes.
     * @param axesArray The axes array.
     * @return Axes.
     */
    [[nodiscard]] afft::View<afft::Axis> parseAxes(std::optional<matlabw::mx::ArrayCref> axesArray)
    {
      if (!axesArray)
      {
        return afft::allAxes;
      }

      return mAxesParser(*axesArray);
    }

    /**
     * @brief Parse normalization.
     * @param normalizationArray The normalization array.
     * @return Normalization.
     */
    [[nodiscard]] afft::Normalization parseNormalization(std::optional<matlabw::mx::ArrayCref> normalizationArray)
    {
      if (!normalizationArray)
      {
        return afft::Normalization::none;
      }

      return mNormalizationParser(*normalizationArray);
    }

    /**
     * @brief Parse DFT transform parameters.
     * @param transformParamsStruct The transform parameters struct.
     * @return DFT transform parameters.
     */
    [[nodiscard]] afft::dft::Parameters parseDftTransformParameters(matlabw::mx::StructArrayCref transformParamsStruct)
    {
      afft::dft::Parameters dftParams{};
      dftParams.direction     = parseDirection(transformParamsStruct.getField("direction"));
      dftParams.precision     = parsePrecision(transformParamsStruct.getField("precision"));
      dftParams.shape         = parseShape(transformParamsStruct.getField("shape"));
      dftParams.axes          = parseAxes(transformParamsStruct.getField("axes"));
      dftParams.normalization = parseNormalization(transformParamsStruct.getField("normalization"));

      if (const auto dftTypeArray = transformParamsStruct.getField("type"))
      {
        dftParams.type = mDftTypeParser(*dftTypeArray);
      }
      else
      {
        dftParams.type = afft::dft::Type::complexToComplex;
      }

      return dftParams;
    }

    /**
     * @brief Parse DHT transform parameters.
     * @param transformParamsStruct The transform parameters struct.
     * @return DHT transform parameters.
     */
    [[nodiscard]] afft::dht::Parameters parseDhtTransformParameters(matlabw::mx::StructArrayCref transformParamsStruct)
    {
      afft::dht::Parameters dhtParams{};
      dhtParams.direction     = parseDirection(transformParamsStruct.getField("direction"));
      dhtParams.precision     = parsePrecision(transformParamsStruct.getField("precision"));
      dhtParams.shape         = parseShape(transformParamsStruct.getField("shape"));
      dhtParams.axes          = parseAxes(transformParamsStruct.getField("axes"));
      dhtParams.normalization = parseNormalization(transformParamsStruct.getField("normalization"));

      if (const auto dhtTypeArray = transformParamsStruct.getField("type"))
      {
        dhtParams.type = mDhtTypeParser(*dhtTypeArray);
      }
      else
      {
        dhtParams.type = afft::dht::Type::separable;
      }

      return dhtParams;
    }

    /**
     * @brief Parse DTT transform parameters.
     * @param transformParamsStruct The transform parameters struct.
     * @return DTT transform parameters.
     */
    [[nodiscard]] afft::dtt::Parameters parseDttTransformParameters(matlabw::mx::StructArrayCref transformParamsStruct)
    {
      afft::dtt::Parameters dttParams{};
      dttParams.direction     = parseDirection(transformParamsStruct.getField("direction"));
      dttParams.precision     = parsePrecision(transformParamsStruct.getField("precision"));
      dttParams.shape         = parseShape(transformParamsStruct.getField("shape"));
      dttParams.axes          = parseAxes(transformParamsStruct.getField("axes"));
      dttParams.normalization = parseNormalization(transformParamsStruct.getField("normalization"));

      if (const auto dttTypesArray = transformParamsStruct.getField("type"))
      {
        dttParams.types = mDttTypesParser(*dttTypesArray);
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "missing DTT types field");
      }

      return dttParams;
    }

    DirectionParser     mDirectionParser;     ///< Direction parser.
    PrecisionParser     mPrecisionParser;     ///< Precision parser.
    ShapeParser         mShapeParser;         ///< Shape parser.
    AxesParser          mAxesParser;          ///< Axes parser.
    NormalizationParser mNormalizationParser; ///< Normalization parser.
    union
    {
      DftTypeParser     mDftTypeParser;       ///< DFT type parser.
      DhtTypeParser     mDhtTypeParser;       ///< DHT type parser.
      DttTypesParser    mDttTypesParser;      ///< DTT types parser;
    };
};

/// @brief Target parser.
class TargetParser
{
  public:
    /**
     * @brief Parse target.
     * @param array The array to parse.
     * @return Target.
     */
    [[nodiscard]] afft::Target operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "target must be a char array");
      }

      std::u16string_view strView{mx::CharArrayCref{array}};

      if (strView == u"cpu")
      {
        return afft::Target::cpu;
      }
      else if (strView == u"gpu" || strView == u"cuda")
      {
        return afft::Target::cuda;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid target");
      }
    }
};

/// @brief Target parameters parser.
class TargetParametersParser
{
  public:
    /**
     * @brief Parse target parameters.
     * @param targetParamsArray Target parameters array.
     * @return Target parameters.
     */
    [[nodiscard]] afft::TargetParametersVariant operator()(matlabw::mx::ArrayCref targetParamsArray)
    {
      if (!targetParamsArray.isStruct())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "target parameters must be a struct array");
      }

      matlabw::mx::StructArrayCref targetParamsStruct{targetParamsArray};

      const auto targetArray = targetParamsStruct.getField("target");

      if (!targetArray)
      {
        return afft::cpu::Parameters{};
      }

      TargetParser targetParser{};
      
      switch (targetParser(*targetArray))
      {
      case afft::Target::cpu:
        return parseCpuTargetParameters(targetParamsStruct);
      case afft::Target::cuda:
#     ifdef MATLABW_ENABLE_GPU
        return parseCudaTargetParameters(targetParamsStruct);
#     else
        throw mx::Exception("afft:planCreate:invalidArgument", "GPU target is disabled");
#     endif
      default:
        throw mx::Exception("afft:planCreate:internal", "invalid target");
      }
    }

  private:
    /**
     * @brief Parse CPU target parameters.
     * @param targetParamsStruct The target parameters struct.
     * @return CPU target parameters.
     */
    [[nodiscard]] afft::cpu::Parameters parseCpuTargetParameters(matlabw::mx::StructArrayCref)
    {
      return afft::cpu::Parameters{};
    }

# ifdef MATLABW_ENABLE_GPU
    /**
     * @brief Parse CUDA target parameters.
     * @param targetParamsStruct The target parameters struct.
     * @return CUDA target parameters.
     */
    [[nodiscard]] afft::cuda::Parameters parseCudaTargetParameters(matlabw::mx::StructArrayCref)
    {
      afft::cuda::Parameters cudaParams{};

      if (cudaGetDevice(&cudaDevice) != cudaSuccess)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "failed to get current CUDA device");
      }

      cudaParams.devices = afft::makeScalarView(cudaDevice);

      return cudaParams;
    }
# endif
  
    union
    {
#   ifdef MATLABW_ENABLE_GPU
      int cudaDevice{}; ///< CUDA device.
#   endif
    };
};

/// @brief Cpu thread limit parser.
class CpuThreadLimitParser
{
  public:
    /**
     * @brief Parse CPU thread limit.
     * @param array The array to parse.
     * @return CPU thread limit.
     */
    [[nodiscard]] std::uint32_t operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isScalar() || !array.isNumeric())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "threads must be a scalar uint64");
      }

      return mx::visit(array, [](auto&& array) -> std::uint32_t
      {
        using TypedArrayT = std::decay_t<decltype(array)>;

        if constexpr (mx::isRealNumeric<typename TypedArrayT::value_type>)
        {
          const auto value = array[0];

          if (value < typename TypedArrayT::value_type{})
          {
            throw mx::Exception("afft:planCreate:invalidArgument", "thread limit must be a non-negative number");
          }

          return static_cast<std::uint32_t>(value);
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "thread limit must be a non-negative number");
        }
      });
    }
};

/// @brief Backend parameters parser.
class BackendParametersParser
{
  public:
    /**
     * @brief Parse backend parameters.
     * @param backendParamsStruct Backend parameters struct.
     * @param target Target.
     * @return Backend parameters.
     */
    [[nodiscard]] afft::BackendParametersVariant operator()(matlabw::mx::ArrayCref array, afft::Target target)
    {
      if (!array.isStruct())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "backend parameters must be a struct array");
      }

      matlabw::mx::StructArrayCref backendParamsStruct{array};

      switch (target)
      {
      case afft::Target::cpu:
        return parseCpuBackendParams(backendParamsStruct);
      case afft::Target::cuda:
        return parseCudaBackendParams(backendParamsStruct);
      default:
        throw mx::Exception{"afft:planCreate:internal", "invalid backend parameters target"};
      }
    }
  private:
    /**
     * @brief Parse cpu thread limit.
     * @param threadLimitArray Thread limit array.
     * @return Cpu thread limit.
     */
    [[nodiscard]] std::uint32_t parseCpuThreadLimit(std::optional<matlabw::mx::ArrayCref> threadLimitArray)
    {
      if (!threadLimitArray)
      {
        return 0;
      }

      return CpuThreadLimitParser{}(*threadLimitArray);
    }

    /**
     * @brief Parse cpu backend parameters.
     * @param backendParamsStruct Backend parameters struct.
     * @return Cpu backend parameters.
     */
    [[nodiscard]] afft::cpu::BackendParameters parseCpuBackendParams(matlabw::mx::StructArrayCref backendParamsStruct)
    {
      afft::cpu::BackendParameters cpuBackendParams{};
      cpuBackendParams.threadLimit = parseCpuThreadLimit(backendParamsStruct.getField("threadLimit"));

      // TODO: Implement CPU backend parameters.

      return cpuBackendParams;
    }

    /**
     * @brief Parse cuda backend parameters.
     * @param backendParamsStruct Backend parameters struct.
     * @return Cuda backend parameters.
     */
    [[nodiscard]] afft::cuda::BackendParameters parseCudaBackendParams(mx::StructArrayCref backendParamsStruct)
    {
      // TODO: Implement CUDA backend parameters.
      return afft::cuda::BackendParameters{};
    }
};

/// @brief Select strategy parser.
class SelectStrategyParser
{
  public:
    /**
     * @brief Parse select strategy.
     * @param array The array to parse.
     * @return Select strategy.
     */
    [[nodiscard]] afft::SelectStrategy operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "select strategy must be a char array");
      }

      std::u16string_view strView{mx::CharArrayCref{array}};

      if (strView == u"first")
      {
        return afft::SelectStrategy::first;
      }
      else if (strView == u"best")
      {
        return afft::SelectStrategy::best;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid select strategy");
      }
    }
};

/// @brief Select parameters parser.
class SelectParametersParser
{
  public:
    /**
     * @brief Parse select parameters.
     * @param selectParamsStruct Select parameters struct.
     * @return Select parameters.
     */
    [[nodiscard]] afft::SelectParametersVariant operator()(matlabw::mx::ArrayCref array)
    {
      if (!array.isStruct())
      {
        throw mx::Exception{"afft:planCreate:invalidArgument", "select parameters must be a struct array"};
      }

      matlabw::mx::StructArrayCref selectParamsStruct{array};

      auto strategyArray = selectParamsStruct.getField(0, "strategy");

      if (!strategyArray)
      {
        return afft::FirstSelectParameters{};
      }

      switch (mSelectStrategyParser(*strategyArray))
      {
      case afft::SelectStrategy::first:
        return afft::FirstSelectParameters{};
      case afft::SelectStrategy::best:
        return afft::BestSelectParameters{};
      default:
        throw mx::Exception{"afft:planCreate:internal", "invalid select strategy"};
      }
    }
  private:
    SelectStrategyParser mSelectStrategyParser; ///< Select strategy parser.
};

#endif /* PARSER_HPP */
