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

#include <afft/afft.hpp>
#include <matlabw/mx/mx.hpp>

#include "planCache.hpp"

using namespace matlabw;

/// @brief Matlab plan data.
struct PlanData
{
  afft::Plan*   plan;       ///< Plan pointer.
  std::uint64_t cacheEpoch; ///< Plan cache epoch.
};

/// @brief Transform parameters parser.
class TransformParametersParser
{
  public:
    /**
     * @brief Parse transform parameters.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform parameters.
     */
    [[nodiscard]] afft::TransformParametersVariant operator()(mx::StructArrayCref transformParamsStruct)
    {
      const afft::Transform transform = parseTransform(transformParamsStruct);

      switch (transform)
      {
      case afft::Transform::dft:
        return parseDftParameters(transformParamsStruct);
      case afft::Transform::dht:
        return parseDhtParameters(transformParamsStruct);
      case afft::Transform::dtt:
        return parseDttParameters(transformParamsStruct);
      default:
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform");
      }
    }
  private:
    /**
     * @brief Parse transform type.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform type.
     */
    [[nodiscard]] afft::Transform parseTransform(mx::StructArrayCref transformParamsStruct)
    {
      auto transformArray = transformParamsStruct.getField(0, "transform");

      if (!transformArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform type");
      }

      if (!transformArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform type must be a char array");
      }

      std::u16string_view transformStrView{mx::CharArrayCref{*transformArray}};

      if (transformStrView == u"dft")
      {
        return afft::Transform::dft;
      }
      else if (transformStrView == u"dht")
      {
        return afft::Transform::dht;
      }
      else if (transformStrView == u"dtt")
      {
        return afft::Transform::dtt;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }

    /**
     * @brief Parse DFT parameters.
     * @param transformParamsStruct Transform parameters struct.
     * @return DFT parameters.
     */
    [[nodiscard]] afft::dft::Parameters parseDftParameters(mx::StructArrayCref transformParamsStruct)
    {
      afft::dft::Parameters dftParams{};
      dftParams.direction     = parseDirection(transformParamsStruct);
      dftParams.precision     = parsePrecision(transformParamsStruct);
      dftParams.shape         = parseShape(transformParamsStruct);
      dftParams.axes          = parseAxes(transformParamsStruct);
      dftParams.normalization = parseNormalization(transformParamsStruct);
      dftParams.placement     = afft::Placement::outOfPlace;
      dftParams.type          = parseDftType(transformParamsStruct);

      return dftParams;
    }

    /**
     * @brief Parse DHT parameters.
     * @param transformParamsStruct Transform parameters struct.
     * @return DHT parameters.
     */
    [[nodiscard]] afft::dht::Parameters parseDhtParameters(mx::StructArrayCref transformParamsStruct)
    {
      afft::dht::Parameters dhtParams{};
      dhtParams.direction     = parseDirection(transformParamsStruct);
      dhtParams.precision     = parsePrecision(transformParamsStruct);
      dhtParams.shape         = parseShape(transformParamsStruct);
      dhtParams.axes          = parseAxes(transformParamsStruct);
      dhtParams.normalization = parseNormalization(transformParamsStruct);
      dhtParams.placement     = afft::Placement::outOfPlace;
      dhtParams.type          = parseDhtType(transformParamsStruct);

      return dhtParams;
    }

    /**
     * @brief Parse DTT parameters.
     * @param transformParamsStruct Transform parameters struct.
     * @return DTT parameters.
     */
    [[nodiscard]] afft::dtt::Parameters parseDttParameters(mx::StructArrayCref transformParamsStruct)
    {
      afft::dtt::Parameters dttParams{};
      dttParams.direction     = parseDirection(transformParamsStruct);
      dttParams.precision     = parsePrecision(transformParamsStruct);
      dttParams.shape         = parseShape(transformParamsStruct);
      dttParams.axes          = parseAxes(transformParamsStruct);
      dttParams.normalization = parseNormalization(transformParamsStruct);
      dttParams.placement     = afft::Placement::outOfPlace;
      dttParams.types         = parseDttTypes(transformParamsStruct);

      return dttParams;
    }

    /**
     * @brief Parse transform direction.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform direction.
     */
    [[nodiscard]] afft::Direction parseDirection(mx::StructArrayCref transformParamsStruct)
    {
      auto directionArray = transformParamsStruct.getField(0, "direction");

      if (!directionArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform direction");
      }

      if (!directionArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform direction must be a char array");
      }

      std::u16string_view directionStrView{mx::CharArrayCref{*directionArray}};

      if (directionStrView == u"forward")
      {
        return afft::Direction::forward;
      }
      else if (directionStrView == u"backward" || directionStrView == u"inverse")
      {
        return afft::Direction::backward;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform direction");
      }
    }

    /**
     * @brief Parse transform precision.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform precision.
     */
    [[nodiscard]] afft::PrecisionTriad parsePrecision(mx::StructArrayCref transformParamsStruct)
    {
      auto precisionArray = transformParamsStruct.getField(0, "precision");

      if (!precisionArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform precision");
      }

      if (!precisionArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform precision must be a char array");
      }

      std::u16string_view precisionStrView{mx::CharArrayCref{*precisionArray}};

      if (precisionStrView == u"f32" || precisionStrView == u"single")
      {
        return afft::makePrecision<float>();
      }
      else if (precisionStrView == u"f64" || precisionStrView == u"double")
      {
        return afft::makePrecision<double>();
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform precision");
      }
    }

    /**
     * @brief Parse transform shape.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform shape.
     */
    [[nodiscard]] afft::View<afft::Size> parseShape(mx::StructArrayCref transformParamsStruct)
    {
      auto shapeArray = transformParamsStruct.getField(0, "shape");

      if (!shapeArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform shape");
      }

      if (!shapeArray->isNumeric())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform shape must be a numeric array");
      }

      const std::size_t shapeRank = shapeArray->getSize();

      if (shapeRank > afft::maxDimCount)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform shape exceeds maximum dimension count");
      }

      mx::visit(*shapeArray, [&](auto&& typedShapeArray)
      {
        if constexpr (mx::isRealNumeric<decltype(typedShapeArray[0])>)
        {
          for (std::size_t i = 0; i < shapeRank; ++i)
          {
            mShape[shapeRank - 1 - i] = static_cast<afft::Size>(typedShapeArray[i]);
          }
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "transform shape must be a numeric array");
        }
      });

      return afft::View<afft::Size>{mShape, shapeRank};
    }

    /**
     * @brief Parse transform axes.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform axes.
     */
    [[nodiscard]] afft::View<afft::Axis> parseAxes(mx::StructArrayCref transformParamsStruct)
    {
      auto axesArray = transformParamsStruct.getField(0, "axes");

      if (!axesArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform axes");
      }

      if (!axesArray->isNumeric())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform axes must be a numeric array");
      }

      mTransformRank = axesArray->getSize();

      if (mTransformRank > afft::maxDimCount)
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform axes exceeds maximum dimension count");
      }

      mx::visit(*axesArray, [&](auto&& typedAxesArray)
      {
        if constexpr (mx::isRealNumeric<decltype(typedAxesArray[0])>)
        {
          for (std::size_t i = 0; i < mTransformRank; ++i)
          {
            mAxes[mTransformRank - 1 - i] = static_cast<afft::Axis>(typedAxesArray[i]);
          }
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "transform axes must be a numeric array");
        }
      });

      return afft::View<afft::Axis>{mAxes, mTransformRank};
    }

    /**
     * @brief Parse transform normalization.
     * @param transformParamsStruct Transform parameters struct.
     * @return Transform normalization.
     */
    [[nodiscard]] afft::Normalization parseNormalization(mx::StructArrayCref transformParamsStruct)
    {
      auto normalizationArray = transformParamsStruct.getField(0, "normalization");

      if (!normalizationArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform normalization");
      }

      if (!normalizationArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform normalization must be a char array");
      }

      std::u16string_view normalizationStrView{mx::CharArrayCref{*normalizationArray}};

      if (normalizationStrView == u"none")
      {
        return afft::Normalization::none;
      }
      else if (normalizationStrView == u"unitary")
      {
        return afft::Normalization::unitary;
      }
      else if (normalizationStrView == u"ortho" || normalizationStrView == u"orthogonal")
      {
        return afft::Normalization::orthogonal;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform normalization");
      }
    }

    /**
     * @brief Parse DFT type.
     * @param transformParamsStruct Transform parameters struct.
     * @return DFT type.
     */
    [[nodiscard]] afft::dft::Type parseDftType(mx::StructArrayCref transformParamsStruct)
    {
      auto typeArray = transformParamsStruct.getField(0, "type");

      if (!typeArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform type");
      }

      if (!typeArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform type must be a char array");
      }

      std::u16string_view typeStrView{mx::CharArrayCref{*typeArray}};

      if (typeStrView == u"complexToComplex" || typeStrView == u"c2c")
      {
        return afft::dft::Type::complexToComplex;
      }
      else if (typeStrView == u"realToComplex" || typeStrView == u"r2c")
      {
        return afft::dft::Type::realToComplex;
      }
      else if (typeStrView == u"complexToReal" || typeStrView == u"c2r")
      {
        return afft::dft::Type::complexToReal;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }

    /**
     * @brief Parse DHT type.
     * @param transformParamsStruct Transform parameters struct.
     * @return DHT type.
     */
    [[nodiscard]] afft::dht::Type parseDhtType(mx::StructArrayCref transformParamsStruct)
    {
      auto typeArray = transformParamsStruct.getField(0, "type");

      if (!typeArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform type");
      }

      if (!typeArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform type must be a char array");
      }

      std::u16string_view typeStrView{mx::CharArrayCref{*typeArray}};

      if (typeStrView == u"separable")
      {
        return afft::dht::Type::separable;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
      }
    }

    /**
     * @brief Parse DTT types.
     * @param transformParamsStruct Transform parameters struct.
     * @return DTT types.
     */
    [[nodiscard]] afft::View<afft::dtt::Type> parseDttTypes(mx::StructArrayCref transformParamsStruct)
    {
      auto parseSingleDttType = [](std::u16string_view typeStrView)
      {
        if (typeStrView.size() != 3 && typeStrView.size() != 4)
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
        }

        bool isDct{};

        if (typeStrView.substr(0, 3) == u"dct")
        {
          isDct = true;
        }
        else if (typeStrView.substr(0, 3) == u"dst")
        {
          isDct = false;
        }
        else
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform type");
        }

        if (typeStrView.size() == 3)
        {
          return (isDct) ? afft::dtt::Type::dct : afft::dtt::Type::dst;
        }
        
        if (auto numStrView = typeStrView.substr(3, 1); numStrView == u"1")
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
      };

      auto typeArray = transformParamsStruct.getField(0, "type");

      if (!typeArray)
      {
        throw mx::Exception("afft:planCreate:missingArgument", "missing transform type");
      }

      if (!typeArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "transform type must be a char array");
      }

      mx::CharArrayCref typeCharArray{*typeArray};

      if (typeCharArray.isSingleString())
      {
        std::fill_n(mDttTypes, mTransformRank, parseSingleDttType(std::u16string_view{typeCharArray}));
      }
      else if (typeCharArray.getDimM() == mTransformRank)
      {
        const std::size_t n = typeCharArray.getDimN();

        for (std::size_t i{}; i < mTransformRank; ++i)
        {
          mDttTypes[i] = parseSingleDttType(std::u16string_view{typeCharArray.getData() + i * n});
        }
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid dtt type array size");
      }

      return afft::View<afft::dtt::Type>{mDttTypes, mTransformRank};
    }

    afft::Size        mShape[afft::maxDimCount]{};    ///< Shape of the transform.
    std::size_t       mTransformRank{};               ///< Rank of the transform.
    afft::Axis        mAxes[afft::maxDimCount]{};     ///< Axes of the transform.
    union
    {
      afft::dtt::Type mDttTypes[afft::maxDimCount]{}; ///< DTT types of the transform.
    };
};

/// @brief Target parameters parser.
class TargetParametersParser
{
  public:
    /**
     * @brief Parse target parameters.
     * @param targetParamsStruct Target parameters struct.
     * @return Target parameters.
     */
    [[nodiscard]] afft::TargetParametersVariant operator()(mx::StructArrayCref targetParamsStruct)
    {
      const afft::Target target = parseTarget(targetParamsStruct);

      switch (target)
      {
      case afft::Target::cpu:
        return parseCpuTargetParameters(targetParamsStruct);
      case afft::Target::cuda:
#     if defined(AFFT_ENABLE_CUDA) && defined(MATLABW_ENABLE_GPU)
        return parseCudaTargetParameters(targetParamsStruct);
#     else
        throw mx::Exception("afft:planCreate:invalidArgument", "GPU target is disabled");
#     endif
      default:
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid target");        
      }
    }
  private:
    /**
     * @brief Parse target parameter.
     * @param targetParamsStruct Target parameters struct.
     * @return Target. If target is not specified, then cpu target is returned.
     */
    [[nodiscard]] afft::Target parseTarget(mx::StructArrayCref targetParamsStruct)
    {
      auto targetArray = targetParamsStruct.getField(0, "target");

      if (!targetArray)
      {
        return afft::Target::cpu;
      }

      if (!targetArray->isChar())
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "target must be a char array");
      }

      std::u16string_view targetStrView{mx::CharArrayCref{*targetArray}};

      if (targetStrView == u"cpu")
      {
        return afft::Target::cpu;
      }
      else if (targetStrView == u"cuda" || targetStrView == u"gpu")
      {
        return afft::Target::cuda;
      }
      else
      {
        throw mx::Exception("afft:planCreate:invalidArgument", "invalid target");
      }
    }

    /**
     * @brief Parse cpu target parameters.
     * @param targetParamsStruct Target parameters struct.
     * @return Cpu target parameters.
     */
    [[nodiscard]] afft::cpu::Parameters parseCpuTargetParameters(mx::StructArrayCref targetParamsStruct)
    {
      afft::cpu::Parameters cpuParams{};

      auto threadLimitArray = targetParamsStruct.getField(0, "threadLimit");

      if (threadLimitArray)
      {
        if (!threadLimitArray->isScalar() || !threadLimitArray->isNumeric())
        {
          throw mx::Exception("afft:planCreate:invalidArgument", "threads must be a scalar uint64");
        }

        cpuParams.threadLimit = mx::visit(*threadLimitArray, [](auto&& array) -> unsigned
        {
          if constexpr (mx::isRealNumeric<decltype(array[0])>)
          {
            const auto value = array[0];

            if (value < decltype(value){})
            {
              throw mx::Exception("afft:planCreate:invalidArgument", "thread limit must be a non-negative number");
            }

            return static_cast<unsigned>(value);
          }
          else
          {
            throw mx::Exception("afft:planCreate:invalidArgument", "thread limit must be a non-negative number");
          }
        });
      }

      return cpuParams;
    }

# ifdef MATLABW_ENABLE_GPU
    /**
     * @brief Parse cuda target parameters.
     * @param targetParamsStruct Target parameters struct.
     * @return Cuda target parameters.
     */
    [[nodiscard]] afft::cuda::Parameters parseCudaTargetParameters(mx::StructArrayCref targetParamsStruct)
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
# ifdef MATLABW_ENABLE_GPU
    int cudaDevice{}; ///< CUDA device.
# endif
  };
};

/**
 * @brief Create a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the plan data.
 * @param rhs Right-hand side array of size 2.
 *            * rhs[0] holds the transform parameters as a struct,
 *            * rhs[1] holds the target parameters as a struct.
 */
void planCreate(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 2)
  {
    throw mx::Exception("afft:planCreate:invalidInputCount", "invalid number of input arguments, expected 2");
  }

  if (lhs.size() != 1)
  {
    throw mx::Exception("afft:planCreate:invalidOutputCount", "invalid number of output arguments, expected 1");
  }

  if (!rhs[0].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "transform parameters must be a struct");
  }

  if (!rhs[1].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "target parameters must be a struct");
  }

  TransformParametersParser transformParamsParser{};
  TargetParametersParser    targetParamsParser{};

  const afft::Description desc{transformParamsParser(mx::StructArrayCref{rhs[0]}),
                               targetParamsParser(mx::StructArrayCref{rhs[1]})};

  auto planIt = planCache.find(desc);

  if (planIt == planCache.end())
  {
    planIt = planCache.emplace(std::cref(desc));
  }

  lhs[0] = mx::makeUninitNumericArray<std::uint8_t>({{sizeof(PlanData)}});

  PlanData* planData = static_cast<PlanData*>(lhs[0].getData());
  planData->plan       = planIt->get();
  planData->cacheEpoch = planCacheEpoch;
}

/**
 * @brief Execute a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 2.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>,
 *            * rhs[1] holds the input array.
 */
void planExecute(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 2)
  {
    throw mx::Exception{"afft:invalidInputCount", "invalid number of input arguments, expected 2"};
  }

  if (lhs.size() != 1)
  {
    throw mx::Exception{"afft:invalidOutputCount", "invalid number of output arguments, expected 1"};
  }

  if (!rhs[0].isUint8() || rhs[0].getSize() != sizeof(PlanData))
  {
    throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid plan data"};
  }

  const PlanData* planData = static_cast<const PlanData*>(rhs[0].getData());

  if (planData->plan == nullptr)
  {
    throw mx::Exception{"afft:Plan:execute:invalidPlan", "invalid plan"};
  }

  if (planData->cacheEpoch != planCacheEpoch)
  {
    throw mx::Exception{"afft:Plan:execute:invalidPlan", "the plan was created in a different plan cache epoch"};
  }

  afft::Plan* plan = planData->plan;

  const auto [srcPrec, dstPrec] = plan->getSrcDstPrecision();
  const auto [srcCmpl, dstCmpl] = plan->getSrcDstComplexity();

  switch (rhs[0].getClassId())
  {
  case mx::ClassId::single:
    if (srcPrec != afft::Precision::f32)
    {
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array precision"};
    }
    if (rhs[0].isComplex() && srcCmpl != afft::Complexity::complex)
    {
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array complexity"};
    }
    break;
  case mx::ClassId::_double:
    if (srcPrec != afft::Precision::f64)
    {
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array precision"};
    }
    if (rhs[0].isComplex() && srcCmpl != afft::Complexity::complex)
    {
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array complexity"};
    }
    break;
  default:
    throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array type"};
  }

  mx::ClassId dstClassId{};

  switch (dstPrec)
  {
  case afft::Precision::f32:
    dstClassId = mx::ClassId::single;
    break;
  case afft::Precision::f64:
    dstClassId = mx::ClassId::_double;
    break;
  default:
    throw mx::Exception{"afft:Plan:execute:internalError", "invalid destination precision"};
  }

  lhs[0] = mx::makeUninitNumericArray(dims,
                                      dstClassId,
                                      (srcCmpl == afft::Complexity::complex) ? mx::Complexity::complex : mx::Complexity::real);

  // TODO: check input array type and size

  planData->plan->executeUnsafe(rhs[1].getData(), lhs[0].getData());
}

/**
 * @brief Get the transform parameters of a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the transform parameters as a scalar StructArray.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>.
 */
void planGetTransformParameters(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  
}

/**
 * @brief Get the target parameters of a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the target parameters as a scalar StructArray.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>.
 */
void planGetTargetParameters(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  
}
