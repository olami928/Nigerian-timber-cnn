const FIXED_ONE = 1 << 22;

export function calculateSharpness(imageData) {
  const { data, width, height } = imageData;
  const gray = new Uint8Array(width * height);
  for (let pixel = 0, offset = 0; pixel < gray.length; pixel += 1, offset += 4) {
    gray[pixel] = (
      4899 * data[offset]
      + 9617 * data[offset + 1]
      + 1868 * data[offset + 2]
      + 8192
    ) >> 14;
  }
  if (width < 2 || height < 2) return 0;
  let sum = 0;
  let sumSquares = 0;
  for (let y = 0; y < height; y += 1) {
    const up = y === 0 ? 1 : y - 1;
    const down = y === height - 1 ? height - 2 : y + 1;
    for (let x = 0; x < width; x += 1) {
      const left = x === 0 ? 1 : x - 1;
      const right = x === width - 1 ? width - 2 : x + 1;
      const index = y * width + x;
      const value = gray[up * width + x]
        + gray[down * width + x]
        + gray[y * width + left]
        + gray[y * width + right]
        - 4 * gray[index];
      sum += value;
      sumSquares += value * value;
    }
  }
  const mean = sum / gray.length;
  return Math.max(0, sumSquares / gray.length - mean * mean);
}

function coefficients(inputLength, outputLength) {
  const scale = inputLength / outputLength;
  const filterScale = Math.max(scale, 1);
  const support = filterScale;
  return Array.from({ length: outputLength }, (_, outputIndex) => {
    const center = (outputIndex + 0.5) * scale;
    const first = Math.max(0, Math.trunc(center - support + 0.5));
    const last = Math.min(inputLength, Math.trunc(center + support + 0.5));
    const weights = [];
    for (let sourceIndex = first; sourceIndex < last; sourceIndex += 1) {
      weights.push(Math.max(
        0,
        1 - Math.abs((sourceIndex + 0.5 - center) / filterScale),
      ));
    }
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return {
      first,
      weights: weights.map((weight) => Math.floor((weight / total) * FIXED_ONE + 0.5)),
    };
  });
}

export function resizePatch(imageData, sourceX, sourceY, sourceWidth, sourceHeight, outputSize = 112) {
  const { data, width: imageWidth } = imageData;
  const horizontalWeights = coefficients(sourceWidth, outputSize);
  const verticalWeights = coefficients(sourceHeight, outputSize);
  const intermediate = new Uint8Array(outputSize * sourceWidth * 3);
  for (let y = 0; y < outputSize; y += 1) {
    const { first, weights } = verticalWeights[y];
    for (let x = 0; x < sourceWidth; x += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        let value = 0;
        for (let tap = 0; tap < weights.length; tap += 1) {
          const sourceOffset = ((sourceY + first + tap) * imageWidth + sourceX + x) * 4;
          value += data[sourceOffset + channel] * weights[tap];
        }
        intermediate[(y * sourceWidth + x) * 3 + channel] = Math.min(
          255,
          Math.max(0, (value + (FIXED_ONE >> 1)) >> 22),
        );
      }
    }
  }

  const output = new Uint8Array(outputSize * outputSize * 3);
  for (let y = 0; y < outputSize; y += 1) {
    for (let x = 0; x < outputSize; x += 1) {
      const { first, weights } = horizontalWeights[x];
      for (let channel = 0; channel < 3; channel += 1) {
        let value = 0;
        for (let tap = 0; tap < weights.length; tap += 1) {
          value += intermediate[(y * sourceWidth + first + tap) * 3 + channel]
            * weights[tap];
        }
        output[(y * outputSize + x) * 3 + channel] = Math.min(
          255,
          Math.max(0, (value + (FIXED_ONE >> 1)) >> 22),
        );
      }
    }
  }
  return output;
}

export function makeTensorData(imageData, normalization, gridRows = 6, gridCols = 8, patchSize = 112) {
  const patchWidth = Math.floor(imageData.width / gridCols);
  const patchHeight = Math.floor(imageData.height / gridRows);
  const values = new Float32Array(gridRows * gridCols * 3 * patchSize * patchSize);
  const planeSize = patchSize * patchSize;
  for (let row = 0; row < gridRows; row += 1) {
    for (let column = 0; column < gridCols; column += 1) {
      const patchIndex = row * gridCols + column;
      const pixels = resizePatch(
        imageData,
        column * patchWidth,
        row * patchHeight,
        patchWidth,
        patchHeight,
        patchSize,
      );
      for (let pixel = 0; pixel < planeSize; pixel += 1) {
        for (let channel = 0; channel < 3; channel += 1) {
          const value = Math.fround(pixels[pixel * 3 + channel] / 255);
          const mean = Math.fround(normalization.mean[channel]);
          const std = Math.fround(normalization.std[channel]);
          values[(patchIndex * 3 + channel) * planeSize + pixel] =
            Math.fround(Math.fround(value - mean) / std);
        }
      }
    }
  }
  return values;
}
