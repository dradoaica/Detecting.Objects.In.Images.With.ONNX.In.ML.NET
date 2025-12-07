using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.Common;
using ObjectDetection.DataStructures;

namespace ObjectDetection;

/// <summary>Scorer class for the ONNX model.</summary>
internal class OnnxModelScorer
{
    private readonly string imagesFolder;
    private readonly MLContext mlContext;
    private readonly string modelLocation;

    /// <summary>Initializes a new instance of the <see cref="OnnxModelScorer" /> class.</summary>
    /// <param name="imagesFolder">The path to the images folder.</param>
    /// <param name="modelLocation">The path to the ONNX model file.</param>
    /// <param name="mlContext">The MLContext instance.</param>
    public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.mlContext = mlContext;
    }

    /// <summary>Scores the data using the ONNX model.</summary>
    /// <param name="data">The input data view.</param>
    /// <returns>An enumerable of float arrays representing the probabilities.</returns>
    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }

    private ITransformer LoadModel(string location)
    {
        ConsoleHelper.ConsoleWriteHeader("Read model");
        Console.WriteLine($"Model location: {location}");
        Console.WriteLine(
            $"Default parameters: image size=({ImageNetSettings.ImageWidth},{ImageNetSettings.ImageHeight})"
        );
        // Create IDataView from empty list to obtain input data schema
        var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
        // Define scoring pipeline
        var pipeline = mlContext.Transforms.LoadImages("image", "", nameof(ImageNetData.ImagePath))
            .Append(
                mlContext.Transforms.ResizeImages(
                    "image",
                    ImageNetSettings.ImageWidth,
                    ImageNetSettings.ImageHeight,
                    "image"
                )
            )
            .Append(mlContext.Transforms.ExtractPixels("image"))
            .Append(
                mlContext.Transforms.ApplyOnnxModel(
                    modelFile: location,
                    outputColumnNames: new[]
                    {
                        TinyYoloModelSettings.ModelOutput,
                    },
                    inputColumnNames: new[]
                    {
                        TinyYoloModelSettings.ModelInput,
                    }
                )
            );
        // Fit scoring pipeline
        var model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {imagesFolder}");
        Console.WriteLine("");
        ConsoleHelper.ConsoleWriteHeader("=====Identify the objects in the images=====");
        Console.WriteLine("");
        var scoredData = model.Transform(testData);
        var probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

        return probabilities;
    }
}
