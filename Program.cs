using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnx;

namespace npu
{
	public class Program
	{
		const string npuProgramJson = @"
        {
              ""irVersion"": 7,
              ""producerName"": ""Copilot"",
             ""opsetImport"": [
                {
                  ""domain"": """",
                  ""version"": 13
                }
              ],
              ""graph"": {
                ""name"": ""BatchedMatMul"",
                ""input"": [
                  {
                    ""name"": ""X"",
                    ""type"": {
                      ""tensorType"": {
                        ""elemType"": ""1"",
                        ""shape"": {
                          ""dim"": [
                            { ""dimParam"": ""batch"" },
                            { ""dimValue"": 4 },
                            { ""dimValue"": 1 }
                          ]
                        }
                      }
                    }
                  }
                ],
                ""output"": [
                  {
                    ""name"": ""Y"",
                    ""type"": {
                      ""tensorType"": {
                        ""elemType"": ""1"",
                        ""shape"": {
                          ""dim"": [
                            { ""dimParam"": ""batch"" },
                            { ""dimValue"": 4 },
                            { ""dimValue"": 1 }
                          ]
                        }
                      }
                    }
                  }
                ],
                ""initializer"": [
                  {
                    ""name"": ""W"",
                    ""dims"": [4, 4],
                    ""dataType"": 1,
                    ""floatData"": [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0]
                  }
                ],
                ""node"": [
                  {
                    ""input"": [""W"", ""X""],
                    ""output"": [""Y""],
                    ""name"": ""MatMulNode"",
                    ""opType"": ""MatMul""
                  }
                ]
              }
            }
        ";

		static void Main(string[] args)
		{
			var model = ModelProto.Parser.ParseJson(npuProgramJson);

			const int batchSize = 5;
			var inputTensor = new DenseTensor<float>(new[] { batchSize, 4, 1 });

			// Fill the tensor manually
			float[,] vectors = new float[,]
			{
				{ 1, 2, 3, 4 },
				{ 5, 6, 7, 8 },
				{ 9, 10, 11, 12 },
				{ 13, 14, 15, 16 },
				{ 17, 18, 19, 20 }
			};

			for (int b = 0; b < batchSize; b++)
			{
				for (int i = 0; i < 4; i++)
				{
					inputTensor[b, i, 0] = vectors[b, i];
				}
			}

			var options = new SessionOptions();
			options.AppendExecutionProvider_DML(); // DirectML auto-selects NPU if available

			using var session = new InferenceSession(model.ToByteArray(), options);

			var inputs = new List<NamedOnnxValue>
			{
				NamedOnnxValue.CreateFromTensor("X", inputTensor)
			};

			using var results = session.Run(inputs);

			var outputTensor = results[0].AsTensor<float>();
			var outputArray = outputTensor.ToArray();

			for (int i = 0; i < batchSize; i++)
			{
				Console.Write($"Result {i + 1}: ");
				for (int j = 0; j < 4; j++)
				{
					Console.Write($"{outputArray[i * 4 + j]} ");
				}
				Console.WriteLine();
			}
		}
	}
}
