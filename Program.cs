using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnx;
using System.Runtime.InteropServices;
using System.Text;
using Vanara.PInvoke;

#pragma warning disable CA1416 // Validate platform compatibility

namespace npu
{
	public class Program
	{
		private const string npuProgramJson = @"
		{
			""irVersion"": 7,
			""producerName"": ""Jim"",
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
				},
				{
					""name"": ""W"",
					""type"": { ""tensorType"": { ""elemType"": 1, ""shape"": { ""dim"": [ { ""dimValue"": 4 } , { ""dimValue"": 4 }  ] } } }
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
			EnumerateAdapters();

			// load the model

			var model = ModelProto.Parser.ParseJson(npuProgramJson);

			// create the input vectors (by manually copying the values into the input)

			const int batchSize = 5;
			var inputTensor = new DenseTensor<float>(new[] { batchSize, 4, 1 });

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

			// create the input matrix (by mapping the matrix array into a Memory span)

			float[] matrix2 = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
			var matrixTensor = new DenseTensor<float>(new Memory<float>(matrix2), new[] { 4, 4 });

			// create the session

			var options = new SessionOptions();
			options.AppendExecutionProvider_DML(); // DirectML auto-selects NPU if available
			//options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
			using var session = new InferenceSession(model.ToByteArray(), options);

			// set the inputs

			var inputs = new List<NamedOnnxValue>
			{
				NamedOnnxValue.CreateFromTensor("X", inputTensor),
				NamedOnnxValue.CreateFromTensor("W", matrixTensor)
			};

			// run the model

			using var results = session.Run(inputs);

			// show the outputs

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

		private static Guid iid_IDXCoreAdapterFactory = new Guid("78ee5945-c36e-4b13-a669-005dd11c0f06");
		private static Guid IID_IDXCoreAdapterList = new Guid("526c7776-40e9-459b-b711-f32ad76dfc28");
		private static Guid IID_IDxCoreAdapter = new Guid("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e");

		private static Guid DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE = new Guid("248e2800-a793-4724-abaa-23a6de1be090");
		private static Guid DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_ML = new Guid("b71b0d41-1088-422f-a27c-0250b7d3a988");

		private static void EnumerateAdapters()
		{
			HRESULT err;

			err = DXCore.DXCoreCreateAdapterFactory(iid_IDXCoreAdapterFactory, out var opaqueFactory);
			if (err != HRESULT.S_OK || opaqueFactory == null)
			{
				Console.WriteLine("Failed to create DXCoreAdapterFactory.");
				return;
			}

			var factory = (DXCore.IDXCoreAdapterFactory)opaqueFactory;

			//Guid[] attributes = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
			Guid[] attributes = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_ML };

			err = factory.CreateAdapterList(attributes.Length, attributes, IID_IDXCoreAdapterList, out var opaqueAdapterList);
			if (err != HRESULT.S_OK || opaqueAdapterList == null)
			{
				Console.WriteLine("Failed to create DXCoreAdapterList.");
				return;
			}

			var adapterList = (DXCore.IDXCoreAdapterList)opaqueAdapterList;
			uint count = adapterList.GetAdapterCount();
			Console.WriteLine($"Found {count} compute-capable adapters:");

			for (uint i = 0; i < count; i++)
			{
				err = adapterList.GetAdapter(i, IID_IDxCoreAdapter, out var opaqueAdapter);
				if (err != HRESULT.S_OK || opaqueAdapter == null)
				{
					Console.WriteLine($"Failed to get DXCoreAdapter {i}.");
					continue;
				}
				var adapter = (DXCore.IDXCoreAdapter)opaqueAdapter;
				Console.WriteLine($"\nAdapter {i}:");

				PrintProperty(adapter, DXCore.DXCoreAdapterProperty.HardwareID, "Hardware ID");
				PrintProperty(adapter, DXCore.DXCoreAdapterProperty.DriverVersion, "Driver Version");
				PrintProperty(adapter, DXCore.DXCoreAdapterProperty.DriverDescription, "Driver Description");
				PrintProperty(adapter, DXCore.DXCoreAdapterProperty.IsHardware, "Is Hardware");
				PrintProperty(adapter, DXCore.DXCoreAdapterProperty.InstanceLuid, "LUID");
			}
			Console.WriteLine();
		}

		private static void PrintProperty(DXCore.IDXCoreAdapter adapter, DXCore.DXCoreAdapterProperty prop, string label)
		{
			if (!adapter.IsPropertySupported(prop)) return;

			adapter.GetPropertySize(prop, out ulong size);
			var bytes = new byte[size];

			IntPtr buffer = Marshal.AllocHGlobal((int)size);
			adapter.GetProperty(prop, size, buffer);
			Marshal.Copy(buffer, bytes, 0, (int)size);
			Marshal.FreeHGlobal(buffer);

			string value = "(unknown)";
			switch (prop)
			{
				case DXCore.DXCoreAdapterProperty.IsHardware:
					value = Marshal.ReadByte(buffer) != 0 ? "Yes" : "No";
					break;
				case DXCore.DXCoreAdapterProperty.HardwareID:
					value = $"{BitConverter.ToUInt32(bytes, 0)} {BitConverter.ToUInt32(bytes, 4)} {BitConverter.ToUInt32(bytes, 8)} {BitConverter.ToUInt32(bytes, 12)} ({BitConverter.ToUInt32(bytes, 0):X8} {BitConverter.ToUInt32(bytes, 4):X8} {BitConverter.ToUInt32(bytes, 8):X8} {BitConverter.ToUInt32(bytes, 12):X8})";
					break;
				case DXCore.DXCoreAdapterProperty.DriverVersion:
					value = $"{BitConverter.ToUInt16(bytes, 6)}.{BitConverter.ToUInt16(bytes, 4)}.{BitConverter.ToUInt16(bytes, 2)}.{BitConverter.ToUInt16(bytes)}";
					break;
				case DXCore.DXCoreAdapterProperty.InstanceLuid:
					value = $"{BitConverter.ToUInt32(bytes, 0):X8} {BitConverter.ToUInt32(bytes, 4):X8}";
					break;
				case DXCore.DXCoreAdapterProperty.DriverDescription:
					value = Encoding.UTF8.GetString(bytes);
					break;
			}

			Console.WriteLine($"  {label}: {value}");
		}
	}
}