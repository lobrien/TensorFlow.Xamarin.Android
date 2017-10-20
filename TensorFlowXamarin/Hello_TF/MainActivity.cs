using Android.App;
using Android.Widget;
using Android.OS;
using Org.Tensorflow.Contrib.Android;
using Android.Graphics;
using Android.Content.Res;
using System.Linq;
using System;
using Org.Tensorflow;
using System.IO;

namespace Hello_TF
{
	[Activity(Label = "Hello_TF", MainLauncher = true, Icon = "@mipmap/icon")]
	public class MainActivity : Activity
	{
		int count = 1;

		long byteHash(byte[] contents)
		{
			long hash = 0;
			for (int i = 0; i < contents.Length; i++)
			{
				hash += contents[i];
			}
			return hash;
		}

		protected override void OnCreate(Bundle savedInstanceState)
		{
			base.OnCreate(savedInstanceState);

			// Set our view from the "main" layout resource
			SetContentView(Resource.Layout.Main);

			// Get our button from the layout resource,
			// and attach an event to it
			Button button = FindViewById<Button>(Resource.Id.myButton);

			button.Click += delegate { button.Text = $"{count++} clicks!"; };

			try
			{
				TensorFlowInferenceInterface tfi = new TensorFlowInferenceInterface(Assets, "file:///android_asset/TF_LSTM_Inference.pb");
				float[] inputSeaLevels = new float[] {
		  4.92F, 2.022F, -0.206F, 2.355F, 4.08F, 1.828F, -0.005F, 2.83F,
				  4.966F, 2.715F, -0.073F, 1.69F, 3.958F, 2.5F, 0.201F, 2.075F, 4.754F, 3.475F,
				  0.345F, 0.954F, 3.665F, 3.165F, 0.562F, 1.285F, 4.415F, 4.083F, 0.83F, 0.327F,
				  3.304F, 3.589F, 0.976F, 0.707F, 3.989F, 4.37F, 1.375F, 0.039F, 2.863F, 3.715F,
				  1.525F, 0.507F, 3.403F, 4.38F, 2.06F, 0.097F, 2.251F, 3.671F, 2.223F, 0.579F,
				  2.605F, 4.25F, 2.791F, 0.306F, 1.489F, 3.575F, 2.907F, 0.759F, 1.741F, 4.018F,
				  3.381F, 0.589F, 0.796F, 3.382F, 3.441F, 1.103F, 1.099F, 3.566F, 3.75F, 1.09F,
				  0.393F, 2.906F, 3.831F, 1.77F, 0.816F, 2.739F, 3.922F, 1.891F, 0.285F, 2.02F,
				  4.083F, 2.727F, 0.833F, 1.593F, 3.863F, 2.824F, 0.408F, 0.906F, 4.071F, 3.738F,
				  1.149F, 0.511F, 3.431F, 3.626F, 0.863F, -0.004F, 3.57F, 4.568F, 1.906F,
				  -0.079F, 2.489F, 4.115F, 1.777F, -0.396F, 2.456F, 5.046F, 3.106F, -0.04F,
				  1.113F, 4.168F, 2.998F, -0.246F, 0.941F, 5.009F, 4.425F, 0.516F, -0.267F,
				  3.695F, 4.092F, 0.405F, -0.384F, 4.32F, 5.429F, 1.541F, -1.036F, 2.628F,
				  4.711F, 1.479F, -0.977F, 3.043F, 5.82F, 2.856F, -0.932F, 1.249F, 4.713F,
				  2.828F, -0.734F, 1.438F, 5.546F, 4.182F, -0.159F, -0.109F, 4.137F, 4.03F,
				  0.09F, 0.042F, 4.693F, 5.075F, 0.921F, -0.925F, 3.14F, 4.67F, 1.227F, -0.599F,
				  3.458F, 5.274F, 2.089F, -0.893F, 1.945F, 4.618F, 2.489F, -0.336F, 2.106F,
				  4.828F, 3.178F, -0.192F, 0.816F, 4.026F, 3.61F, 0.491F, 0.961F, 3.972F, 3.93F,
				  0.717F, 0.057F, 3.155F, 4.273F, 1.457F, 0.369F, 2.96F, 4.147F, 1.559F, -0.088F,
				  2.226F, 4.349F, 2.398F, 0.485F, 1.964F, 3.861F, 2.329F, 0.335F, 1.357F, 3.972F,
				  3.286F, 1.092F, 1.083F, 3.277F, 3.011F, 0.981F, 0.632F, 3.375F, 3.987F
	  };
				String INPUT_ARGUMENT_NAME = "lstm_1_input";
				String OUTPUT_VARIABLE_NAME = "output_node0";
				int OUTPUT_SIZE = 100;

				tfi.Feed(INPUT_ARGUMENT_NAME, inputSeaLevels, inputSeaLevels.Length, 1, 1);
				tfi.Run(new String[] { OUTPUT_VARIABLE_NAME });
				float[] predictions = new float[OUTPUT_SIZE];
				tfi.Fetch(OUTPUT_VARIABLE_NAME, predictions);
				button.Text = predictions[0].ToString();
			}
			catch (Exception x)
			{
				Console.WriteLine(x);
			}

			// Seriously, it's this easy?
			using (var ic = new InceptionClassifier(BaseContext.Assets))
			{
				var results = ic.Recognize(BitmapCreate(BaseContext.Assets, "husky.png"));
				var top = results.First();
				//				TextView label = FindViewById<TextView>(Resource.Id.)
			}
		}


		static Bitmap BitmapCreate(AssetManager assetManager, string assetName)
		{
			using (var str = assetManager.Open(assetName))
			{
				return BitmapFactory.DecodeStream(str);
			}
		}
	}
}

